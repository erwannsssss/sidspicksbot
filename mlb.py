"""
MLB
---
Data: the free MLB Stats API (statsapi.mlb.com) — every game since 2021 with the final score and
each team's starting pitcher, plus today's probable starters.

Model: a team rating that moves after every game (carried over between seasons with a pull back
toward average), the starting pitcher's quality (his previous-season FIP, built from strikeouts,
walks and home runs, plus runs his team allowed over his last 10 starts), each team's recent run differential and days of rest, plus home field. It's tested on the
most recent games before it's used.
"""
import math
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import common as c

API = "https://statsapi.mlb.com/api/v1"
CACHE = os.path.join(c.STATE_DIR, "mlb_history.csv.gz")
META = os.path.join(c.STATE_DIR, "mlb_meta.json")
FIRST_SEASON = 2021
SERIES = {"KXMLBGAME": "MLB"}
FEATURES = ["home_field", "rating", "starting_pitcher", "run_diff_form", "rest", "pitcher_fip"]
PITCHERS = os.path.join(c.STATE_DIR, "mlb_pitchers.json")
LEAGUE_FIP = 4.2
LEAGUE_RA = 4.5
# Approximate run factors by home team (1.00 = neutral). Rough public estimates, used only for totals.
PARK = {"COL": 1.28, "CIN": 1.08, "BOS": 1.06, "PHI": 1.03, "KC": 1.03, "AZ": 1.03, "ARI": 1.03, "CHC": 1.02,
        "NYY": 1.02, "ATH": 1.02, "BAL": 1.00, "LAA": 1.01, "ATL": 1.01, "WSH": 1.00, "TOR": 1.00, "CWS": 1.00,
        "MIN": 0.99, "HOU": 0.99, "TEX": 0.98, "MIL": 0.98, "LAD": 0.98, "CLE": 0.97, "DET": 0.97, "STL": 0.97,
        "PIT": 0.97, "NYM": 0.96, "TB": 0.96, "SD": 0.95, "SF": 0.95, "MIA": 0.95, "SEA": 0.92}
ROOF = {"TB", "TOR", "MIA", "HOU", "AZ", "ARI", "MIL", "SEA", "TEX"}  # domes and retractable roofs


def _schedule(start, end):
    r = c.get(f"{API}/schedule", params={"sportId": 1, "startDate": start, "endDate": end,
                                           "gameType": "R,F,D,L,W", "hydrate": "probablePitcher"}, timeout=120)
    rows = []
    for d in r.json().get("dates", []):
        for g in d.get("games", []):
            h, a = g["teams"]["home"], g["teams"]["away"]
            rows.append({
                "pk": g["gamePk"], "date": g["gameDate"], "status": g["status"].get("detailedState", ""),
                "home_id": h["team"]["id"], "away_id": a["team"]["id"],
                "home": h["team"]["name"], "away": a["team"]["name"],
                "hs": h.get("score"), "as": a.get("score"),
                "hp": (h.get("probablePitcher") or {}).get("id"), "ap": (a.get("probablePitcher") or {}).get("id"),
                "hpn": (h.get("probablePitcher") or {}).get("fullName"),
                "apn": (a.get("probablePitcher") or {}).get("fullName"),
            })
    return rows


def load_history():
    cache = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame()
    if len(cache) and c.fresh(META, 12):
        return cache
    year = datetime.now(timezone.utc).year
    have = set(pd.to_datetime(cache["date"]).dt.year) if len(cache) else set()
    rows, redo = [], set()
    for y in range(FIRST_SEASON, year + 1):
        if y in have and y != year:
            continue
        try:
            rows += _schedule(f"{y}-02-15", f"{y}-11-30")
            redo.add(y)
        except Exception as e:
            print(f"mlb: {y} unavailable ({e})")
    if rows:
        new = pd.DataFrame(rows)
        new = new[new["status"].isin(["Final", "Game Over", "Completed Early"])]
        if len(cache):
            cache = cache[~pd.to_datetime(cache["date"]).dt.year.isin(redo)]
        cache = pd.concat([cache, new], ignore_index=True).drop_duplicates("pk", keep="last")
        os.makedirs(c.STATE_DIR, exist_ok=True)
        cache.to_csv(CACHE, index=False, compression="gzip")
        c.mark_fresh(META)
    print(f"MLB history: {len(cache)} games")
    return cache


def _ip(v):
    try:
        whole, _, part = str(v).partition(".")
        return int(whole) + int(part or 0) / 3
    except ValueError:
        return 0.0


def load_pitchers(first, last):
    """Season FIP for every pitcher: (13*HR + 3*(BB+HBP) - 2*K) / IP + 3.1"""
    data = c.load_json(PITCHERS, {})
    stale = not c.fresh(os.path.join(c.STATE_DIR, "mlb_pitchers_meta.json"), 20)
    for y in range(first, last + 1):
        if str(y) in data and not (y == last and stale):
            continue
        try:
            r = c.get(f"{API}/stats", params={"stats": "season", "group": "pitching", "season": y,
                                               "playerPool": "all", "limit": 3000, "sportId": 1}, timeout=120)
            season = {}
            for sp in r.json()["stats"][0]["splits"]:
                st, ip = sp["stat"], _ip(sp["stat"].get("inningsPitched"))
                if ip > 0:
                    fip = (13 * st.get("homeRuns", 0) + 3 * (st.get("baseOnBalls", 0) + st.get("hitByPitch", 0))
                           - 2 * st.get("strikeOuts", 0)) / ip + 3.1
                    season[str(sp["player"]["id"])] = [round(fip, 3), round(ip, 1)]
            data[str(y)] = season
        except Exception as e:
            print(f"mlb: pitcher stats {y} unavailable ({e})")
    c.save_json(PITCHERS, data)
    c.mark_fresh(os.path.join(c.STATE_DIR, "mlb_pitchers_meta.json"))
    return data


class MLBModel:
    def __init__(self):
        self.r = defaultdict(lambda: 1500.0)
        self.season = {}
        self.sp = defaultdict(lambda: deque(maxlen=10))
        self.rd = defaultdict(lambda: deque(maxlen=10))
        self.last = {}
        self.coef = np.array([0.14, 2.3, 0, 0, 0, 0])
        self.fip = {}
        self.run_scale = 1.0
        self.rs = defaultdict(lambda: deque(maxlen=15))
        self.ra = defaultdict(lambda: deque(maxlen=15))

    def rating(self, team, day):
        yr = day.year
        if self.season.get(team) not in (None, yr):
            self.r[team] = 1500 + (self.r[team] - 1500) * 2 / 3
        self.season[team] = yr
        return self.r[team]

    def sp_quality(self, pid):
        q = self.sp[pid] if pid else []
        return (sum(q) + LEAGUE_RA * 3) / (len(q) + 3)

    def features(self, g, day):
        rh, ra = self.rating(g["home_id"], day), self.rating(g["away_id"], day)
        sph, spa = self.sp_quality(g.get("hp")), self.sp_quality(g.get("ap"))

        def form(t):
            q = self.rd[t]
            return sum(q) / (len(q) + 3)

        def rest(t):
            if t not in self.last:
                return math.log1p(3)
            return math.log1p(min(max((day - self.last[t]).days, 0), 5))
        def fip(pid):
            prev = self.fip.get(str(day.year - 1), {}).get(str(pid)) if pid else None
            if not prev:
                return LEAGUE_FIP + 0.3  # unknown or new starters are usually below average
            f, ip = prev
            return (f * ip + LEAGUE_FIP * 40) / (ip + 40)
        fh, fa = fip(g.get("hp")), fip(g.get("ap"))
        x = np.array([1.0, (rh - ra) / 400, (spa - sph) / 2, (form(g["home_id"]) - form(g["away_id"])) / 2,
                      rest(g["home_id"]) - rest(g["away_id"]), (fa - fh)])
        return x, {"rh": rh, "ra": ra, "sph": sph, "spa": spa, "fiph": fh, "fipa": fa, "fh": list(self.rd[g["home_id"]]),
                   "fa": list(self.rd[g["away_id"]])}

    def update(self, g, day):
        h, a = g["home_id"], g["away_id"]
        rh, ra = self.rating(h, day), self.rating(a, day)
        hs, as_ = float(g["hs"]), float(g["as"])
        exp = 1 / (1 + 10 ** (-(rh + 24 - ra) / 400))
        res = 1.0 if hs > as_ else 0.0
        mult = math.log(abs(hs - as_) + 1) + 0.5
        self.r[h] = rh + 4 * mult * (res - exp)
        self.r[a] = ra - 4 * mult * (res - exp)
        if g.get("hp") and not pd.isna(g["hp"]):
            self.sp[int(g["hp"])].append(as_)
        if g.get("ap") and not pd.isna(g["ap"]):
            self.sp[int(g["ap"])].append(hs)
        self.rd[h].append(hs - as_)
        self.rd[a].append(as_ - hs)
        self.rs[h].append(hs)
        self.ra[h].append(as_)
        self.rs[a].append(as_)
        self.ra[a].append(hs)
        self.last[h] = self.last[a] = day


def _day(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).replace(tzinfo=None)


def build(s):
    hist = load_history()
    games = hist.sort_values("date").to_dict("records")
    model = MLBModel()
    model.fip = load_pitchers(FIRST_SEASON - 1, datetime.now(timezone.utc).year - 1)
    X, y = [], []
    runs = []  # (expected total, actual total) to correct any bias in the runs projection
    for g in games:
        for k in ("hp", "ap"):
            g[k] = None if pd.isna(g.get(k)) else int(g[k])
        day = _day(g["date"])
        if len(model.rd[g["home_id"]]) >= 5 and len(model.rd[g["away_id"]]) >= 5:
            x, fi = model.features(g, day)
            eh, ea = expected_runs(model, g, day, fi["fiph"], fi["fipa"])
            runs.append((eh + ea, float(g["hs"]) + float(g["as"])))
            X.append(x)
            y.append(1.0 if g["hs"] > g["as"] else 0.0)
        model.update(g, day)
    none = [None] * len(y)
    coef, info = c.evaluate_binary(X, y, none, none, FEATURES, base_cols=(0, 1), base_coef=[0.14, 2.3],
                                   min_edge=s["min_edge"], max_edge=s["max_edge"], min_weight=s["min_model_weight"],
                                   min_price=s["min_price"])
    model.coef = coef
    info["matches"] = len(games)
    recent = runs[-3000:]
    if recent:
        model.run_scale = sum(a for _, a in recent) / sum(e for e, _ in recent)
        info["runs_scale"] = round(model.run_scale, 3)
    try:
        today = datetime.now(timezone.utc).date()
        upcoming = _schedule(str(today - timedelta(days=1)), str(today + timedelta(days=2)))
        teams = c.get(f"{API}/teams", params={"sportId": 1}).json()["teams"]
    except Exception as e:
        print(f"mlb: upcoming schedule unavailable ({e})")
        upcoming, teams = [], []
    return Engine(model, upcoming, teams, info)


def expected_runs(model, g, day, fip_h, fip_a):
    """Runs each team should score: its offence, the other side's starter and bullpen, and the park."""
    def rate(q):
        return (sum(q) + LEAGUE_RA * 8) / (len(q) + 8)
    pitch_a = 0.6 * fip_a / LEAGUE_FIP + 0.4 * rate(model.ra[g["away_id"]]) / LEAGUE_RA
    pitch_h = 0.6 * fip_h / LEAGUE_FIP + 0.4 * rate(model.ra[g["home_id"]]) / LEAGUE_RA
    return rate(model.rs[g["home_id"]]) * pitch_a, rate(model.rs[g["away_id"]]) * pitch_h


class Engine:
    sport = "Baseball"
    series = SERIES
    research_hint = "starting pitcher changes or scratches, injuries, lineup news, bullpen fatigue and weather"

    def __init__(self, model, upcoming, teams, info):
        self.model, self.upcoming, self.info = model, upcoming, info
        self.abbr = {t["abbreviation"].upper(): t["id"] for t in teams}
        self.names = {t["id"]: [t.get("name", ""), t.get("locationName", ""), t.get("teamName", ""),
                                t.get("shortName", "")] for t in teams}

    def team_id(self, market):
        code = market["ticker"].split("-")[-1].upper()
        alias = {"AZ": "ARI", "ARI": "AZ", "CHW": "CWS", "CWS": "CHW", "WAS": "WSH", "WSH": "WAS",
                 "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "OAK": "ATH", "ATH": "OAK"}
        if code in self.abbr:
            return self.abbr[code]
        if alias.get(code) in self.abbr:
            return self.abbr[alias[code]]
        sub = market.get("yes_sub_title", "")
        for tid, names in self.names.items():
            if c.match_name(sub, [n for n in names if n], cutoff=0.85):
                return tid
        return None

    def evaluate(self, series, event, markets, day):
        if len(markets) != 2:
            return None
        ids = [self.team_id(m) for m in markets]
        if None in ids:
            return None
        end = datetime.fromisoformat(markets[0]["expected_expiration_time"].replace("Z", "+00:00")).replace(tzinfo=None)
        game = None
        for g in self.upcoming:
            if {g["home_id"], g["away_id"]} == set(ids) and g["status"] not in ("Final", "Game Over", "Postponed"):
                gd = _day(g["date"])
                if abs((gd - end).total_seconds()) < 12 * 3600 and (game is None or abs((gd - end).total_seconds()) <
                                                                    abs((_day(game["date"]) - end).total_seconds())):
                    game = g
        if not game:
            return None
        m = self.model
        if len(m.rd[game["home_id"]]) < 5 or len(m.rd[game["away_id"]]) < 5:
            return None
        x, info = m.features(game, day)
        ph = 1 / (1 + math.exp(-float(m.coef @ x)))
        outcomes = []
        for mk, tid in zip(markets, ids):
            home = tid == game["home_id"]
            p = ph if home else 1 - ph
            outcomes.append(c.Outcome(mk, p, f"{mk.get('yes_sub_title')} to win", self.why(game, info, home, p)))
        cand = c.Candidate(event, self.sport, "MLB", f"{game['away']} at {game['home']}", outcomes)
        cand.start_exact = _day(game["date"])  # first pitch, from the MLB schedule
        cand.sides = {mk["ticker"].split("-")[-1]: ("home" if tid == game["home_id"] else "away")
                      for mk, tid in zip(markets, ids)}
        lh, la = expected_runs(m, game, day, info["fiph"], info["fipa"])
        code = next((k for k, v in self.abbr.items() if v == game["home_id"]), "")
        park, note = PARK.get(code, 1.0), ""
        if code not in ROOF:
            loc = next((n for n in self.names.get(game["home_id"], [])[1:2] if n), None)
            spot = c.place(loc) if loc else None
            wx = c.weather(spot["lat"], spot["lon"], _day(game["date"])) if spot else None
            if wx:
                park *= 1 + 0.01 * (wx["temp"] - 70) / 10 * 3  # warm air carries the ball
                note = f"Forecast {wx['temp']:.0f}F, wind {wx['wind']:.0f} mph."
        cand.dist = c.PoissonDist(lh * park * m.run_scale, la * park * m.run_scale, (f"Park factor {PARK.get(code, 1.0):.2f}. " + note).strip())
        return cand

    def why(self, g, info, home, p):
        me, opp = (g["home"], g["away"]) if home else (g["away"], g["home"])
        r_me, r_opp = (info["rh"], info["ra"]) if home else (info["ra"], info["rh"])
        sp_me, sp_opp = (g.get("hpn"), g.get("apn")) if home else (g.get("apn"), g.get("hpn"))
        q_me, q_opp = (info["sph"], info["spa"]) if home else (info["spa"], info["sph"])
        fp_me, fp_opp = (info["fiph"], info["fipa"]) if home else (info["fipa"], info["fiph"])
        f_me, f_opp = (info["fh"], info["fa"]) if home else (info["fa"], info["fh"])
        parts = [f"Ratings {r_me:.0f} vs {r_opp:.0f}{' with home field' if home else ' on the road'}."]
        if sp_me or sp_opp:
            parts.append(f"Starters {sp_me or 'TBD'} (last season FIP {fp_me:.2f}, team allows {q_me:.1f} runs in his "
                         f"recent starts) vs {sp_opp or 'TBD'} ({fp_opp:.2f}, {q_opp:.1f}).")
        parts.append(f"Run differential last 10: {sum(f_me):+.0f} vs {sum(f_opp):+.0f}.")
        parts.append(f"Model gives {me} {p:.0%}.")
        return " ".join(parts)
