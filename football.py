"""
NFL and college football
------------------------
NFL data: nflverse (free) — every game since 2010 with scores, rest days, starting
quarterbacks and closing moneylines.
College data: CollegeFootballData.com (free API key) — every FBS and FCS game since 2014
with scores, neutral sites and betting lines.

Model: a team rating that moves after every game based on the margin (pulled back toward
average each offseason), home field, rest days, recent point differential, and for the NFL
expected points added per play (EPA, from nflverse play-by-play, the stat serious NFL models are
built on) and whether the usual starting quarterback is out. Tested against closing moneylines before use.
"""
import io
import math
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import common as c

NFL_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
NFL_CACHE = os.path.join(c.STATE_DIR, "nfl_games.csv.gz")
NFL_META = os.path.join(c.STATE_DIR, "nfl_meta.json")
CFB_CACHE = os.path.join(c.STATE_DIR, "cfb_games.csv.gz")
CFB_META = os.path.join(c.STATE_DIR, "cfb_meta.json")
CFB_API = "https://api.collegefootballdata.com"
FEATURES = ["home_field", "rating", "rest", "point_diff_form", "qb_change", "epa"]
EPA_CACHE = os.path.join(c.STATE_DIR, "nfl_epa.csv.gz")
EPA_META = os.path.join(c.STATE_DIR, "nfl_epa_meta.json")
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.csv.gz"
NFL_CODES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LA"}


def ml_to_prob(ml):
    try:
        ml = float(ml)
    except (TypeError, ValueError):
        return None
    if math.isnan(ml) or ml == 0:
        return None
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def _day(s):
    return pd.Timestamp(s).to_pydatetime().replace(tzinfo=None)


class GridModel:
    def __init__(self, k, hfa, regress, new_rating=1500):
        self.k, self.hfa, self.regress, self.new = k, hfa, regress, new_rating
        self.r, self.season = {}, {}
        self.form = defaultdict(lambda: deque(maxlen=5))
        self.last = {}
        self.qbs = defaultdict(lambda: deque(maxlen=4))
        self.epa = defaultdict(lambda: deque(maxlen=8))
        self.coef = np.array([0.35, 2.5, 0, 0, 0, 0])

    def rating(self, t, season, default=None):
        if t not in self.r:
            self.r[t] = default if default is not None else self.new
        if self.season.get(t) not in (None, season):
            self.r[t] = 1500 + (self.r[t] - 1500) * (1 - self.regress)
        self.season[t] = season
        return self.r[t]

    def qb_change(self, t, qb):
        seen = [q for q in self.qbs[t] if q]
        if not qb or not seen:
            return 0.0
        usual = max(set(seen), key=seen.count)
        return 0.0 if qb == usual else 1.0

    def features(self, g, day):
        h, a = g["home"], g["away"]
        rh, ra = self.rating(h, g["season"], g.get("hdef")), self.rating(a, g["season"], g.get("adef"))

        def rest(t):
            if t not in self.last:
                return math.log1p(7)
            return math.log1p(min(max((day - self.last[t]).days, 0), 14))

        def form(t):
            q = self.form[t]
            return sum(q) / (len(q) + 2)
        def epa(t):
            q = self.epa[t]
            return sum(q) / (len(q) + 2)
        x = np.array([0.0 if g.get("neutral") else 1.0, (rh - ra) / 400, rest(h) - rest(a),
                      (form(h) - form(a)) / 14, self.qb_change(a, g.get("aqb")) - self.qb_change(h, g.get("hqb")),
                      (epa(h) - epa(a)) * 5])
        return x, {"rh": rh, "ra": ra, "fh": list(self.form[h]), "fa": list(self.form[a]),
                   "eh": epa(h), "ea": epa(a),
                   "rest": (self.last.get(h), self.last.get(a))}

    def update(self, g, day):
        h, a = g["home"], g["away"]
        rh, ra = self.rating(h, g["season"], g.get("hdef")), self.rating(a, g["season"], g.get("adef"))
        hs, as_ = float(g["hs"]), float(g["as"])
        diff = rh + (0 if g.get("neutral") else self.hfa) - ra
        exp = 1 / (1 + 10 ** (-diff / 400))
        res = 1.0 if hs > as_ else 0.5 if hs == as_ else 0.0
        mov = math.log(abs(hs - as_) + 1) * 2.2 / (((diff if res == 1 else -diff) * 0.001) + 2.2)
        self.r[h] = rh + self.k * mov * (res - exp)
        self.r[a] = ra - self.k * mov * (res - exp)
        self.form[h].append(hs - as_)
        self.form[a].append(as_ - hs)
        self.last[h] = self.last[a] = day
        if g.get("hepa") is not None:
            self.epa[h].append(g["hepa"])
        if g.get("aepa") is not None:
            self.epa[a].append(g["aepa"])
        if g.get("hqb"):
            self.qbs[h].append(g["hqb"])
        if g.get("aqb"):
            self.qbs[a].append(g["aqb"])


def _train(model, games, s, name):
    X, y, mkt, price = [], [], [], []
    for g in games:
        day = g["day"]
        if g["done"]:
            if g["home"] in model.r and g["away"] in model.r and len(model.form[g["home"]]) >= 2 \
                    and len(model.form[g["away"]]) >= 2 and g["hs"] != g["as"]:
                x, _ = model.features(g, day)
                X.append(x)
                y.append(1.0 if g["hs"] > g["as"] else 0.0)
                ph, pa = g.get("hml"), g.get("aml")
                if ph and pa:
                    mkt.append(ph / (ph + pa))
                    price.append(ph)
                else:
                    mkt.append(None)
                    price.append(None)
            model.update(g, day)
    coef, info = c.evaluate_binary(X, y, mkt, price, FEATURES, base_cols=(0, 1), base_coef=[0.35, 2.5],
                                   min_edge=s["min_edge"], max_edge=s["max_edge"], min_weight=s["min_model_weight"],
                                   min_price=s["min_price"])
    model.coef = coef
    info["matches"] = sum(1 for g in games if g["done"])
    print(f"{name}: trained on {info['matches']} games, using {info.get('using')}")
    return info


# ---------------------------------------------------------------- NFL
def load_nfl():
    if os.path.exists(NFL_CACHE) and c.fresh(NFL_META, 12):
        return pd.read_csv(NFL_CACHE)
    try:
        df = pd.read_csv(io.StringIO(c.get(NFL_URL, timeout=120).text))
        df = df[df["season"] >= 2010]
        os.makedirs(c.STATE_DIR, exist_ok=True)
        df.to_csv(NFL_CACHE, index=False, compression="gzip")
        c.mark_fresh(NFL_META)
        return df
    except Exception as e:
        print(f"nfl: data unavailable ({e})")
        return pd.read_csv(NFL_CACHE) if os.path.exists(NFL_CACHE) else pd.DataFrame()


def load_epa(seasons):
    """Net EPA per play for every team in every game (offense EPA minus defense EPA allowed)."""
    cache = pd.read_csv(EPA_CACHE) if os.path.exists(EPA_CACHE) else pd.DataFrame()
    if len(cache) and c.fresh(EPA_META, 20):
        return cache
    have = set(cache["season"]) if len(cache) else set()
    current = max(seasons)
    frames = []
    for y in seasons:
        if y in have and y != current:
            continue
        try:
            raw = c.get(PBP_URL.format(y), timeout=300).content
            pbp = pd.read_csv(io.BytesIO(raw), compression="gzip", usecols=["game_id", "posteam", "defteam", "epa", "play_type"],
                              low_memory=False)
            pbp = pbp[pbp["play_type"].isin(["pass", "run"])].dropna(subset=["epa", "posteam"])
            off = pbp.groupby(["game_id", "posteam"])["epa"].mean().rename("off").reset_index().rename(columns={"posteam": "team"})
            de = pbp.groupby(["game_id", "defteam"])["epa"].mean().rename("def").reset_index().rename(columns={"defteam": "team"})
            g = off.merge(de, on=["game_id", "team"])
            g["net"] = g["off"] - g["def"]
            g["season"] = y
            frames.append(g[["game_id", "team", "net", "season"]])
            print(f"nfl: play-by-play {y} loaded")
        except Exception as e:
            print(f"nfl: play-by-play {y} unavailable ({e})")
    if frames:
        new = pd.concat(frames)
        if len(cache):
            cache = cache[~cache["season"].isin(set(new["season"]))]
        cache = pd.concat([cache, new], ignore_index=True)
        os.makedirs(c.STATE_DIR, exist_ok=True)
        cache.to_csv(EPA_CACHE, index=False, compression="gzip")
        c.mark_fresh(EPA_META)
    return cache


def build_nfl(s):
    df = load_nfl()
    if not len(df):
        return None
    epa = load_epa(sorted(set(int(x) for x in df["season"])))
    epa_map = {(r["game_id"], r["team"]): r["net"] for r in epa.to_dict("records")} if len(epa) else {}
    games = []
    for r in df.sort_values(["gameday", "gametime"]).to_dict("records"):
        done = not pd.isna(r.get("home_score"))
        games.append({"season": int(r["season"]), "day": _day(r["gameday"]), "home": r["home_team"],
                      "away": r["away_team"], "hs": r.get("home_score"), "as": r.get("away_score"), "done": done,
                      "neutral": r.get("location") == "Neutral", "hqb": r.get("home_qb_id") if isinstance(r.get("home_qb_id"), str) else None,
                      "aqb": r.get("away_qb_id") if isinstance(r.get("away_qb_id"), str) else None,
                      "hml": ml_to_prob(r.get("home_moneyline")), "aml": ml_to_prob(r.get("away_moneyline")),
                      "hepa": epa_map.get((r["game_id"], r["home_team"])), "aepa": epa_map.get((r["game_id"], r["away_team"]))})
    model = GridModel(k=20, hfa=48, regress=1 / 3)
    info = _train(model, games, s, "NFL")
    upcoming = [g for g in games if not g["done"]]
    return Engine("Football", {"KXNFLGAME": "NFL"}, model, upcoming, info, code_match=True)


# ---------------------------------------------------------------- college
def _cfb_get(path, params, key):
    return c.get(CFB_API + path, params=params, headers={"Authorization": f"Bearer {key}"}, timeout=120).json()


def load_cfb(key):
    cache = pd.read_csv(CFB_CACHE) if os.path.exists(CFB_CACHE) else pd.DataFrame()
    if len(cache) and c.fresh(CFB_META, 12):
        return cache
    now = datetime.now(timezone.utc)
    year = now.year if now.month >= 3 else now.year - 1
    have = set(cache["season"]) if len(cache) else set()
    rows, redo = [], set()
    for y in range(2014, year + 1):
        if y in have and y != year:
            continue
        try:
            lines = {}
            try:
                for g in _cfb_get("/lines", {"year": y, "seasonType": "both"}, key):
                    for ln in g.get("lines") or []:
                        hm, am = ln.get("homeMoneyline"), ln.get("awayMoneyline")
                        if hm and am:
                            lines[g["id"]] = (hm, am)
                            break
            except Exception as e:
                print(f"cfb: lines {y} unavailable ({e})")
            for g in _cfb_get("/games", {"year": y, "seasonType": "both"}, key):
                hm, am = lines.get(g.get("id"), (None, None))
                rows.append({"id": g.get("id"), "season": y, "start": g.get("startDate"),
                             "neutral": bool(g.get("neutralSite")), "home": g.get("homeTeam"),
                             "away": g.get("awayTeam"), "hs": g.get("homePoints"), "as": g.get("awayPoints"),
                             "hdiv": g.get("homeClassification"), "adiv": g.get("awayClassification"),
                             "hml": hm, "aml": am})
            redo.add(y)
        except Exception as e:
            print(f"cfb: {y} unavailable ({e})")
    if rows:
        new = pd.DataFrame(rows)
        if len(cache):
            cache = cache[~cache["season"].isin(redo)]
        cache = pd.concat([cache, new], ignore_index=True)
        os.makedirs(c.STATE_DIR, exist_ok=True)
        cache.to_csv(CFB_CACHE, index=False, compression="gzip")
        c.mark_fresh(CFB_META)
    return cache


def build_cfb(s):
    key = os.getenv("CFBD_API_KEY")
    if not key:
        print("College football skipped: add the CFBD_API_KEY secret to turn it on")
        return None
    df = load_cfb(key)
    if not len(df):
        return None
    games = []
    for r in df.sort_values("start").to_dict("records"):
        done = not pd.isna(r.get("hs")) and not pd.isna(r.get("as"))
        start = _day(r["start"])
        lower = lambda d: 1300 if str(d).lower() == "fcs" else 1100 if str(d).lower() not in ("fbs", "nan") else None
        games.append({"season": int(r["season"]), "day": start, "home": r["home"], "away": r["away"],
                      "hs": r.get("hs"), "as": r.get("as"), "done": done, "neutral": bool(r.get("neutral")),
                      "hdef": lower(r.get("hdiv")), "adef": lower(r.get("adiv")),
                      "hml": ml_to_prob(r.get("hml")), "aml": ml_to_prob(r.get("aml")),
                      "start": start})
    model = GridModel(k=25, hfa=55, regress=0.4)
    info = _train(model, games, s, "College football")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    upcoming = [g for g in games if not g["done"] and g["day"] > now - timedelta(days=1)]
    return Engine("Football", {"KXNCAAFGAME": "College Football"}, model, upcoming, info, code_match=False)


# ---------------------------------------------------------------- live
def cfb_name(n):
    n = re.sub(r"\bSt\.?$", "State", str(n).strip())
    return n.replace("St. ", "State ") if n.endswith("St.") else n


class Engine:
    research_hint = ("injuries (especially the starting quarterback), suspensions, weather, travel "
                     "and anything that changes the expected starters")

    def __init__(self, sport, series, model, upcoming, info, code_match):
        self.sport, self.series, self.model, self.upcoming, self.info = sport, series, model, upcoming, info
        self.code_match = code_match
        self.teams = sorted({g["home"] for g in upcoming} | {g["away"] for g in upcoming} | set(model.r))

    def team(self, market):
        if self.code_match:
            code = market["ticker"].split("-")[-1].upper()
            code = NFL_CODES.get(code, code)
            return code if code in self.model.r else None
        return c.match_name(cfb_name(market.get("yes_sub_title", "")), self.teams, cutoff=0.8)

    def evaluate(self, series, event, markets, day):
        if len(markets) != 2:
            return None
        ids = [self.team(m) for m in markets]
        if None in ids or ids[0] == ids[1]:
            return None
        end = datetime.fromisoformat(markets[0]["expected_expiration_time"].replace("Z", "+00:00")).replace(tzinfo=None)
        game = None
        for g in self.upcoming:
            if {g["home"], g["away"]} == set(ids) and abs((g["day"] - end).days) <= 3:
                game = g
        if not game:
            return None
        m = self.model
        if len(m.form[game["home"]]) < 2 or len(m.form[game["away"]]) < 2:
            return None
        x, info = m.features(game, day)
        ph = 1 / (1 + math.exp(-float(m.coef @ x)))
        outs = []
        for mk, t in zip(markets, ids):
            home = t == game["home"]
            p = ph if home else 1 - ph
            outs.append(c.Outcome(mk, p, f"{mk.get('yes_sub_title')} to win", self.why(game, info, home, p)))
        return c.Candidate(event, self.sport, self.series[series], f"{game['away']} at {game['home']}", outs)

    def why(self, g, info, home, p):
        me, opp = (g["home"], g["away"]) if home else (g["away"], g["home"])
        r_me, r_opp = (info["rh"], info["ra"]) if home else (info["ra"], info["rh"])
        f_me, f_opp = (info["fh"], info["fa"]) if home else (info["fa"], info["fh"])
        where = "neutral site" if g.get("neutral") else ("at home" if home else "on the road")
        parts = [f"Ratings {r_me:.0f} vs {r_opp:.0f}, {where}."]
        e_me, e_opp = (info["eh"], info["ea"]) if home else (info["ea"], info["eh"])
        if e_me or e_opp:
            parts.append(f"Net EPA per play lately {e_me:+.2f} vs {e_opp:+.2f}.")
        parts += [
                 f"Point differential last {len(f_me)}: {sum(f_me):+.0f} vs {sum(f_opp):+.0f}."]
        lh, la = info["rest"]
        if lh and la:
            rm, ro = ((g["day"] - lh).days, (g["day"] - la).days) if home else ((g["day"] - la).days, (g["day"] - lh).days)
            if rm != ro:
                parts.append(f"Days of rest {rm} vs {ro}.")
        parts.append(f"Model gives {me} {p:.0%}.")
        return " ".join(parts)
