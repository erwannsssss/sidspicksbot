"""
Soccer
------
Data: football-data.co.uk (free) — results, shots on target and closing sportsbook odds for
Europe's top leagues since 2015, plus MLS, Liga MX, Brazil, Argentina, Japan and the Nordic
leagues. Champions League results come from Kalshi's own settled markets.

Model: one rating for every club on the same scale (so Champions League games between leagues
work), updated after each match using the goal margin. On top of the rating it looks at recent
chance quality (expected-goals share where football-data has xG, shots-on-target share where it
doesn't), recent goal difference and days of rest, and turns all of that into
home win / draw / away win chances with an ordered logistic model that is tested against
closing odds before it's used.
"""
import io
import math
import os
import re
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import common as c

CACHE = os.path.join(c.STATE_DIR, "soccer_history.csv.gz")
META = os.path.join(c.STATE_DIR, "soccer_meta.json")
FIRST_SEASON = 2015

# Kalshi series -> (football-data code, league name, starting rating for new clubs)
LEAGUES = {
    "KXEPLGAME": ("E0", "Premier League", 1650), "KXLALIGAGAME": ("SP1", "La Liga", 1620),
    "KXSERIEAGAME": ("I1", "Serie A", 1610), "KXBUNDESLIGAGAME": ("D1", "Bundesliga", 1600),
    "KXLIGUE1GAME": ("F1", "Ligue 1", 1580), "KXLIGAPORTUGALGAME": ("P1", "Liga Portugal", 1540),
    "KXEREDIVISIEGAME": ("N1", "Eredivisie", 1530), "KXBELGIANPLGAME": ("B1", "Belgian Pro League", 1500),
    "KXEFLCHAMPIONSHIPGAME": ("E1", "EFL Championship", 1480), "KXSUPERLIGGAME": ("T1", "Turkish Super Lig", 1480),
    "KXSLGREECEGAME": ("G1", "Greek Super League", 1450), "KXSCOTTISHPREMGAME": ("SC0", "Scottish Premiership", 1440),
    "KXMLSGAME": ("USA", "MLS", 1450), "KXLIGAMXGAME": ("MEX", "Liga MX", 1450),
    "KXBRASILEIROGAME": ("BRA", "Brasileirao", 1500), "KXARGPREMDIVGAME": ("ARG", "Argentina Primera", 1470),
    "KXJLEAGUEGAME": ("JPN", "J League", 1430), "KXDANISHSUPERLIGAGAME": ("DNK", "Danish Superliga", 1430),
    "KXELITESERIENGAME": ("NOR", "Eliteserien", 1400), "KXALLSVENSKANGAME": ("SWE", "Allsvenskan", 1400),
    "KXEKSTRAKLASAGAME": ("POL", "Ekstraklasa", 1400), "KXSWISSLEAGUEGAME": ("SWZ", "Swiss Super League", 1440),
}
EUROPE = {"KXUCLGAME": "Champions League"}
SERIES = {**{k: v[1] for k, v in LEAGUES.items()}, **EUROPE}
MAIN_CODES = {"E0", "E1", "SP1", "I1", "D1", "F1", "P1", "N1", "B1", "T1", "G1", "SC0"}
EURO_CODES = MAIN_CODES | {"DNK", "NOR", "SWE", "POL", "SWZ"}
FEATURES = ["rating", "chance_quality", "goal_diff_form", "rest"]


# ---------------------------------------------------------------- data
def _season_code(y):
    return f"{y % 100:02d}{(y + 1) % 100:02d}"


def _odds(row, names):
    for trio in names:
        vals = [row.get(k) for k in trio]
        try:
            vals = [float(v) for v in vals]
        except (TypeError, ValueError):
            continue
        if all(v > 1 for v in vals):
            return vals
    return None


def _first(row, keys):
    for k in keys:
        try:
            v = float(row.get(k))
            if v > 1:
                return v
        except (TypeError, ValueError):
            continue
    return None


def _main_season(code, y):
    url = f"https://www.football-data.co.uk/mmz4281/{_season_code(y)}/{code}.csv"
    df = pd.read_csv(io.StringIO(c.get(url).content.decode("utf-8-sig", "ignore")), on_bad_lines="skip")
    rows = []
    for r in df.to_dict("records"):
        if pd.isna(r.get("FTHG")) or pd.isna(r.get("HomeTeam")):
            continue
        o = _odds(r, [("AvgCH", "AvgCD", "AvgCA"), ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"),
                      ("B365H", "B365D", "B365A")])
        rows.append({"date": pd.to_datetime(r["Date"], dayfirst=True, errors="coerce"), "league": code,
                     "home": r["HomeTeam"], "away": r["AwayTeam"], "hg": r["FTHG"], "ag": r["FTAG"],
                     "hst": r.get("HST"), "ast": r.get("AST"), "hxg": r.get("HxG"), "axg": r.get("AxG"),
                     "o25": _first(r, ("AvgC>2.5", "Avg>2.5", "BbAv>2.5", "B365>2.5")),
                     "u25": _first(r, ("AvgC<2.5", "Avg<2.5", "BbAv<2.5", "B365<2.5")),
                     "oh": o[0] if o else None, "od": o[1] if o else None, "oa": o[2] if o else None,
                     "season": y})
    return rows


def _extra_league(code):
    url = f"https://www.football-data.co.uk/new/{code}.csv"
    df = pd.read_csv(io.StringIO(c.get(url).content.decode("utf-8-sig", "ignore")), on_bad_lines="skip")
    rows = []
    for r in df.to_dict("records"):
        if pd.isna(r.get("HG")) or pd.isna(r.get("Home")):
            continue
        d = pd.to_datetime(r["Date"], dayfirst=True, errors="coerce")
        if pd.isna(d) or d.year < FIRST_SEASON:
            continue
        o = _odds(r, [("AvgCH", "AvgCD", "AvgCA"), ("PSCH", "PSCD", "PSCA"), ("B365CH", "B365CD", "B365CA")])
        rows.append({"date": d, "league": code, "home": r["Home"], "away": r["Away"], "hg": r["HG"],
                     "ag": r["AG"], "hst": None, "ast": None, "o25": None, "u25": None, "oh": o[0] if o else None,
                     "od": o[1] if o else None, "oa": o[2] if o else None, "season": d.year})
    return rows


CACHE_VERSION = 2  # v2 adds over/under 2.5 goals odds


def load_history():
    cache = pd.read_csv(CACHE, parse_dates=["date"]) if os.path.exists(CACHE) else pd.DataFrame()
    if c.load_json(META, {}).get("version") != CACHE_VERSION:
        cache = pd.DataFrame()  # rebuild once with the new columns
    if len(cache) and c.fresh(META, 20):
        return cache
    now = datetime.now(timezone.utc)
    current = now.year if now.month >= 7 else now.year - 1
    have = set(zip(cache["league"], cache["season"])) if len(cache) else set()
    rows, replace = [], set()
    for code in {v[0] for v in LEAGUES.values()}:
        try:
            if code in MAIN_CODES:
                for y in range(FIRST_SEASON, current + 1):
                    if (code, y) not in have or y == current:
                        rows += _main_season(code, y)
                        replace.add((code, y))
            else:
                extra = _extra_league(code)
                rows += extra
                replace |= {(code, r["season"]) for r in extra}
        except Exception as e:
            print(f"soccer: {code} unavailable this run ({e})")
    if rows:
        new = pd.DataFrame(rows)
        if len(cache):
            cache = cache[~cache.set_index(["league", "season"]).index.isin(list(replace))]
        cache = pd.concat([cache, new], ignore_index=True).dropna(subset=["date"])
        os.makedirs(c.STATE_DIR, exist_ok=True)
        cache.to_csv(CACHE, index=False, compression="gzip")
        c.mark_fresh(META)
        c.save_json(META, {**c.load_json(META, {}), "version": CACHE_VERSION})
    print(f"Soccer history: {len(cache)} matches")
    return cache


# ---------------------------------------------------------------- model
def ordered_probs(s, c1, c2):
    """Probabilities of (home win, draw, away win) from a score s and two cut points."""
    pa = 1 / (1 + math.exp(-(c1 - s)))
    pad = 1 / (1 + math.exp(-(c2 - s)))
    return 1 - pad, pad - pa, pa


def fit_ordered(X, y, lam=1.0):
    """y: 0 = home win, 1 = draw, 2 = away win. Returns weights and cut points."""
    from scipy.optimize import minimize
    X, y = np.asarray(X, float), np.asarray(y)

    def nll(params):
        w, c1, d = params[:-2], params[-2], params[-1]
        c2 = c1 + math.exp(d)
        s = X @ w
        pa = 1 / (1 + np.exp(-(c1 - s)))
        pad = 1 / (1 + np.exp(-(c2 - s)))
        p = np.where(y == 2, pa, np.where(y == 1, pad - pa, 1 - pad))
        return -np.sum(np.log(np.clip(p, 1e-6, 1))) + lam * np.sum(w ** 2)
    x0 = np.zeros(X.shape[1] + 2)
    x0[-2], x0[-1] = -0.6, math.log(0.55)
    res = minimize(nll, x0, method="L-BFGS-B")
    w, c1, d = res.x[:-2], res.x[-2], res.x[-1]
    return w, c1, c1 + math.exp(d)


def multi_ll(P, y):
    P = np.clip(np.asarray(P, float), 1e-6, 1)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


class SoccerModel:
    def __init__(self):
        self.r = {}
        self.sot = defaultdict(lambda: deque(maxlen=8))
        self.gd = defaultdict(lambda: deque(maxlen=6))
        self.last = {}
        self.league_of = {}
        self.gf = defaultdict(lambda: deque(maxlen=10))
        self.ga = defaultdict(lambda: deque(maxlen=10))
        self.lg = defaultdict(lambda: [1.5, 1.2])   # league average home and away goals (slow moving)
        self.w, self.c1, self.c2 = np.array([1.0, 0, 0, 0]), -0.45, 0.25
        self.goal_scale = 1.0

    def goal_rates(self, h, a, lh, la):
        """Expected goals for the home and away team, from recent scoring and the league's average."""
        lg_h, lg_a = self.lg[lh]
        avg = (lg_h + lg_a) / 2

        def rate(q, k=6):
            return (sum(q) + avg * k) / (len(q) + k) / avg
        lam_h = lg_h * rate(self.gf[h]) * rate(self.ga[a]) * self.goal_scale
        lam_a = lg_a * rate(self.gf[a]) * rate(self.ga[h]) * self.goal_scale
        # nudge toward what the ratings say about who is stronger
        tilt = 10 ** ((self.rating(h, lh) - self.rating(a, la)) / 1600)
        return lam_h * tilt ** 0.5, lam_a / tilt ** 0.5

    def rating(self, team, league):
        if team not in self.r:
            prior = next((v[2] for v in LEAGUES.values() if v[0] == league), 1450)
            self.r[team] = prior
        return self.r[team]

    def features(self, home, away, lh, la, day):
        rh, ra = self.rating(home, lh), self.rating(away, la)

        def share(t):
            q = [x for x in self.sot[t] if x is not None]
            return (sum(q) + 2) / (len(q) + 4)

        def gdf(t):
            q = self.gd[t]
            return sum(q) / (len(q) + 2)

        def rest(t):
            if t not in self.last:
                return math.log1p(10)
            return math.log1p(min(max((day - self.last[t]).days, 0), 10))
        x = np.array([(rh - ra) / 400, (share(home) - share(away)) * 5, (gdf(home) - gdf(away)) / 2,
                      rest(home) - rest(away)])
        info = {"rh": rh, "ra": ra, "sot": (share(home), share(away)), "gd": (list(self.gd[home]), list(self.gd[away])),
                "rest": (self.last.get(home), self.last.get(away))}
        return x, info

    def probs(self, x):
        return ordered_probs(float(self.w @ x), self.c1, self.c2)

    def update(self, m):
        h, a, lh, la = m["home"], m["away"], m["lh"], m["la"]
        rh, ra = self.rating(h, lh), self.rating(a, la)
        hg, ag = int(m["hg"]), int(m["ag"])
        exp = 1 / (1 + 10 ** (-(rh + (0 if m.get("neutral") else 60) - ra) / 400))
        res = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        n = abs(hg - ag)
        mult = 1 if n <= 1 else 1.5 if n == 2 else (11 + n) / 8
        k = 24 * mult
        self.r[h] = rh + k * (res - exp)
        self.r[a] = ra - k * (res - exp)
        if m.get("hxg") is not None and m.get("axg") is not None and m["hxg"] + m["axg"] > 0:
            self.sot[h].append(m["hxg"] / (m["hxg"] + m["axg"]))
            self.sot[a].append(m["axg"] / (m["hxg"] + m["axg"]))
        elif m.get("hst") is not None and m.get("ast") is not None and m["hst"] + m["ast"] > 0:
            self.sot[h].append(m["hst"] / (m["hst"] + m["ast"]))
            self.sot[a].append(m["ast"] / (m["hst"] + m["ast"]))
        self.gd[h].append(hg - ag)
        self.gd[a].append(ag - hg)
        if not m.get("euro"):
            self.gf[h].append(hg)
            self.ga[h].append(ag)
            self.gf[a].append(ag)
            self.ga[a].append(hg)
            L = self.lg[lh]
            L[0] += 0.004 * (hg - L[0])
            L[1] += 0.004 * (ag - L[1])
        self.last[h] = self.last[a] = m["date"]
        self.league_of[h], self.league_of[a] = lh, la


def _num(v):
    try:
        v = float(v)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def build(kalshi_results, s):
    """kalshi_results: settled Kalshi Champions League games for cross-league links."""
    hist = load_history()
    matches = []
    teams_by_league = defaultdict(set)
    recent = defaultdict(set)   # clubs in each league over roughly the last season
    latest = hist.groupby("league")["date"].max().to_dict()
    for r in hist.to_dict("records"):
        teams_by_league[r["league"]].add(r["home"])
        teams_by_league[r["league"]].add(r["away"])
        if (pd.Timestamp(latest[r["league"]]) - pd.Timestamp(r["date"])).days <= 400:
            recent[r["league"]] |= {r["home"], r["away"]}
        matches.append({"date": pd.Timestamp(r["date"]).to_pydatetime(), "home": r["home"], "away": r["away"],
                        "lh": r["league"], "la": r["league"], "hg": r["hg"], "ag": r["ag"],
                        "hst": _num(r.get("hst")), "ast": _num(r.get("ast")),
                        "hxg": _num(r.get("hxg")), "axg": _num(r.get("axg")),
                        "odds": (r["oh"], r["od"], r["oa"]) if _num(r.get("oh")) and _num(r.get("od")) and _num(r.get("oa")) else None,
                        "ou": (_num(r.get("o25")), _num(r.get("u25"))) if _num(r.get("o25")) and _num(r.get("u25")) else None})
    euro_teams = sorted({t for code in EURO_CODES for t in teams_by_league.get(code, ())})
    league_of = {}
    for code, ts in teams_by_league.items():
        for t in ts:
            league_of[t] = code
    added = 0
    for k in kalshi_results:
        h, a = c.match_name(k["home"], euro_teams), c.match_name(k["away"], euro_teams)
        if h and a:
            matches.append({"date": k["date"], "home": h, "away": a, "lh": league_of[h], "la": league_of[a],
                            "hg": k["hg"], "ag": k["ag"], "hst": None, "ast": None, "odds": None, "euro": True})
            added += 1
    matches.sort(key=lambda m: m["date"])
    model = SoccerModel()
    X, Y, odds = [], [], []
    tot = []  # (model chance of over 2.5, market chance, went over)
    goals = []  # (projected total, actual total)
    for m in matches:
        if m.get("ou") and len(model.gf[m["home"]]) >= 5 and len(model.gf[m["away"]]) >= 5:
            lh_, la_ = model.goal_rates(m["home"], m["away"], m["lh"], m["la"])
            io, iu = 1 / m["ou"][0], 1 / m["ou"][1]
            tot.append((p_over(lh_ + la_, 2.5), io / (io + iu), 1.0 if m["hg"] + m["ag"] > 2.5 else 0.0, io, iu))
        if m["home"] in model.r and len(model.gf[m["home"]]) >= 5 and len(model.gf[m["away"]]) >= 5 and not m.get("euro"):
            lh2, la2 = model.goal_rates(m["home"], m["away"], m["lh"], m["la"])
            goals.append((lh2 + la2, m["hg"] + m["ag"]))
        if m["home"] in model.r and m["away"] in model.r and len(model.gd[m["home"]]) >= 4 and len(model.gd[m["away"]]) >= 4:
            x, _ = model.features(m["home"], m["away"], m["lh"], m["la"], m["date"])
            X.append(x)
            Y.append(0 if m["hg"] > m["ag"] else 1 if m["hg"] == m["ag"] else 2)
            odds.append(m["odds"])
        model.update(m)
    info = {"matches": len(matches), "from_kalshi": added, "training_rows": len(Y)}
    if len(tot) > 2000:
        t = np.array(tot[-int(len(tot) * 0.15):])
        info["_tot"] = t
        t = t[:, :3]
        info["totals_bias"] = round(float(np.mean(t[:, 0]) - np.mean(t[:, 2])), 3)
        grid = {g / 10: c.log_loss(g / 10 * t[:, 0] + (1 - g / 10) * t[:, 1], t[:, 2]) for g in range(11)}
        info["totals"] = {"matches": len(t), "model_log_loss": round(c.log_loss(t[:, 0], t[:, 2]), 4),
                          "market_log_loss": round(c.log_loss(t[:, 1], t[:, 2]), 4),
                          "best_weight": min(grid, key=grid.get)}
    X, Y = np.array(X), np.array(Y)
    if len(Y) > 3000:
        cut = int(len(Y) * 0.85)
        w, c1, c2 = fit_ordered(X[:cut], Y[:cut])
        wb, cb1, cb2 = fit_ordered(X[:cut, :1], Y[:cut])
        Ph = np.array([ordered_probs(float(w @ x), c1, c2) for x in X[cut:]])
        Pb = np.array([ordered_probs(float(wb @ x[:1]), cb1, cb2) for x in X[cut:]])
        yh = Y[cut:]
        llf, llb = multi_ll(Ph, yh), multi_ll(Pb, yh)
        use_full = llf < llb - 0.001
        P = Ph if use_full else Pb
        info["holdout"] = {"matches": int(len(yh)), "rating_log_loss": round(llb, 4), "model_log_loss": round(llf, 4),
                           "rating_accuracy": round(float(np.mean(Pb.argmax(1) == yh)), 3),
                           "model_accuracy": round(float(np.mean(Ph.argmax(1) == yh)), 3)}
        info["using"] = "full" if use_full else "rating"
        idx = [i for i in range(cut, len(Y)) if odds[i]]
        if len(idx) > 500:
            raw = np.array([[1 / o for o in odds[i]] for i in idx])
            fair = raw / raw.sum(1, keepdims=True)
            pm = P[[i - cut for i in idx]]
            yy = Y[idx]
            grid = {g / 10: multi_ll(g / 10 * pm + (1 - g / 10) * fair, yy) for g in range(11)}
            best = min(grid, key=grid.get)
            bw = max(best, s["min_model_weight"])
            info["market"] = {"market_log_loss": round(multi_ll(fair, yy), 4), "model_log_loss": round(multi_ll(pm, yy), 4),
                              "best_weight": best, "backtest_weight": bw, "matches": len(idx)}
            q = bw * pm + (1 - bw) * fair
            edges = q - raw
            pick = edges.argmax(1)
            info["backtest"] = c.backtest(q[np.arange(len(idx)), pick], raw[np.arange(len(idx)), pick],
                                          (pick == yy).astype(float), s["min_edge"], s["max_edge"], s["min_price"])
            info["_bt3"] = {"model": pm, "market": fair, "price": raw, "y": yy}
        if use_full:
            model.w, model.c1, model.c2 = fit_ordered(X, Y)
        else:
            wb, cb1, cb2 = fit_ordered(X[:, :1], Y)
            model.w, model.c1, model.c2 = np.array([wb[0], 0, 0, 0]), cb1, cb2
        info["weights"] = {f: round(float(v), 3) for f, v in zip(FEATURES, model.w)}
    last_goals = goals[-5000:]
    if last_goals:
        model.goal_scale = sum(a for _, a in last_goals) / sum(e for e, _ in last_goals)
        info["goals_scale"] = round(model.goal_scale, 3)
    live_euro = sorted({t for code in EURO_CODES for t in recent.get(code, ())})
    return Engine(model, recent, live_euro, info)


def _pois(lam, k):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def p_over(lam_total, line):
    return 1 - sum(_pois(lam_total, k) for k in range(int(math.floor(line)) + 1))


class GoalDist:
    """Independent Poisson goals for each team: totals, spreads and both-teams-to-score."""
    def __init__(self, lam_h, lam_a):
        self.lh, self.la = lam_h, lam_a
        self.grid = [[_pois(lam_h, i) * _pois(lam_a, j) for j in range(12)] for i in range(12)]

    def total_over(self, x):
        return sum(p for i, row in enumerate(self.grid) for j, p in enumerate(row) if i + j > x)

    def margin_over(self, side, x):
        sign = 1 if side == "home" else -1
        return sum(p for i, row in enumerate(self.grid) for j, p in enumerate(row) if sign * (i - j) > x)

    def btts(self):
        return (1 - math.exp(-self.lh)) * (1 - math.exp(-self.la))

    def team_over(self, side, x):
        lam = self.lh if side == "home" else self.la
        return 1 - sum(_pois(lam, k) for k in range(int(math.floor(x)) + 1))

    def part(self, which):
        """First half has about 45% of the goals, second half about 55%."""
        share = 0.45 if which == "1h" else 0.55
        return GoalDist(self.lh * share, self.la * share)

    def describe(self):
        return f"Expected goals {self.lh:.2f} vs {self.la:.2f} ({self.lh + self.la:.2f} total)."


FIXTURES = os.path.join(c.STATE_DIR, "soccer_fixtures.json")
COUNTRY = {"USA": "USA", "MEX": "Mexico", "BRA": "Brazil", "ARG": "Argentina", "JPN": "Japan", "DNK": "Denmark",
           "NOR": "Norway", "SWE": "Sweden", "POL": "Poland", "SWZ": "Switzerland"}


def load_fixtures():
    """Upcoming kickoff times (football-data lists them in UK time; converted to UTC)."""
    cached = c.load_json(FIXTURES, {})
    if cached.get("rows") and c.fresh(os.path.join(c.STATE_DIR, "soccer_fixtures_meta.json"), 6):
        return cached["rows"]
    rows = []
    for url, code_col, home, away in (("https://www.football-data.co.uk/fixtures.csv", "Div", "HomeTeam", "AwayTeam"),
                                      ("https://www.football-data.co.uk/new_league_fixtures.csv", "Country", "Home", "Away")):
        try:
            df = pd.read_csv(io.StringIO(c.get(url, timeout=60).content.decode("utf-8-sig", "ignore")))
            for r in df.to_dict("records"):
                try:
                    t = pd.to_datetime(f"{r['Date']} {r['Time']}", dayfirst=True).tz_localize("Europe/London").tz_convert("UTC")
                except Exception:
                    continue
                key = r[code_col] if code_col == "Div" else next((k for k, v in COUNTRY.items() if v == r[code_col]), None)
                if key:
                    rows.append({"lg": key, "home": r[home], "away": r[away], "t": t.isoformat()})
        except Exception as e:
            print(f"soccer: fixtures unavailable ({e})")
    if rows:
        c.save_json(FIXTURES, {"rows": rows})
        c.mark_fresh(os.path.join(c.STATE_DIR, "soccer_fixtures_meta.json"))
    return rows or cached.get("rows", [])


# ---------------------------------------------------------------- live
TEAMS_RE = re.compile(r"(?:wins|result of) the (.+?) vs\.? (.+?) (?:professional|soccer|UEFA|game|match)", re.I)


def teams_from(markets):
    for m in markets:
        hit = TEAMS_RE.search(m.get("rules_primary", ""))
        if hit:
            return hit.group(1).strip(), hit.group(2).strip()
    return None


class Engine:
    sport = "Soccer"
    series = SERIES
    research_hint = ("injuries, suspensions, expected lineups, squad rotation because of other competitions, "
                     "manager changes, motivation and weather")

    def __init__(self, model, teams_by_league, euro_teams, info):
        self.model, self.teams, self.euro, self.info = model, teams_by_league, euro_teams, info
        self.fixtures = load_fixtures()

    def evaluate(self, series, event, markets, day):
        if len(markets) != 3:
            return None
        names = teams_from(markets)
        if not names:
            return None
        if series in LEAGUES:
            code = LEAGUES[series][0]
            pool = sorted(self.teams.get(code, ()))
        else:
            code, pool = None, self.euro
        h, a = c.match_name(names[0], pool), c.match_name(names[1], pool)
        if not h or not a or len(self.model.gd[h]) < 4 or len(self.model.gd[a]) < 4:
            return None
        lh = code or self.model.league_of.get(h)
        la = code or self.model.league_of.get(a)
        x, info = self.model.features(h, a, lh, la, day)
        ph, pd_, pa = self.model.probs(x)
        outcomes = []
        for m in markets:
            sub = (m.get("yes_sub_title") or "").strip()
            if sub.lower() == "tie":
                outcomes.append(c.Outcome(m, pd_, "Draw", self.why(info, h, a, "draw", pd_)))
            elif c.match_name(sub, [names[0], names[1]]) == names[0]:
                outcomes.append(c.Outcome(m, ph, f"{sub} to win", self.why(info, h, a, "home", ph)))
            else:
                outcomes.append(c.Outcome(m, pa, f"{sub} to win", self.why(info, h, a, "away", pa)))
        if len(outcomes) != 3:
            return None
        cand = c.Candidate(event, self.sport, SERIES[series], f"{names[0]} vs {names[1]}", outcomes)
        cand.sides = {}
        for m in markets:
            code = m["ticker"].split("-")[-1]
            sub = (m.get("yes_sub_title") or "").strip()
            if sub.lower() != "tie":
                cand.sides[code] = "home" if c.match_name(sub, [names[0], names[1]]) == names[0] else "away"
        cand.dist = GoalDist(*self.model.goal_rates(h, a, lh, la))
        # exact kickoff from the fixture list, so it never picks a game that has already started
        for f in self.fixtures:
            if f["lg"] == lh and f["home"] == h and f["away"] == a:
                t = datetime.fromisoformat(f["t"]).replace(tzinfo=None)
                if abs((t - day).days) <= 4:
                    cand.start_exact = t
                    break
        return cand

    def why(self, info, h, a, side, p):
        gh, ga = info["gd"]
        parts = [f"Ratings {info['rh']:.0f} ({h}, home) vs {info['ra']:.0f} ({a}).",
                 f"Share of chances lately (xG or shots on target) {info['sot'][0]:.0%} vs {info['sot'][1]:.0%}.",
                 f"Goal difference last {len(gh)}: {sum(gh):+d} vs {sum(ga):+d}."]
        rh, ra = info["rest"]
        if rh and ra:
            parts.append(f"Last played {rh:%b %d} vs {ra:%b %d}.")
        parts.append(f"Model gives the {side if side != 'draw' else 'draw'} {p:.0%}.")
        return " ".join(parts)


def parse_kalshi_results(markets):
    """Settled Champions League markets -> list of results with a score proxy for ratings."""
    by_event = defaultdict(list)
    for m in markets:
        by_event[m["event_ticker"]].append(m)
    out = []
    for ev, ms in by_event.items():
        names = teams_from(ms)
        if not names or len(ms) != 3:
            continue
        won = [m for m in ms if m.get("result") == "yes"]
        if len(won) != 1:
            continue
        sub = (won[0].get("yes_sub_title") or "").lower()
        hg, ag = (1, 1) if sub == "tie" else ((1, 0) if c.match_name(sub, list(names)) == names[0] else (0, 1))
        t = won[0].get("settlement_ts") or won[0].get("close_time")
        out.append({"date": datetime.fromisoformat(t.replace("Z", "+00:00")).replace(tzinfo=None),
                    "home": names[0], "away": names[1], "hg": hg, "ag": ag})
    return out
