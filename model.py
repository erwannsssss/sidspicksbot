"""
Tennis data and prediction model
--------------------------------
Data sources (all free):
  * tennis-data.co.uk: every ATP and WTA tour match since 2015 with the official
    surface, rankings, set scores, venue and closing sportsbook odds
  * Kalshi: every settled match on Kalshi, which adds the Challenger tours
  * Open-Meteo geocoding: map coordinates and altitude of each venue

For every match the model looks at, before the match was played:
  elo        surface-aware Elo rating gap
  dominance  share of games won in recent matches (the free stand-in for
             serve and return stats: it measures how easily players hold and break)
  rank       official ranking gap
  h2h        head-to-head record
  form       recent win rate
  fatigue    games played in the last 3 days and matches in the last 7
  rest       days since each player's last match
  travel     distance travelled since the last match
  altitude   venue altitude, which helps big servers
  best_of_5  favourites win more often over five sets
It learns how much each factor matters with logistic regression, and checks
itself on the most recent 15% of matches before it is allowed to replace plain Elo.
"""
import gzip
import io
import json
import math
import os
import re
import time
import unicodedata
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import common as c

STATE_DIR = "state"
TD_CACHE = os.path.join(STATE_DIR, "td_history.csv.gz")
TD_META = os.path.join(STATE_DIR, "td_meta.json")
GEO_FILE = os.path.join(STATE_DIR, "geo.json")
TD_PAGE = "https://www.tennis-data.co.uk/alldata.php"
TD_FIRST_YEAR = 2015
UA = {"User-Agent": "Mozilla/5.0 (picks-agent; personal research)"}

FEATURES = ["elo", "dominance", "rank", "h2h", "form", "fatigue_games", "fatigue_matches",
            "rest", "travel", "altitude", "best_of_5"]
LN10 = math.log(10)

SLAM_PLACES = {"us open": "New York", "australian open": "Melbourne", "roland garros": "Paris",
               "french open": "Paris", "wimbledon": "London"}
CLAY = ["roland garros", "french open", "madrid", "rome", "italian", "monte carlo", "barcelona",
        "hamburg", "gstaad", "kitzbuhel", "umag", "bastad", "buenos aires", "rio de janeiro", "rio",
        "santiago", "estoril", "munich", "geneva", "lyon", "bucharest", "houston", "marrakech",
        "cordoba", "palermo", "prague", "bogota", "charleston", "rabat", "strasbourg", "iasi",
        "parma", "portoroz", "sao paulo", "tigre", "braunschweig", "heilbronn", "sassuolo"]
GRASS = ["wimbledon", "queens", "halle", "hertogenbosch", "eastbourne", "mallorca", "berlin",
         "bad homburg", "nottingham", "newport", "birmingham", "ilkley", "surbiton"]


# ---------------------------------------------------------------- small helpers
def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    s = s.replace("-", "").replace("'", "").replace(".", " ")
    return re.sub(r"[^a-z ]", " ", s).split()


def td_key(name):
    """'Pacheco Mendez R.' -> 'pachecomendez_r'"""
    toks = norm(name)
    if len(toks) < 2:
        return "".join(toks)
    # trailing tokens of one or two letters are initials
    i = len(toks)
    while i > 1 and len(toks[i - 1]) <= 2:
        i -= 1
    return "".join(toks[:i]) + "_" + toks[i][0] if i < len(toks) else "".join(toks[:-1]) + "_" + toks[-1][0]


def full_key(name, known):
    """'Rodrigo Pacheco Mendez' -> 'pachecomendez_r' (matched against known tour keys)."""
    toks = norm(name)
    if len(toks) < 2:
        return "".join(toks)
    first = toks[0][0]
    for i in range(1, len(toks)):
        k = "".join(toks[i:]) + "_" + first
        if k in known:
            return k
    return toks[-1] + "_" + first


def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=str)


def place_of(tournament):
    """'2026 ATP Challenger Phan Thiet 4' -> 'Phan Thiet'"""
    t = re.sub(r"^\d{4}\s+", "", tournament or "")
    low = t.lower()
    for slam, city in SLAM_PLACES.items():
        if slam in low:
            return city
    t = re.sub(r"\b(ATP|WTA|Challenger|Men'?s|Women'?s|Singles|Qualification|Qualifying|Open|Tour)\b",
               " ", t, flags=re.I)
    t = re.sub(r"\(.*?\)|\b\d+\b", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def haversine(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


# ---------------------------------------------------------------- venues
class Geo:
    def __init__(self):
        self.cache = load_json(GEO_FILE, {})
        self.lookups = 0

    def get(self, place):
        key = " ".join(norm(place))
        if not key:
            return None
        if key in self.cache:
            return self.cache[key]
        if self.lookups >= 400:  # spread first-time lookups over a few runs
            return None
        self.lookups += 1
        try:
            r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": place, "count": 1}, timeout=15)
            res = (r.json().get("results") or [None])[0] if r.ok else "error"
        except requests.RequestException:
            res = "error"
        if res == "error":
            return None
        self.cache[key] = None if res is None else {
            "lat": res["latitude"], "lon": res["longitude"], "elev": res.get("elevation") or 0}
        time.sleep(0.1)
        return self.cache[key]

    def save(self):
        save_json(GEO_FILE, self.cache)


# ---------------------------------------------------------------- tennis-data.co.uk
def _td_links():
    page = requests.get(TD_PAGE, headers=UA, timeout=30).text
    links = {}
    for href in re.findall(r'href="([^"]+/(\d{4})(w?)/\d{4}\.xlsx?)"', page, flags=re.I):
        path, year, w = href
        links[(int(year), "W" if w else "M")] = "https://www.tennis-data.co.uk/" + path.lstrip("/")
    return links


def _read_td(url, pool, year):
    raw = requests.get(url, headers=UA, timeout=180)
    raw.raise_for_status()
    df = pd.read_excel(io.BytesIO(raw.content))
    for c in ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5", "AvgW", "AvgL",
              "WRank", "LRank", "Best of"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["wg"] = df[["W1", "W2", "W3", "W4", "W5"]].sum(axis=1, min_count=1)
    df["lg"] = df[["L1", "L2", "L3", "L4", "L5"]].sum(axis=1, min_count=1)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "pool": pool, "location": df.get("Location"), "tournament": df.get("Tournament"),
        "surface": df.get("Surface"), "best_of": df["Best of"], "winner": df["Winner"],
        "loser": df["Loser"], "wrank": df["WRank"], "lrank": df["LRank"], "wg": df["wg"],
        "lg": df["lg"], "comment": df.get("Comment"), "avgw": df["AvgW"], "avgl": df["AvgL"],
        "year": year})
    return out.dropna(subset=["date", "winner", "loser"])


def load_tennis_data():
    """Past seasons are cached in the repo; the current season is refreshed once a day."""
    meta = load_json(TD_META, {})
    cache = pd.read_csv(TD_CACHE) if os.path.exists(TD_CACHE) else pd.DataFrame()
    this_year = datetime.now(timezone.utc).year
    have = set(zip(cache["year"], cache["pool"])) if len(cache) else set()
    stale = time.time() - meta.get("refreshed", 0) > 20 * 3600
    want = [(y, p) for y in range(TD_FIRST_YEAR, this_year + 1) for p in ("M", "W")
            if (y, p) not in have or (stale and y >= this_year - (1 if datetime.now().month == 1 else 0))]
    if want:
        try:
            links = _td_links()
            frames = []
            for y, p in want:
                if (y, p) in links:
                    try:
                        frames.append(_read_td(links[(y, p)], p, y))
                        print(f"tennis-data: loaded {y} {'WTA' if p == 'W' else 'ATP'}")
                    except Exception as e:
                        print(f"tennis-data: {y} {p} failed ({e})")
            if frames:
                fresh = pd.concat(frames)
                keep = cache[~cache.set_index(["year", "pool"]).index.isin(list(zip(fresh.year, fresh.pool)))] \
                    if len(cache) else cache
                cache = pd.concat([keep, fresh], ignore_index=True)
                os.makedirs(STATE_DIR, exist_ok=True)
                cache.to_csv(TD_CACHE, index=False, compression="gzip")
                meta["refreshed"] = time.time()
                save_json(TD_META, meta)
        except Exception as e:
            print(f"tennis-data unavailable this run, using cache ({e})")
    return cache


# ---------------------------------------------------------------- combine sources
def build_matches(td, kalshi, geo):
    """One time-ordered list of matches from both sources, duplicates removed."""
    surf_map = {}
    matches = []
    known = {"M": set(), "W": set()}
    if len(td):
        td = td[~td["comment"].astype(str).str.contains("Walkover|w/o", case=False, na=False)]
        for r in td.itertuples(index=False):
            wk, lk = td_key(r.winner), td_key(r.loser)
            known[r.pool].update((wk, lk))
            surf = str(r.surface).lower() if isinstance(r.surface, str) else "hard"
            surf = "clay" if "clay" in surf else "grass" if "grass" in surf else "hard"
            for name in (r.location, r.tournament):
                if isinstance(name, str):
                    surf_map[" ".join(norm(name))] = surf
            mkt = None
            if r.avgw > 1 and r.avgl > 1:
                iw, il = 1 / r.avgw, 1 / r.avgl
                mkt = (iw / (iw + il), iw, il)  # fair chance, and the prices you'd actually pay
            matches.append({
                "t": r.date, "pool": r.pool, "w": wk, "l": lk, "wn": r.winner, "ln": r.loser,
                "surface": surf, "place": r.location if isinstance(r.location, str) else "",
                "best_of": 5 if r.best_of == 5 else 3,
                "games": (None if pd.isna(r.wg) or pd.isna(r.lg) else (float(r.wg), float(r.lg))),
                "wrank": None if pd.isna(r.wrank) else float(r.wrank),
                "lrank": None if pd.isna(r.lrank) else float(r.lrank),
                "mkt": mkt, "src": "td"})
    seen = defaultdict(list)
    for m in matches:
        seen[(m["pool"], frozenset((m["w"], m["l"])))].append(m["t"])
    added = 0
    for k in kalshi:
        pool = k["pool"]
        wk, lk = full_key(k["wn"], known[pool]), full_key(k["ln"], known[pool])
        day = k["t"][:10]
        dup = any(abs((datetime.fromisoformat(day) - datetime.fromisoformat(d)).days) <= 4
                  for d in seen.get((pool, frozenset((wk, lk))), []))
        k["wkey"], k["lkey"] = wk, lk
        if dup:
            continue
        place = place_of(k["tourn"])
        matches.append({
            "t": day, "pool": pool, "w": wk, "l": lk, "wn": k["wn"], "ln": k["ln"],
            "surface": surface_for(k["tourn"], surf_map), "place": place, "best_of": k["best_of"],
            "games": None, "wrank": None, "lrank": None, "mkt": None, "src": "kalshi"})
        added += 1
    matches.sort(key=lambda m: m["t"])
    for m in matches:
        v = geo.get(m["place"]) if m["place"] else None
        m["venue"] = v
    print(f"History: {len(matches)} matches ({len(matches) - added} tour, {added} extra from Kalshi)")
    return matches, surf_map, known


def surface_for(tournament, surf_map):
    place = " ".join(norm(place_of(tournament)))
    if place in surf_map:
        return surf_map[place]
    low = " ".join(norm(tournament))
    for k, v in surf_map.items():
        if len(k) > 4 and k in low:
            return v
    if any(re.search(r"\b" + w + r"\b", low) for w in CLAY):
        return "clay"
    if any(re.search(r"\b" + w + r"\b", low) for w in GRASS):
        return "grass"
    return "hard"


# ---------------------------------------------------------------- the model
def days_between(a, b):
    return (datetime.fromisoformat(b[:10]) - datetime.fromisoformat(a[:10])).days


class TennisModel:
    def __init__(self, k_scale=1.0):
        self.k_scale = k_scale
        self.r = defaultdict(lambda: 1500.0)
        self.n = defaultdict(int)
        self.games = defaultdict(lambda: deque(maxlen=30))
        self.results = defaultdict(lambda: deque(maxlen=10))
        self.hist = defaultdict(lambda: deque(maxlen=15))
        self.rank = {}
        self.h2h = defaultdict(int)
        self.coef = np.array([LN10] + [0.0] * (len(FEATURES) - 1))

    # --- ratings
    def k(self, n):
        return 250.0 / ((n + 5) ** 0.4) * self.k_scale

    def rating(self, pool, p, surface):
        return 0.5 * self.r[(pool, p, "all")] + 0.5 * self.r[(pool, p, surface)]

    def played(self, pool, p):
        return self.n[(pool, p, "all")]

    # --- feature pieces
    def dominance(self, pool, p):
        g = self.games[(pool, p)]
        won, tot = sum(x[0] for x in g), sum(x[0] + x[1] for x in g)
        return (won + 30) / (tot + 60)

    def form(self, pool, p):
        r = self.results[(pool, p)]
        return (sum(r) + 2.5) / (len(r) + 5)

    def log_rank(self, pool, p):
        return math.log(min(self.rank.get((pool, p), 600), 2000))

    def load(self, pool, p, day):
        g3 = m7 = 0
        last = None
        for (t, games, loc) in self.hist[(pool, p)]:
            d = days_between(t, day)
            if 0 <= d <= 3:
                g3 += games if games else 22
            if 0 <= d <= 7:
                m7 += 1
            last = (t, loc)
        return g3, m7, last

    def features(self, pool, a, b, surface, day, venue, best_of):
        ra, rb = self.rating(pool, a, surface), self.rating(pool, b, surface)
        elo = (ra - rb) / 400
        dom = (self.dominance(pool, a) - self.dominance(pool, b)) * 10
        rank = self.log_rank(pool, b) - self.log_rank(pool, a)
        ha, hb = self.h2h[(pool, a, b)], self.h2h[(pool, b, a)]
        h2h = (ha - hb) / (ha + hb + 2)
        form = self.form(pool, a) - self.form(pool, b)
        ga, ma, la = self.load(pool, a, day)
        gb, mb, lb = self.load(pool, b, day)

        def rest_travel(last):
            if not last:
                return math.log1p(30), 0.0
            d = max(days_between(last[0], day), 0)
            km = 0.0
            if last[1] and venue and d <= 14:
                km = haversine((last[1]["lat"], last[1]["lon"]), (venue["lat"], venue["lon"]))
            return math.log1p(min(d, 30)), math.log1p(km)
        rsa, tra = rest_travel(la)
        rsb, trb = rest_travel(lb)
        elev = (venue or {}).get("elev", 0) / 1000
        x = [elo, dom, rank, h2h, form, (ga - gb) / 30, (ma - mb) / 3, rsa - rsb, (tra - trb) / 5,
             elev * dom, (1.0 if best_of == 5 else 0.0) * elo]
        info = {"ra": ra, "rb": rb, "h2h": (ha, hb), "form": (list(self.results[(pool, a)]),
                list(self.results[(pool, b)])), "rank": (self.rank.get((pool, a)), self.rank.get((pool, b))),
                "dom": (self.dominance(pool, a), self.dominance(pool, b)), "games3": (ga, gb),
                "km": (math.expm1(tra), math.expm1(trb)), "elev": elev * 1000}
        return np.array(x), info

    def prob(self, x):
        return 1 / (1 + math.exp(-float(np.dot(self.coef, x))))

    # --- learning from one match
    def update(self, m):
        pool, w, l, s = m["pool"], m["w"], m["l"], m["surface"]
        for key in ("all", s):
            kw, kl = (pool, w, key), (pool, l, key)
            exp = 1.0 / (1.0 + 10 ** ((self.r[kl] - self.r[kw]) / 400.0))
            self.r[kw] += self.k(self.n[kw]) * (1 - exp)
            self.r[kl] -= self.k(self.n[kl]) * (1 - exp)
            self.n[kw] += 1
            self.n[kl] += 1
        if m["games"]:
            wg, lg = m["games"]
            self.games[(pool, w)].append((wg, lg))
            self.games[(pool, l)].append((lg, wg))
        total = sum(m["games"]) if m["games"] else None
        self.results[(pool, w)].append(1)
        self.results[(pool, l)].append(0)
        self.hist[(pool, w)].append((m["t"], total, m.get("venue")))
        self.hist[(pool, l)].append((m["t"], total, m.get("venue")))
        self.h2h[(pool, w, l)] += 1


def _fit(X, y, lam=2.0, iters=15):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def _ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train(matches, k_scale, min_edge=0.03, max_edge=0.15, min_weight=0.25, min_price=0.10):
    """Walk through history once: record features before each match, then learn from it."""
    model = TennisModel(k_scale)
    rows, labels, mkts, days = [], [], [], []
    for m in matches:
        pool = m["pool"]
        if m["wrank"]:
            model.rank[(pool, m["w"])] = m["wrank"]
        if m["lrank"]:
            model.rank[(pool, m["l"])] = m["lrank"]
        if model.played(pool, m["w"]) >= 5 and model.played(pool, m["l"]) >= 5:
            x, _ = model.features(pool, m["w"], m["l"], m["surface"], m["t"], m.get("venue"), m["best_of"])
            rows += [x, -x]
            labels += [1.0, 0.0]
            mk = m["mkt"]
            mkts += [None, None] if mk is None else [(mk[0], mk[1]), (1 - mk[0], mk[2])]
            days += [m["t"], m["t"]]
        model.update(m)
    X, y = np.array(rows), np.array(labels)
    info = {"matches": len(matches), "training_rows": len(rows) // 2}
    if len(rows) < 2000:
        info["using"] = "elo"
        return model, info
    cut = int(len(rows) * 0.85) // 2 * 2
    w_train = _fit(X[:cut], y[:cut])
    Xh, yh = X[cut:], y[cut:]
    p_model = 1 / (1 + np.exp(-Xh @ w_train))
    p_elo = 1 / (1 + np.exp(-LN10 * Xh[:, 0]))
    ll_model, ll_elo = _ll(p_model, yh), _ll(p_elo, yh)
    info["holdout"] = {"from": days[cut], "matches": len(yh) // 2,
                       "elo_log_loss": round(ll_elo, 4), "model_log_loss": round(ll_model, 4),
                       "elo_accuracy": round(float(np.mean((p_elo > 0.5) == (yh == 1))), 3),
                       "model_accuracy": round(float(np.mean((p_model > 0.5) == (yh == 1))), 3)}
    use_full = ll_model < ll_elo - 0.001
    info["using"] = "full" if use_full else "elo"
    # market comparison and best blend, on holdout matches that have sportsbook odds
    idx = [i for i in range(cut, len(rows)) if mkts[i] is not None]
    if len(idx) > 500:
        pm = np.array([mkts[i][0] for i in idx])
        price = np.array([mkts[i][1] for i in idx])
        pp = (p_model if use_full else p_elo)[[i - cut for i in idx]]
        yy = y[idx]
        grid = {w / 10: _ll(w / 10 * pp + (1 - w / 10) * pm, yy) for w in range(11)}
        best = min(grid, key=grid.get)
        info["market"] = {"market_log_loss": round(_ll(pm, yy), 4), "model_log_loss": round(_ll(pp, yy), 4),
                          "best_weight": best, "blend_log_loss": round(grid[best], 4), "matches": len(idx) // 2}
        bw = max(best, min_weight)
        info["market"]["backtest_weight"] = bw
        # backtest: the live rules, paying the average closing sportsbook price (a tough test)
        q = bw * pp + (1 - bw) * pm
        info["backtest"] = c.backtest(q, price, yy, min_edge, max_edge, min_price)
        info["_bt"] = {"model": pp, "market": pm, "price": price, "won": yy}
    model.coef = _fit(X, y) if use_full else np.array([LN10] + [0.0] * (len(FEATURES) - 1))
    info["weights"] = {f: round(float(v), 3) for f, v in zip(FEATURES, model.coef)}
    return model, info


def elo_log_loss(matches, k_scale, holdout=0.2):
    """Plain Elo prediction error on the newest matches, used to tune rating speed."""
    model = TennisModel(k_scale)
    cut = int(len(matches) * (1 - holdout))
    loss, n = 0.0, 0
    for i, m in enumerate(matches):
        if i >= cut and model.played(m["pool"], m["w"]) >= 5 and model.played(m["pool"], m["l"]) >= 5:
            d = model.rating(m["pool"], m["w"], m["surface"]) - model.rating(m["pool"], m["l"], m["surface"])
            p = min(max(1 / (1 + 10 ** (-d / 400)), 1e-4), 1 - 1e-4)
            loss -= math.log(p)
            n += 1
        model.update(m)
    return loss / max(n, 1)
