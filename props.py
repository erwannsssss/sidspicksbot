"""
Player props
------------
NFL: nflverse weekly player stats (free) — passing, rushing and receiving yards, receptions and
touchdowns for every player. Each player's recent games (last season plus this one, recent games
weighted more) give an expected number; yards use a bell curve with that player's own spread,
counts (receptions, touchdowns) use a Poisson.

MLB: the free MLB Stats API season stats for every hitter and pitcher — hits, home runs, total
bases, hits + runs + RBIs per game, and strikeouts per start — shrunk toward league average for
players with few games.

Kalshi prop markets look like "Cooper Rush: 300+" with the line in floor_strike.
"""
import io
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

import common as c

NFL_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{}.csv"
NFL_CACHE = os.path.join(c.STATE_DIR, "nfl_players.csv.gz")
NFL_META = os.path.join(c.STATE_DIR, "nfl_players_meta.json")
MLB_CACHE = os.path.join(c.STATE_DIR, "mlb_players.json")
MLB_META = os.path.join(c.STATE_DIR, "mlb_players_meta.json")

# Kalshi series -> (stat, kind). kind: "yards" (bell curve) or "count" (Poisson)
NFL_PROPS = {
    "KXNFLPASSYDS": (("passing_yards",), "yards"), "KXNFLRECYDS": (("receiving_yards",), "yards"),
    "KXNFLRUSHYDS": (("rushing_yards",), "yards"), "KXNFLRSHYDS": (("rushing_yards",), "yards"),
    "KXNFLRRYDS": (("rushing_yards", "receiving_yards"), "yards"), "KXNFLREC": (("receptions",), "count"),
    "KXNFLPASSTDS": (("passing_tds",), "count"), "KXNFLGAMETD": (("rushing_tds", "receiving_tds"), "count"),
    "KXNFLANYTD": (("rushing_tds", "receiving_tds"), "count"), "KXNFL2TD": (("rushing_tds", "receiving_tds"), "count"),
}
MLB_PROPS = {"KXMLBKS": "strikeouts", "KXMLBHIT": "hits", "KXMLBHR": "homeRuns", "KXMLBTB": "totalBases",
             "KXMLBHRR": "hrr"}


def _poisson_at_least(lam, k):
    return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(int(k)))


# ---------------------------------------------------------------- NFL
def load_nfl_players():
    if os.path.exists(NFL_CACHE) and c.fresh(NFL_META, 12):
        return pd.read_csv(NFL_CACHE)
    now = datetime.now(timezone.utc)
    season = now.year if now.month >= 3 else now.year - 1
    frames = []
    for y in (season - 1, season):
        try:
            frames.append(pd.read_csv(io.StringIO(c.get(NFL_URL.format(y), timeout=180).text), low_memory=False))
        except Exception as e:
            print(f"props: NFL player stats {y} unavailable ({e})")
    if not frames:
        return pd.read_csv(NFL_CACHE) if os.path.exists(NFL_CACHE) else pd.DataFrame()
    cols = ["player_display_name", "team", "season", "week", "passing_yards", "passing_tds", "rushing_yards",
            "rushing_tds", "receptions", "receiving_yards", "receiving_tds"]
    df = pd.concat(frames)[cols].fillna(0)
    os.makedirs(c.STATE_DIR, exist_ok=True)
    df.to_csv(NFL_CACHE, index=False, compression="gzip")
    c.mark_fresh(NFL_META)
    return df


class NFLProps:
    def __init__(self):
        df = load_nfl_players()
        self.games = defaultdict(list)
        if len(df):
            df = df.sort_values(["season", "week"])
            for r in df.to_dict("records"):
                self.games[r["player_display_name"]].append(r)
        self.names = list(self.games)

    def prob(self, series, player, line):
        stat, kind = NFL_PROPS[series]
        name = c.match_name(player, self.names, cutoff=0.85)
        if not name:
            return None, None
        g = self.games[name][-12:]
        if len(g) < 3:
            return None, None
        vals = [sum(r[s] for s in stat) for r in g]
        w = [0.85 ** (len(vals) - 1 - i) for i in range(len(vals))]  # recent games count more
        mean = sum(v * wi for v, wi in zip(vals, w)) / sum(w)
        if kind == "yards":
            var = sum(wi * (v - mean) ** 2 for v, wi in zip(vals, w)) / sum(w)
            sd = max(math.sqrt(var), 0.45 * mean, 12)
            p = 1 - c.norm_cdf((line - mean) / sd)
            note = f"{name}: {mean:.0f} a game lately (last {len(vals)} games)."
        else:
            p = _poisson_at_least(max(mean, 0.02), math.floor(line) + 1)
            note = f"{name}: {mean:.2f} a game lately (last {len(vals)} games)."
        return p, note


# ---------------------------------------------------------------- MLB
def load_mlb_players():
    data = c.load_json(MLB_CACHE, {})
    if data and c.fresh(MLB_META, 12):
        return data
    year = datetime.now(timezone.utc).year
    out = {"hitters": {}, "pitchers": {}}
    try:
        for group in ("hitting", "pitching"):
            r = c.get("https://statsapi.mlb.com/api/v1/stats", params={
                "stats": "season", "group": group, "season": year, "playerPool": "all", "limit": 3000, "sportId": 1},
                timeout=120)
            for sp in r.json()["stats"][0]["splits"]:
                st, name = sp["stat"], sp["player"]["fullName"]
                if group == "hitting" and st.get("gamesPlayed"):
                    g = st["gamesPlayed"]
                    out["hitters"][name] = {"g": g, "hits": st.get("hits", 0) / g, "homeRuns": st.get("homeRuns", 0) / g,
                                            "totalBases": st.get("totalBases", 0) / g,
                                            "hrr": (st.get("hits", 0) + st.get("runs", 0) + st.get("rbi", 0)) / g}
                elif group == "pitching" and st.get("gamesStarted"):
                    gs = st["gamesStarted"]
                    out["pitchers"][name] = {"g": gs, "strikeouts": st.get("strikeOuts", 0) / gs}
        c.save_json(MLB_CACHE, out)
        c.mark_fresh(MLB_META)
        return out
    except Exception as e:
        print(f"props: MLB player stats unavailable ({e})")
        return data or out


LEAGUE_MLB = {"hits": 0.95, "homeRuns": 0.13, "totalBases": 1.55, "hrr": 2.1, "strikeouts": 5.2}


class MLBProps:
    def __init__(self):
        d = load_mlb_players()
        self.hitters, self.pitchers = d.get("hitters", {}), d.get("pitchers", {})

    def prob(self, series, player, line):
        stat = MLB_PROPS[series]
        table = self.pitchers if stat == "strikeouts" else self.hitters
        name = c.match_name(player, list(table), cutoff=0.85)
        if not name:
            return None, None
        row = table[name]
        k = 5 if stat == "strikeouts" else 25
        rate = (row[stat] * row["g"] + LEAGUE_MLB[stat] * k) / (row["g"] + k)
        need = math.floor(line) + 1
        if stat == "strikeouts":  # starters' strikeouts swing more than a Poisson allows
            p = 1 - sum(c.negbin_pmf(rate, 12, need))
        else:
            p = _poisson_at_least(rate, need)
        return p, f"{name}: {row[stat]:.2f} a game this season over {row['g']} games (adjusted {rate:.2f})."


# ---------------------------------------------------------------- shared
def player_of(market):
    sub = market.get("yes_sub_title") or ""
    return sub.split(":")[0].strip() if ":" in sub else None


def outcomes(props, series, markets):
    """(market, yes chance, label, why) for each prop market the model can price."""
    out = []
    for m in markets:
        player = player_of(m)
        try:
            line = float(m["floor_strike"])
        except (TypeError, ValueError, KeyError):
            continue
        if not player:
            continue
        p, note = props.prob(series, player, line)
        if p is None:
            continue
        out.append((m, min(max(p, 0.005), 0.995), m.get("yes_sub_title"), note))
    return out
