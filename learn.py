"""
Learning from everything
------------------------
Every market the agent evaluates (picked or not) is recorded with what the model said, what the
Kalshi price was, how that price moved, and later how it finished. From that it learns, every run:

  * trust:        how much to believe its own model vs the Kalshi price, per sport and market type
                  (winner, total, spread, both teams to score) and per league
  * confidence:   a correction to its percentages if, say, its "60%" calls only hit 55% of the time
  * price trend:  whether prices drifting for or against a pick predict how it finishes

Guardrails: nothing changes until there are enough finished games, league-level settings are
blended with the sport-level ones until each league has plenty of its own data, and every
change is checked on the most recent quarter of games before it's used.
"""
import gzip
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np

import common as c

OBS_FILE = os.path.join(c.STATE_DIR, "observations.json.gz")
LEARNED_FILE = os.path.join(c.STATE_DIR, "learned.json")
MIN_GROUP, MIN_LEAGUE, MIN_CALIB = 150, 60, 300
MIN_GAMES, MAX_TRUST = 40, 0.8


def _now():
    return datetime.now(timezone.utc)


def load_obs():
    try:
        with gzip.open(OBS_FILE, "rt") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}


def save_obs(obs):
    cutoff = (_now() - timedelta(days=120)).isoformat()
    keep = {k: v for k, v in obs.items() if v.get("start", "") >= cutoff or v.get("won") is None}
    if len(keep) > 40000:  # keep the newest
        keep = dict(sorted(keep.items(), key=lambda kv: kv[1].get("start", ""))[-40000:])
    os.makedirs(c.STATE_DIR, exist_ok=True)
    with gzip.open(OBS_FILE, "wt") as f:
        json.dump(keep, f, separators=(",", ":"))


def observe(obs, ticker, group, league, sport, series, game, p, mid, start, now):
    """Record the model's view and the Kalshi price for one market (called every run)."""
    o = obs.get(ticker)
    stamp = now.strftime("%Y-%m-%dT%H:%M")
    if o is None:
        o = obs[ticker] = {"g": group, "lg": league, "sp": sport, "se": series, "game": game,
                           "t0": stamp, "mid0": round(mid, 3), "h": [], "won": None, "new": True}
    else:
        o["new"] = False
    if now < start:  # only prices from before the start count
        o["p"], o["mid"], o["start"] = round(p, 4), round(mid, 3), start.isoformat()
        o["h"] = (o["h"] + [[stamp, round(mid, 3)]])[-30:]
    return o


def trend(o, hours=12):
    """How much the price has moved over the last `hours` (positive = rising)."""
    if not o or len(o["h"]) < 2:
        return 0.0
    cutoff = (_now() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M")
    past = next((m for t, m in o["h"] if t >= cutoff), o["h"][0][1])
    return round(o["h"][-1][1] - past, 3)


def resolve(obs, kalshi_all, limit_series=60):
    """Look up finished markets in bulk (one request per series) and record how they settled."""
    now = _now()
    pending = defaultdict(list)
    for t, o in obs.items():
        if o.get("won") is None and o.get("start") and datetime.fromisoformat(o["start"]) < now - timedelta(hours=2):
            pending[o["se"]].append(t)
    done = 0
    for series, tickers in list(pending.items())[:limit_series]:
        earliest = min(datetime.fromisoformat(obs[t]["start"]) for t in tickers) - timedelta(days=1)
        try:
            settled = kalshi_all("/markets", {"series_ticker": series, "status": "settled",
                                              "min_close_ts": int(earliest.timestamp())})
        except Exception as e:
            print(f"learn: could not check {series} ({e})")
            continue
        result = {m["ticker"]: m.get("result") for m in settled}
        for t in tickers:
            base, no_side = (t[:-3], True) if t.endswith(":no") else (t, False)
            r = result.get(base)
            if r in ("yes", "no"):
                obs[t]["won"] = (r == "no") if no_side else (r == "yes")
                done += 1
            elif base in result:
                obs[t]["won"] = "void"
    return done


def _fit_trust(rows):
    grid = {}
    for w in range(11):
        q = np.clip([w / 10 * p + (1 - w / 10) * m for p, m, _ in rows], 1e-4, 1 - 1e-4)
        y = np.array([r[2] for r in rows], float)
        grid[w / 10] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    return min(grid, key=grid.get)


def _logit(q):
    q = min(max(q, 1e-4), 1 - 1e-4)
    return math.log(q / (1 - q))


def learn(obs, defaults, floor):
    """Recompute trust and confidence corrections from every finished market."""
    rows = defaultdict(list)
    league_rows = defaultdict(list)
    games = defaultdict(set)       # distinct games per group: many markets on one game aren't independent evidence
    last = {}
    for o in obs.values():
        if o.get("won") in (True, False) and "p" in o and "mid" in o:
            r = (o["p"], o["mid"], 1.0 if o["won"] else 0.0)
            rows[o["g"]].append(r)
            league_rows[(o["g"], o["lg"])].append(r)
            games[o["g"]].add(o.get("game"))
            last[o["g"]] = max(last.get(o["g"], ""), o.get("start", ""))
    trust, trust_league, calib, counts = {}, {}, {}, {}
    for g, rs in rows.items():
        counts[g] = len(rs)
        base = defaults.get(g, floor)
        n_games = len(games[g])
        if len(rs) >= MIN_GROUP and n_games >= MIN_GAMES:
            # move away from the starting trust only as fast as the number of separate games justifies,
            # and never trust the model more than MAX_TRUST however good it has looked
            fitted = max(_fit_trust(rs), floor)
            trust[g] = round(min(base + (fitted - base) * n_games / (n_games + 100), MAX_TRUST), 2)
        else:
            trust[g] = base
        # confidence correction: q' = sigmoid(a + b * logit(q)), fitted on older games, checked on newer
        if len(rs) >= MIN_CALIB and len(games[g]) >= 2 * MIN_GAMES:
            w = trust[g]
            z = np.array([_logit(w * p + (1 - w) * m) for p, m, _ in rs])
            y = np.array([r[2] for r in rs])
            cut = int(len(rs) * 0.75)
            X = np.column_stack([np.ones(len(z)), z])
            ab = c.fit_logistic(X[:cut], y[:cut], lam=5.0)
            before = c.log_loss(1 / (1 + np.exp(-z[cut:])), y[cut:])
            after = c.log_loss(1 / (1 + np.exp(-(X[cut:] @ ab))), y[cut:])
            if after < before - 0.002:
                a, b = float(np.clip(ab[0], -0.5, 0.5)), float(np.clip(ab[1], 0.6, 1.4))
                calib[g] = [round(a, 3), round(b, 3)]
    for (g, lg), rs in league_rows.items():
        if len(rs) >= MIN_LEAGUE:
            w_l = max(_fit_trust(rs), floor)
            n = len(rs)
            trust_league[f"{g}|{lg}"] = round((n * w_l + 300 * trust[g]) / (n + 300), 2)
    return {"trust": trust, "trust_league": trust_league, "calib": calib, "counts": counts,
            "games": {g: len(v) for g, v in games.items()}, "last": last, "updated": _now().isoformat()}


def get_trust(learned, group, league, fallback):
    return learned.get("trust_league", {}).get(f"{group}|{league}", learned.get("trust", {}).get(group, fallback))


def calibrate(learned, group, q):
    ab = learned.get("calib", {}).get(group)
    if not ab:
        return q
    return 1 / (1 + math.exp(-(ab[0] + ab[1] * _logit(q))))


def trend_report(obs):
    """Did prices drifting toward or away from a side predict how it finished? (for the dashboard)"""
    buckets = defaultdict(lambda: [0, 0])
    for o in obs.values():
        if o.get("won") in (True, False) and len(o.get("h", [])) >= 3:
            move = o["h"][-1][1] - o["h"][0][1]
            b = "rising" if move >= 0.03 else "falling" if move <= -0.03 else "flat"
            exp_ = o["h"][-1][1]
            buckets[b][0] += 1
            buckets[b][1] += (1 if o["won"] else 0) - exp_
    return {k: {"markets": n, "hit_vs_price": round(d / n, 3) if n else 0} for k, (n, d) in buckets.items()}
