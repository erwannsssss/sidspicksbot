"""
Watch everything
----------------
Kalshi lists thousands of sports series (NBA, NHL, WNBA, college basketball, golf, F1, UFC,
esports, cricket, player props the agent has no model for yet, corners, goalscorers, futures...).
This tracker cycles through all of them, a batch per run, and records every market closing within
the next few days: its prices over time and how it settled.

Nothing here makes picks on its own. It builds the history needed to add a model for any of these
markets later, and it powers the "biggest movers" list on the dashboard.
"""
import gzip
import json
import os
from datetime import datetime, timedelta, timezone

import common as c

FILE = os.path.join(c.STATE_DIR, "tracker.json.gz")
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def _load():
    try:
        with gzip.open(FILE, "rt") as f:
            return json.load(f)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"series": [], "cursor": 0, "markets": {}, "refreshed": None}


def _save(d):
    os.makedirs(c.STATE_DIR, exist_ok=True)
    with gzip.open(FILE, "wt") as f:
        json.dump(d, f, separators=(",", ":"))


def _price(m, k):
    try:
        return float(m.get(k))
    except (TypeError, ValueError):
        return None


def run(kalshi_all, per_run):
    d = _load()
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%dT%H:%M")
    if not d["series"] or not d["refreshed"] or datetime.fromisoformat(d["refreshed"]) < now - timedelta(days=1):
        try:
            r = c.get(f"{KALSHI}/series", params={"category": "Sports"}, timeout=60)
            d["series"] = sorted(x["ticker"] for x in r.json().get("series", []))
            d["refreshed"] = now.isoformat()
        except Exception as e:
            print(f"tracker: series list unavailable ({e})")
    series = d["series"]
    if not series:
        return "no series list"
    batch = [series[(d["cursor"] + i) % len(series)] for i in range(min(per_run, len(series)))]
    d["cursor"] = (d["cursor"] + len(batch)) % len(series)
    horizon = now + timedelta(days=4)
    seen = 0
    for se in batch:
        try:
            ms = kalshi_all("/markets", {"series_ticker": se, "status": "open"})
        except Exception:
            continue
        for m in ms:
            close = m.get("close_time") or m.get("expected_expiration_time")
            try:
                ct = datetime.fromisoformat(close.replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                continue
            if ct > horizon:
                continue
            ask, bid = _price(m, "yes_ask_dollars"), _price(m, "yes_bid_dollars")
            if ask is None or bid is None:
                continue
            rec = d["markets"].setdefault(m["ticker"], {"se": se, "title": (m.get("yes_sub_title") or "")[:60],
                                                        "close": ct.isoformat(), "h": [], "result": None})
            rec["h"] = (rec["h"] + [[stamp, round((ask + bid) / 2, 3)]])[-24:]
            seen += 1
        # record results for this series' finished markets
        pending = [t for t, r in d["markets"].items() if r["se"] == se and r["result"] is None
                   and datetime.fromisoformat(r["close"]) < now]
        if pending:
            try:
                since = min(datetime.fromisoformat(d["markets"][t]["close"]) for t in pending) - timedelta(days=1)
                done = {m["ticker"]: m.get("result") for m in kalshi_all(
                    "/markets", {"series_ticker": se, "status": "settled", "min_close_ts": int(since.timestamp())})}
                for t in pending:
                    if done.get(t) in ("yes", "no"):
                        d["markets"][t]["result"] = done[t]
            except Exception:
                pass
    # keep the file small: drop markets settled more than 60 days ago
    cutoff = (now - timedelta(days=60)).isoformat()
    d["markets"] = {t: r for t, r in d["markets"].items() if r["close"] >= cutoff}
    _save(d)
    return f"checked {len(batch)} series, {seen} open markets updated, {len(d['markets'])} tracked"


def movers(limit=15):
    """Markets whose price moved the most over their recorded history (for the dashboard)."""
    d = _load()
    out = []
    now = datetime.now(timezone.utc).isoformat()
    for t, r in d["markets"].items():
        if r["result"] is None and r["close"] >= now and len(r["h"]) >= 2:
            move = r["h"][-1][1] - r["h"][0][1]
            out.append({"ticker": t, "series": r["se"], "title": r["title"], "move": round(move, 3),
                        "price": r["h"][-1][1], "close": r["close"]})
    out.sort(key=lambda x: abs(x["move"]), reverse=True)
    return {"tracked": len(d["markets"]), "series": len(d["series"]), "movers": out[:limit]}
