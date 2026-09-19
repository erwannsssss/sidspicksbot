"""
Picks agent (multi-sport v3)
-----------------------
Runs on GitHub Actions. Each run it:
  1. Rebuilds the tennis model from 10+ years of tour results plus every settled
     Kalshi match (see model.py for every factor it uses)
  2. Grades any open picks that have finished
  3. Scans today's Kalshi tennis markets for edges and makes new picks
  4. Sends new picks and results to Telegram
  5. Writes data.json for the dashboard
With "weekly" it also writes the weekly report, tunes itself and asks Gemini
for a written reflection.

Before picking, it searches the web (through Gemini with Google Search) for
injury news and what reputable previews and tipsters are saying.

Usage:  python agent.py run      (normal hourly run)
        python agent.py ask      (you asked in Telegram: always replies)
        python agent.py weekly   (force the weekly report + learning)
The weekly report also runs automatically on the first run after 9am
Eastern each Monday.
"""
import html
import json
import traceback
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

import common
import football
import mlb
import model as tm
import odds
import soccer

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
STATE_DIR = "state"
PICKS_FILE = os.path.join(STATE_DIR, "picks.json")
SETTINGS_FILE = os.path.join(STATE_DIR, "settings.json")
REPORTS_FILE = os.path.join(STATE_DIR, "reports.json")
CHANGELOG_FILE = os.path.join(STATE_DIR, "changelog.json")
SCANS_FILE = os.path.join(STATE_DIR, "scans.json")
RESEARCH_FILE = os.path.join(STATE_DIR, "research.json")
TAKEN_FILE = os.path.join(STATE_DIR, "taken.json")   # written by the Cloudflare worker
DASHBOARD_FILE = "data.json"

# Men's and women's match series on Kalshi (Challengers share the tour's rating pool)
SERIES = {
    "KXATPMATCH": ("M", "ATP"),
    "KXATPCHALLENGERMATCH": ("M", "ATP Challenger"),
    "KXWTAMATCH": ("W", "WTA"),
    "KXWTACHALLENGERMATCH": ("W", "WTA Challenger"),
    "KXATPGAME": ("M", "ATP"),
    "KXWTAGAME": ("W", "WTA"),
    "KXDAVISCUPMATCH": ("M", "Davis Cup"),
    "KXEXHIBITIONMEN": ("M", "Exhibition"),
    "KXEXHIBITIONWOMEN": ("W", "Exhibition"),
    "KXSIXKINGSSLAMMATCH": ("M", "Exhibition"),
}

SETTINGS_VERSION = 3
NEW_IN_V3 = {"min_edge": 0.05, "min_price": 0.20, "skip_expert_disagree": True}
DEFAULT_SETTINGS = {
    "version": SETTINGS_VERSION,
    "min_edge": 0.05,          # minimum edge after fees to make a pick
    "kelly_fraction": 0.25,    # quarter Kelly sizing
    "max_units": 3.0,          # biggest single pick
    "daily_unit_cap": 10.0,    # most units risked per day
    "min_matches": 12,         # each player needs this many rated matches
    "min_volume_24h": 500,     # skip thin markets
    "max_spread": 0.04,        # skip markets with a wide bid/ask spread
    "max_price": 0.90,         # skip heavy favourites
    "min_price": 0.20,         # skip long shots (they're usually overpriced)
    "fee_rate": 0.07,          # Kalshi taker fee multiplier
    "k_scale": 1.0,            # Elo speed, tuned weekly
    "model_weight": 0.5,       # how much to trust the model vs the market price, tuned weekly
    "min_model_weight": 0.25,  # floor, so it keeps making practice picks to learn from
    "weights": {},             # trust in the model vs market, per sport (tuned weekly)
    "horizon_hours": 24,       # fallback pick window
    "horizon": {"tennis": 24, "soccer": 72, "mlb": 30, "nfl": 96, "cfb": 96},  # pick early, when prices are softest
    "start_offset": {"tennis": 3, "soccer": 2.5, "mlb": 3, "nfl": 4, "cfb": 4},  # hours from kickoff to Kalshi's end time
    "sharp_weight": 0.7,       # how much to lean on sharp sportsbook prices when available
    "min_edge_no_sharp": 0.07, # stricter bar when no sharp price is available
    "odds_calls_per_day": 15,  # keeps The Odds API free tier (500 a month) from running out
    "disabled": [],            # leagues switched off because they keep getting worse prices than the close
    "max_edge": 0.15,          # bigger "edges" are almost always model errors
    "starting_bankroll": 100.0,
    "max_research": 6,         # web searches per run (free Gemini limits)
    "skip_expert_disagree": True,   # skip picks that reputable previews disagree with
}

CLAY = ["roland garros", "french open", "madrid", "rome", "italian", "monte carlo",
        "monte-carlo", "barcelona", "hamburg", "gstaad", "kitzbuhel", "kitzbühel",
        "umag", "bastad", "båstad", "buenos aires", "rio", "santiago", "estoril",
        "munich", "geneva", "lyon", "bucharest", "houston", "marrakech", "cordoba",
        "córdoba", "palermo", "prague", "bogota", "charleston", "rabat", "strasbourg",
        "iasi", "parma", "portoroz", "bogotá", "sao paulo", "são paulo", "tigre"]
GRASS = ["wimbledon", "queen", "halle", "hertogenbosch", "eastbourne", "mallorca",
         "berlin", "bad homburg", "nottingham", "newport", "birmingham", "ilkley",
         "surbiton", "stuttgart open"]


# ---------------------------------------------------------------- helpers
def now_utc():
    return datetime.now(timezone.utc)


def parse_time(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


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


def kalshi_get(path, params=None):
    for attempt in range(5):
        try:
            r = requests.get(KALSHI + path, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if 400 <= r.status_code < 500:
                r.raise_for_status()  # bad request: retrying won't help
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            code = getattr(getattr(e, "response", None), "status_code", 0) or 0
            if attempt == 4 or 400 <= code < 500:
                raise
            print(f"Kalshi retry after error: {e}")
            time.sleep(2 ** attempt)
    return {}


def kalshi_all(path, params):
    out, cursor = [], None
    for _ in range(200):
        p = dict(params, limit=1000)
        if cursor:
            p["cursor"] = cursor
        data = kalshi_get(path, p)
        items = data.get("markets") or data.get("events") or []
        out.extend(items)
        cursor = data.get("cursor")
        if not cursor or not items:
            break
    return out


def price(m, key):
    v = m.get(key)
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def tournament_of(m):
    hit = re.search(r"in the (\d{4} .+?) after a ball", m.get("rules_primary", ""))
    if not hit:
        return ""
    return re.sub(r" (Qualification )?(Round|Quarter|Semi|Final|R\d).*$", "", hit.group(1))


def fee(p, rate):
    return rate * p * (1 - p)


# ---------------------------------------------------------------- data
def fetch_results():
    """Every finished match on Kalshi (this is where Challenger results come from)."""
    markets = []
    for series in SERIES:
        markets += kalshi_all("/historical/markets", {"series_ticker": series})
        markets += kalshi_all("/markets", {"series_ticker": series, "status": "settled"})
    events = defaultdict(dict)
    for m in markets:
        events[m["event_ticker"]][m["ticker"]] = m
    matches = []
    for ev_ticker, ms in events.items():
        ms = list(ms.values())
        if len(ms) != 2 or {m.get("result") for m in ms} != {"yes", "no"}:
            continue  # walkovers, voids, odd events
        win = next(m for m in ms if m["result"] == "yes")
        lose = next(m for m in ms if m["result"] == "no")
        pool = SERIES.get(ev_ticker.split("-")[0], ("M", ""))[0]
        tourn = tournament_of(win)
        matches.append({
            "t": win.get("settlement_ts") or win.get("close_time"), "pool": pool,
            "wn": win.get("yes_sub_title"), "ln": lose.get("yes_sub_title"),
            "tourn": tourn, "best_of": best_of(pool, tourn), "ev": ev_ticker,
        })
    matches.sort(key=lambda x: x["t"])
    print(f"Kalshi: {len(matches)} finished matches")
    return matches


def best_of(pool, tourn):
    low = tourn.lower()
    slam = any(x in low for x in ("wimbledon", "us open", "australian open", "roland garros", "french open"))
    return 5 if pool == "M" and slam and "qualif" not in low else 3


# ---------------------------------------------------------------- picks
def open_markets(series_list):
    """Open markets grouped into events: {series: {event_ticker: [markets]}}."""
    out = {}
    for series in series_list:
        grouped = defaultdict(list)
        try:
            for m in kalshi_all("/markets", {"series_ticker": series, "status": "open"}):
                grouped[m["event_ticker"]].append(m)
        except requests.RequestException as e:
            print(f"Kalshi {series} unavailable ({e})")
        out[series] = grouped
    return out


def settled_markets(series):
    ms = []
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        try:
            ms += kalshi_all(path, {"series_ticker": series, **extra})
        except requests.RequestException as e:
            print(f"Kalshi {series} history unavailable ({e})")
    return ms


class TennisEngine:
    key, sport = "tennis", "Tennis"
    series = {k: v[1] for k, v in SERIES.items()}
    research_hint = "injuries, illness, retirements, withdrawals or heavy fatigue in the last two weeks"

    def __init__(self, ctx):
        self.ctx, self.info = ctx, ctx["info"]

    def evaluate(self, series, event, markets, day):
        if len(markets) != 2:
            return None
        ctx, s = self.ctx, self.ctx["s"]
        model, known = ctx["model"], ctx["known"]
        pool = SERIES[series][0]
        a, b = markets
        ida = tm.full_key(a.get("yes_sub_title"), known[pool])
        idb = tm.full_key(b.get("yes_sub_title"), known[pool])
        if min(model.played(pool, ida), model.played(pool, idb)) < s["min_matches"]:
            return None
        tourn = tournament_of(a)
        surf = tm.surface_for(tourn, ctx["surf_map"])
        venue = ctx["geo"].get(tm.place_of(tourn))
        x, finfo = model.features(pool, ida, idb, surf, day.date().isoformat(), venue, best_of(pool, tourn))
        qa = model.prob(x)
        outs = [common.Outcome(a, qa, f"{a.get('yes_sub_title')} to win", (finfo, False, surf)),
                common.Outcome(b, 1 - qa, f"{b.get('yes_sub_title')} to win", (finfo, True, surf))]
        cand = common.Candidate(event, "Tennis", SERIES[series][1], tourn, outs, {"surface": surf})
        cand.names = (a.get("yes_sub_title"), b.get("yes_sub_title"))
        return cand


def size_units(q, p_eff, s):
    f = (q - p_eff) / (1 - p_eff)
    units = s["kelly_fraction"] * f * 100
    units = min(s["max_units"], round(units * 2) / 2)
    return units if units >= 0.5 else 0.0


def tier_of(edge):
    return "A" if edge >= 0.08 else "B" if edge >= 0.05 else "C"


def explain(info, flip, surf, raw, q, ask):
    """Plain-English reasons for a pick, from the pick's point of view."""
    sw = (lambda t: (t[1], t[0])) if flip else (lambda t: t)
    ra, rb = sw((info["ra"], info["rb"]))
    rka, rkb = sw(info["rank"])
    (ha, hb), (fa, fb) = sw(info["h2h"]), sw(info["form"])
    da, db = sw(info["dom"])
    ga, gb = sw(info["games3"])
    ka, kb = sw(info["km"])
    parts = []
    if rka or rkb:
        parts.append(f"Ranked {int(rka) if rka else 'n/a'} vs {int(rkb) if rkb else 'n/a'}.")
    parts.append(f"Elo {ra:.0f} vs {rb:.0f} on {surf}.")
    parts.append(f"Won {da:.0%} vs {db:.0%} of games lately.")
    if fa or fb:
        parts.append(f"Form {sum(fa)}-{len(fa) - sum(fa)} vs {sum(fb)}-{len(fb) - sum(fb)}.")
    if ha or hb:
        parts.append(f"Head to head {ha}-{hb}.")
    if ga or gb:
        parts.append(f"Games played in last 3 days: {ga:.0f} vs {gb:.0f}.")
    if max(ka, kb) > 500:
        parts.append(f"Travelled {ka:,.0f} vs {kb:,.0f} km since last match.")
    if info["elev"] > 500:
        parts.append(f"Venue altitude {info['elev']:,.0f} m.")
    parts.append(f"Model {raw:.0%}, blended with the market {q:.0%}, price {ask:.0%}.")
    return " ".join(parts)


_GEMINI_MODEL = None


def gemini_model(key):
    """Find a current Gemini Flash model so this keeps working as Google renames models."""
    global _GEMINI_MODEL
    if _GEMINI_MODEL:
        return _GEMINI_MODEL
    base = "https://generativelanguage.googleapis.com/v1beta"
    models = requests.get(f"{base}/models", params={"key": key}, timeout=20).json().get("models", [])
    usable = [m["name"] for m in models
              if "generateContent" in m.get("supportedGenerationMethods", [])
              and "flash" in m["name"] and "lite" not in m["name"]
              and not any(x in m["name"] for x in ("image", "tts", "audio", "live", "exp"))]
    _GEMINI_MODEL = next((n for n in usable if "flash-latest" in n), None) or (usable[0] if usable else None)
    return _GEMINI_MODEL


def research(p, cache):
    """Search the web for news and expert opinion on a match. Returns a dict (never raises)."""
    empty = {"red_flag": False, "expert_lean": "none", "summary": "", "sources": []}
    hit = cache.get(p["event"])
    if hit and parse_time(hit["t"]) > now_utc() - timedelta(hours=6):
        return hit
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return empty
    bet = p["market"]
    prompt = (
        f"You are researching a {p['sport'].lower()} event for a prediction-market trader.\n"
        f"Event: {p['matchup']} ({p['tour']}), around {p['expected_end'][:10]}. "
        f"The trade under consideration: {bet}.\n"
        "Use Google Search. Only trust reputable sources: official league and tournament sites, major "
        "sports media (ESPN, BBC, The Athletic, Sky Sports, Marca, Gazzetta, Kicker, MLB.com, NFL.com), "
        "respected analytics sites, and well-known professional handicappers or sportsbook previews with a "
        "track record. Ignore anonymous forums, spam tip sites and social media rumours.\n"
        f"Find: (1) {p.get('_hint', 'injuries and team news')}; (2) recent form; "
        "(3) what reputable previews and tipsters predict.\n"
        "Reply with ONLY a JSON object, no other text:\n"
        '{"red_flag": true or false (true ONLY if credible news clearly makes this trade worse: '
        '"' + bet + '"), '
        '"expert_lean": "agrees" or "disagrees" or "mixed" or "none" (whether reputable previews '
        'support "' + bet + '"; "none" if you found no reputable previews), '
        '"summary": "at most two short sentences with the most useful facts"}'
    )
    try:
        model = gemini_model(key)
        if not model:
            return empty
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/{model}:generateContent",
                          params={"key": key}, timeout=90,
                          json={"contents": [{"parts": [{"text": prompt}]}],
                                "tools": [{"google_search": {}}]})
        data = r.json()
        cand = data["candidates"][0]
        text = "".join(part.get("text", "") for part in cand["content"]["parts"])
        found = re.search(r"\{.*\}", text, re.S)
        out = json.loads(found.group(0)) if found else {}
        chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
        sources = [{"title": c["web"].get("title", ""), "url": c["web"].get("uri", "")}
                   for c in chunks if c.get("web")][:4]
        result = {
            "red_flag": bool(out.get("red_flag", False)),
            "expert_lean": out.get("expert_lean") if out.get("expert_lean") in
            ("agrees", "disagrees", "mixed", "none") else "none",
            "summary": str(out.get("summary", ""))[:400],
            "sources": sources,
            "t": now_utc().isoformat(),
        }
        cache[p["event"]] = result
        time.sleep(5)  # stay inside the free rate limit
        return result
    except Exception as e:
        print(f"Research failed for {p['event']}: {e}")
        return empty


def make_picks(engines, picks, s, scans, research_cache, sharp):
    already = {p["event"] for p in picks}
    today = now_utc().date().isoformat()
    used_today = sum(p["units"] for p in picks if p["made"][:10] == today)
    now = now_utc()
    candidates = []
    for eng in engines:
        weight = s["weights"].get(eng.key, s["model_weight"])
        horizon = s["horizon"].get(eng.key, s["horizon_hours"])
        books = open_markets(list(eng.series))
        for series, events in books.items():
            for event, ms in events.items():
                ms = [m for m in ms if m.get("status") == "active"]
                if event in already or len(ms) < 2:
                    continue
                end = parse_time(ms[0].get("expected_expiration_time") or ms[0].get("occurrence_datetime"))
                start = end - timedelta(hours=s["start_offset"].get(eng.key, 3)) if end else None
                if not end or start < now or end > now + timedelta(hours=horizon + 4):
                    continue
                try:
                    cand = eng.evaluate(series, event, ms, now.replace(tzinfo=None))
                except Exception as e:
                    print(f"{eng.sport}: could not evaluate {event} ({e})")
                    continue
                if not cand or cand.league in s["disabled"]:
                    continue
                mids = []
                for o in cand.outcomes:
                    ask, bid = price(o.market, "yes_ask_dollars"), price(o.market, "yes_bid_dollars")
                    mids.append(None if ask is None or bid is None else (ask + bid) / 2)
                if None in mids or sum(mids) <= 0:
                    continue
                total = sum(mids)
                mids = [m / total for m in mids]
                o0 = cand.outcomes[0]
                if now < end - timedelta(hours=3):  # remember what the model and market said
                    scans[event] = {"key": eng.key, "ticker": o0.market["ticker"], "p": round(o0.prob, 4),
                                    "mid": round(mids[0], 3), "end": end.isoformat(), "t": now.isoformat()}
                best = None
                for o, mid in zip(cand.outcomes, mids):
                    m = o.market
                    q = weight * o.prob + (1 - weight) * mid
                    ask, bid = price(m, "yes_ask_dollars"), price(m, "yes_bid_dollars")
                    vol = price(m, "volume_24h_fp") or 0
                    if not (s["min_price"] <= ask <= s["max_price"]):
                        continue
                    if ask - bid > s["max_spread"] or vol < s["min_volume_24h"]:
                        continue
                    p_eff = ask + fee(ask, s["fee_rate"])
                    edge = q - p_eff
                    if s["min_edge"] <= edge <= s["max_edge"] and (best is None or edge > best["edge"]):
                        units = size_units(q, p_eff, s)
                        if units <= 0:
                            continue
                        if eng.key == "tennis":
                            finfo, flip, surf = o.why
                            why = explain(finfo, flip, surf, o.prob, q, ask)
                            me, opp = cand.names[1] if flip else cand.names[0], cand.names[0] if flip else cand.names[1]
                            teams, side = list(cand.names), me
                        else:
                            why = f"{o.why} Blended with the market that's {q:.0%} vs a {ask:.0%} price."
                            me, opp = o.label, ""
                            teams = [t.strip() for t in re.split(r" vs\.? | at ", cand.title)][:2]
                            side = "Draw" if o.label == "Draw" else o.label.replace(" to win", "")
                        best = {
                            "id": m["ticker"], "event": event, "sport": cand.sport, "model_key": eng.key,
                            "tour": cand.league, "tournament": cand.title, "matchup": cand.title,
                            "surface": cand.extra.get("surface", ""), "pick": me, "opponent": opp,
                            "market": o.label, "price": round(ask, 2), "model_prob": round(q, 3),
                            "raw_prob": round(o.prob, 3), "edge": round(edge, 3), "tier": tier_of(edge),
                            "units": units, "made": now.isoformat(), "expected_end": end.isoformat(),
                            "status": "pending", "latest_price": round(ask, 2), "pnl": None, "why": why,
                            "_hint": eng.research_hint, "_teams": teams, "_side": side,
                            "start_est": start.isoformat(), "close_price": None,
                        }
                if best:
                    candidates.append(best)
    candidates.sort(key=lambda p: p["edge"], reverse=True)
    new, searched = [], 0
    for p in candidates:
        if used_today + p["units"] > s["daily_unit_cap"]:
            continue
        # Compare with sharp sportsbook prices: the most reliable sign of a real edge
        found = sharp.prob(p) if sharp else None
        if found:
            sp, src = found
            q = s["sharp_weight"] * sp + (1 - s["sharp_weight"]) * p["model_prob"]
            p_eff = p["price"] + fee(p["price"], s["fee_rate"])
            p.update(sharp_prob=sp, sharp_source=src, model_prob=round(q, 3), edge=round(q - p_eff, 3))
            p["why"] += f" Sharp sportsbook price ({src}) says {sp:.0%}."
            if not (s["min_edge"] <= p["edge"] <= s["max_edge"]):
                print(f"Skipped {p['market']}: sharp price {sp:.0%} doesn't support it")
                continue
            p["tier"], p["units"] = tier_of(p["edge"]), size_units(q, p_eff, s)
            if p["units"] <= 0:
                continue
        else:
            p["sharp_prob"] = None
            if p["edge"] < s["min_edge_no_sharp"]:
                continue
        if searched < s["max_research"] or p["event"] in research_cache:
            searched += p["event"] not in research_cache
            info = research(p, research_cache)
        else:
            info = {"red_flag": False, "expert_lean": "none", "summary": "", "sources": []}
        for k in ("_hint", "_teams", "_side"):
            p.pop(k, None)
        p["research"] = info.get("summary", "")
        p["sources"] = info.get("sources", [])
        p["expert_lean"] = info.get("expert_lean", "none")
        if info.get("red_flag"):
            print(f"Skipped {p['market']}: news red flag ({p['research']})")
            continue
        if s["skip_expert_disagree"] and p["expert_lean"] == "disagrees":
            print(f"Skipped {p['market']}: experts disagree")
            continue
        used_today += p["units"]
        new.append(p)
    return new


def resolve_scans(scans, limit=150):
    """Look up how scanned games finished, so the agent can learn model-vs-market trust."""
    done = 0
    for ev, sc in sorted(scans.items(), key=lambda kv: kv[1].get("end", "")):
        if "ticker" not in sc or "won" in sc or done >= limit:
            continue
        end = parse_time(sc.get("end"))
        if not end or end > now_utc() - timedelta(hours=3):
            continue
        try:
            m = kalshi_get(f"/markets/{sc['ticker']}").get("market", {})
        except requests.RequestException:
            continue
        done += 1
        if m.get("result") in ("yes", "no"):
            sc["won"] = m["result"] == "yes"
        elif m.get("status") in ("finalized", "settled"):
            sc["won"] = None


def grade_picks(picks, s):
    graded = []
    for p in picks:
        if p["status"] != "pending":
            continue
        try:
            m = kalshi_get(f"/markets/{p['id']}").get("market", {})
        except requests.RequestException:
            try:
                m = kalshi_get(f"/historical/markets/{p['id']}").get("market", {})
            except requests.RequestException:
                continue
        status, result = m.get("status"), m.get("result")
        ask, bid = price(m, "yes_ask_dollars"), price(m, "yes_bid_dollars")
        if status == "active" and ask is not None and bid is not None and ask > 0:
            start = parse_time(p.get("start_est")) or (parse_time(p["expected_end"]) - timedelta(hours=3))
            if now_utc() < start:  # keep the last price seen before the game starts: the "closing" price
                p["close_price"] = round((ask + bid) / 2, 3)
                p["latest_price"] = p["close_price"]
        if status in ("finalized", "settled", "determined"):
            pr, u = p["price"], p["units"]
            fee_units = u * s["fee_rate"] * (1 - pr)
            if result == "yes":
                p["status"], p["pnl"] = "win", round(u * (1 - pr) / pr - fee_units, 2)
            elif result == "no":
                p["status"], p["pnl"] = "loss", round(-u - fee_units, 2)
            else:
                p["status"], p["pnl"] = "void", 0.0
            p["graded"] = now_utc().isoformat()
            graded.append(p)
    return graded


# ---------------------------------------------------------------- stats
def summarize(picks):
    done = [p for p in picks if p["status"] in ("win", "loss")]
    staked = sum(p["units"] for p in done)
    pnl = sum(p["pnl"] for p in done)
    wins = sum(p["status"] == "win" for p in done)
    beat = [p for p in done if p.get("latest_price") is not None]
    clv = [p["close_price"] - p["price"] for p in done if p.get("close_price") is not None]
    return {
        "picks": len(done), "wins": wins, "losses": len(done) - wins,
        "units": round(pnl, 2), "staked": round(staked, 2),
        "roi": round(pnl / staked, 3) if staked else 0.0,
        "beat_price": round(sum(p["latest_price"] > p["price"] for p in beat) / len(beat), 3) if beat else None,
        "clv": round(sum(clv) / len(clv), 4) if clv else None, "clv_picks": len(clv),
    }


def by_group(picks, key):
    groups = defaultdict(list)
    for p in picks:
        groups[p.get(key)].append(p)
    return {k: summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def calibration(picks):
    buckets = defaultdict(list)
    for p in picks:
        if p["status"] in ("win", "loss"):
            lo = min(int(p["model_prob"] * 10), 9) * 10
            buckets[f"{lo}-{lo + 10}%"].append(p["status"] == "win")
    return {k: {"n": len(v), "hit": round(sum(v) / len(v), 3)} for k, v in sorted(buckets.items())}


# ---------------------------------------------------------------- telegram / gemini
def telegram(text):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram skipped]\n" + text)
        return
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
        try:
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": chunk, "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        except requests.RequestException as e:
            print(f"Telegram failed: {e}")


def dashboard_url():
    repo = os.getenv("GITHUB_REPOSITORY", "")
    if "/" not in repo:
        return ""
    owner, name = repo.split("/", 1)
    return f"https://{owner.lower()}.github.io/{name}/"


def gemini(prompt):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    base = "https://generativelanguage.googleapis.com/v1beta"
    try:
        models = requests.get(f"{base}/models", params={"key": key}, timeout=20).json().get("models", [])
        usable = [m["name"] for m in models
                  if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]]
        pick = next((n for n in usable if "flash-latest" in n), None) or (usable[0] if usable else None)
        if not pick:
            return None
        r = requests.post(f"{base}/{pick}:generateContent", params={"key": key}, timeout=60,
                          json={"contents": [{"parts": [{"text": prompt}]}]}).json()
        return r["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:  # never let the reflection break the run
        print(f"Gemini failed: {e}")
        return None


def pick_line(p):
    lean = {"agrees": "experts agree", "disagrees": "experts disagree", "mixed": "experts split"}
    extra = ""
    if p.get("research"):
        extra = f"\n   📰 {html.escape(p['research'])}"
        if p.get("expert_lean") in lean:
            extra += f" ({lean[p['expert_lean']]})"
    return (f"<b>{p['tier']}</b>: {html.escape(p['market'])} at {p['price'] * 100:.0f}¢, "
            f"{p['units']:g}u (model {p['model_prob']:.0%}, edge +{p['edge']:.0%})\n"
            f"   <i>{html.escape(p.get('sport', 'Tennis'))}: "
            + (f"vs {html.escape(p['opponent'])}, " if p.get("opponent") else "")
            + f"{html.escape(p.get('matchup') or p['tournament'])}</i>{extra}")


# ---------------------------------------------------------------- weekly learning
def weekly(picks, ctx, s, scans, taken):
    matches = ctx.get("matches") or []
    changes = []
    # 1. Re-tune how fast Elo reacts, using every finished match (not just our picks)
    if matches:
        scores = {k: tm.elo_log_loss(matches, k) for k in (0.6, 0.8, 1.0, 1.2, 1.4)}
        best_k = min(scores, key=scores.get)
        if best_k != s["k_scale"] and scores[best_k] < scores.get(s["k_scale"], 9) - 0.002:
            changes.append(f"Elo speed changed from {s['k_scale']} to {best_k} "
                           f"(prediction error {scores.get(s['k_scale'], 0):.4f} to {scores[best_k]:.4f} on recent matches)")
            s["k_scale"] = best_k
    # 2. Re-tune how much to trust each sport's model vs the market. Starts from years of
    #    sportsbook closing odds, then Kalshi's own prices take over once 150 scanned games finish.
    for eng in ctx["engines"]:
        old = s["weights"].get(eng.key, s["model_weight"])
        new_w, why = old, ""
        mk = (eng.info or {}).get("market")
        if mk:
            new_w = max(mk["best_weight"], s["min_model_weight"])
            why = f"{mk['matches']:,} past games with closing odds"
        resolved = [(sc["p"], sc["mid"], sc["won"]) for sc in scans.values()
                    if sc.get("key") == eng.key and sc.get("won") is not None and "p" in sc]
        if len(resolved) >= 150:
            def loss(w):
                tot = 0.0
                for pm, mid, won in resolved:
                    q = min(max(w * pm + (1 - w) * mid, 1e-4), 1 - 1e-4)
                    tot -= math.log(q if won else 1 - q)
                return tot / len(resolved)
            grid = {k / 10: loss(k / 10) for k in range(11)}
            new_w = max(min(grid, key=grid.get), s["min_model_weight"])
            why = f"{len(resolved)} games it scanned on Kalshi"
        s["weights"][eng.key] = new_w
        if abs(new_w - old) >= 0.1:
            changes.append(f"{eng.sport} ({eng.key.upper()}): trust in the model vs the market changed from "
                           f"{old:.0%} to {new_w:.0%} based on {why}")
    # 3. Tighten or loosen the minimum edge, only with enough evidence
    done = [p for p in picks if p["status"] in ("win", "loss")]
    recent = done[-80:]
    low_edge = [p for p in recent if p["edge"] < 0.05]
    if len(low_edge) >= 30 and summarize(low_edge)["roi"] < -0.08 and s["min_edge"] < 0.08:
        old = s["min_edge"]
        s["min_edge"] = round(s["min_edge"] + 0.01, 3)
        changes.append(f"Minimum edge raised from {old:.0%} to {s['min_edge']:.0%}: "
                       f"low-edge picks returned {summarize(low_edge)['roi']:.0%} over {len(low_edge)} picks")
    elif len(recent) >= 60 and summarize(recent)["roi"] > 0.05 and s["min_edge"] > 0.03:
        old = s["min_edge"]
        s["min_edge"] = round(max(0.03, s["min_edge"] - 0.005), 3)
        changes.append(f"Minimum edge lowered from {old:.1%} to {s['min_edge']:.1%} after a profitable stretch")

    # 4. Switch off leagues that keep getting worse prices than the close (no real edge there)
    by_league = defaultdict(list)
    for p in done:
        if p.get("close_price") is not None:
            by_league[p.get("tour")].append(p["close_price"] - p["price"])
    for league, vals in by_league.items():
        if len(vals) >= 40 and sum(vals) / len(vals) < -0.01 and league not in s["disabled"]:
            s["disabled"].append(league)
            changes.append(f"Stopped picking {league}: over {len(vals)} picks the price moved against us by "
                           f"{-sum(vals) / len(vals) * 100:.1f}¢ on average before the start")
    # 5. Learn whether the web research is worth listening to
    disagree = [p for p in done if p.get("expert_lean") == "disagrees"]
    if not s["skip_expert_disagree"] and len(disagree) >= 25 and summarize(disagree)["roi"] < -0.10:
        s["skip_expert_disagree"] = True
        changes.append(f"Now skipping picks where reputable experts disagree: those returned "
                       f"{summarize(disagree)['roi']:.0%} over {len(disagree)} picks")

    week_ago = (now_utc() - timedelta(days=7)).isoformat()
    week = [p for p in done if p.get("graded", "") >= week_ago]
    report = {
        "week_ending": now_utc().date().isoformat(),
        "week": summarize(week), "all_time": summarize(done),
        "by_tier": by_group(week, "tier"), "by_tour": by_group(week, "tour"),
        "by_sport": by_group(week, "sport"),
        "by_expert_lean": by_group(done, "expert_lean"),
        "yours_week": summarize([p for p in week if taken.get(p["id"])]),
        "calibration": calibration(done), "changes": changes,
        "models": {e.key: public(e.info) for e in ctx["engines"]},
    }
    prompt = (
        "You are the analyst for a multi-sport prediction-market model (tennis, soccer, MLB, NFL, college "
        "football) that trades on Kalshi. Write a short weekly reflection (under 150 words, plain text, no markdown) for the "
        "owner: what went well, what went badly, likely reasons, and one idea worth testing next. "
        "Also comment on whether the web research (expert_lean) has been helping. "
        "Be honest about small sample sizes. Data:\n" + json.dumps(report)
        + "\nThis week's picks:\n"
        + json.dumps([{k: p.get(k) for k in ("sport", "market", "tour", "price", "model_prob", "edge",
                                             "tier", "status", "pnl", "expert_lean")} for p in week])
    )
    report["ai_note"] = gemini(prompt) or "AI reflection unavailable this week."
    return report, changes


# ---------------------------------------------------------------- dashboard
def public(info):
    """Model info without the raw test arrays (they're only used for tuning)."""
    return {k: v for k, v in (info or {}).items() if not k.startswith("_")}


LABELS = {"tennis": "Tennis", "soccer": "Soccer", "mlb": "MLB", "nfl": "NFL", "cfb": "College football"}


def write_dashboard(picks, s, reports, changelog, status_note, taken, engines):
    done = [p for p in picks if p["status"] in ("win", "loss", "void")]
    bankroll = s["starting_bankroll"] + sum(p["pnl"] or 0 for p in done)
    curve, run = [], s["starting_bankroll"]
    if done:
        first = min(p.get("graded", "") for p in done)[:10]
        curve.append({"t": first, "v": run})
    for p in sorted(done, key=lambda p: p.get("graded", "")):
        run += p["pnl"] or 0
        curve.append({"t": p.get("graded", "")[:10], "v": round(run, 2)})
    week_ago = (now_utc() - timedelta(days=7)).isoformat()
    save_json(DASHBOARD_FILE, {
        "updated": now_utc().isoformat(),
        "status_note": status_note,
        "bankroll": round(bankroll, 2),
        "settings": s,
        "open": [p for p in picks if p["status"] == "pending"],
        "history": sorted(done, key=lambda p: p.get("graded", ""), reverse=True),
        "worker_url": os.getenv("WORKER_URL", "").rstrip("/"),
        "models": {e.key: dict(public(e.info), sport=e.sport, label=LABELS.get(e.key, e.sport),
                               trust=s["weights"].get(e.key, s["model_weight"])) for e in engines},
        "summary": {"all": summarize(picks),
                    "week": summarize([p for p in done if p.get("graded", "") >= week_ago]),
                    "yours": summarize([p for p in done if taken.get(p["id"])])},
        "by_expert_lean": by_group(picks, "expert_lean"),
        "by_tier": by_group(picks, "tier"),
        "by_tour": by_group(picks, "tour"),
        "by_sport": by_group(picks, "sport"),
        "calibration": calibration(picks),
        "curve": curve,
        "reports": reports[-12:][::-1],
        "changelog": changelog[-30:][::-1],
    })


# ---------------------------------------------------------------- main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    stored = load_json(SETTINGS_FILE, {})
    s = {**DEFAULT_SETTINGS, **stored}
    if stored.get("version", 1) < SETTINGS_VERSION:  # apply the new, stricter defaults once
        s.update(NEW_IN_V3, version=SETTINGS_VERSION)
    picks = load_json(PICKS_FILE, [])
    for p in picks:  # picks made before multi-sport support
        p.setdefault("sport", "Tennis")
        p.setdefault("model_key", "tennis")
        p.setdefault("matchup", p.get("tournament", ""))
    reports = load_json(REPORTS_FILE, [])
    changelog = load_json(CHANGELOG_FILE, [])
    scans = load_json(SCANS_FILE, {})
    research_cache = load_json(RESEARCH_FILE, {})
    taken = load_json(TAKEN_FILE, {})
    link = dashboard_url()

    ctx = {"s": s}
    engines = []
    try:
        kalshi = fetch_results()
        td = tm.load_tennis_data()
        geo = tm.Geo()
        matches, surf_map, known = tm.build_matches(td, kalshi, geo)
        model, info = tm.train(matches, s["k_scale"], s["min_edge"], s["max_edge"], s["min_model_weight"], s["min_price"])
        geo.save()
        ctx.update(model=model, info=info, matches=matches, kalshi=kalshi, surf_map=surf_map,
                   known=known, geo=geo)
        engines.append(TennisEngine(ctx))
    except Exception:
        print("Tennis failed this run:\n" + traceback.format_exc())
    builders = [
        ("soccer", lambda: soccer.build(soccer.parse_kalshi_results(settled_markets("KXUCLGAME")), s)),
        ("mlb", lambda: mlb.build(s)),
        ("nfl", lambda: football.build_nfl(s)),
        ("cfb", lambda: football.build_cfb(s)),
    ]
    for key, build in builders:
        try:
            eng = build()
            if eng:
                eng.key = key
                engines.append(eng)
        except Exception:
            print(f"{key} failed this run:\n" + traceback.format_exc())
    ctx["engines"] = engines
    for e in engines:  # first run for a sport: start from what its history says
        mk = (e.info or {}).get("market")
        if e.key not in s["weights"]:
            s["weights"][e.key] = max(mk["best_weight"], s["min_model_weight"]) if mk else s["model_weight"]
    for e in engines:
        print(f"{e.sport} ({e.key}) model check:", json.dumps(public(e.info), default=float))

    graded = grade_picks(picks, s)
    if graded:
        lines = [f"{'✅' if p['status'] == 'win' else '❌' if p['status'] == 'loss' else '➖'} "
                 f"{html.escape(p['market'])} at {p['price'] * 100:.0f}¢: {p['pnl']:+g}u" for p in graded]
        telegram("<b>Results</b>\n" + "\n".join(lines))

    # Weekly report: first run after 9am Eastern (13:00 UTC) on Monday, once per week
    t = now_utc()
    due = t.weekday() == 0 and t.hour >= 13 and not any(r["week_ending"] == t.date().isoformat() for r in reports)
    if mode == "weekly" or due:
        report, changes = weekly(picks, ctx, s, scans, taken)
        reports.append(report)
        for c in changes:
            changelog.append({"date": now_utc().date().isoformat(), "change": c})
        w = report["week"]
        y = report["yours_week"]
        msg = (f"<b>Weekly report</b>\nModel: {w['wins']}-{w['losses']}, {w['units']:+g}u, "
               f"ROI {w['roi']:.1%}\n"
               + (f"Your trades: {y['wins']}-{y['losses']}, {y['units']:+g}u\n" if y["picks"] else "")
               + ("\n".join("• " + html.escape(c) for c in changes) + "\n" if changes else "No settings changed.\n")
               + "\n" + html.escape(report["ai_note"]))
        telegram(msg + (f"\n\n{link}" if link else ""))
        if any("Elo speed" in ch for ch in changes) and "matches" in ctx:
            model, info = tm.train(ctx["matches"], s["k_scale"], s["min_edge"], s["max_edge"], s["min_model_weight"],
                                   s["min_price"])
            ctx.update(model=model, info=info)
            for e in engines:
                if e.key == "tennis":
                    e.info = info

    resolve_scans(scans)
    sharp = odds.Sharp(s["odds_calls_per_day"])
    new = make_picks(engines, picks, s, scans, research_cache, sharp)
    sharp.save()
    picks.extend(new)
    if new:
        total = sum(p["units"] for p in new)
        telegram(f"<b>{len(new)} new pick{'s' if len(new) > 1 else ''}</b> ({total:g} units)\n\n"
                 + "\n\n".join(pick_line(p) for p in new) + (f"\n\n{link}" if link else ""))
    elif mode == "ask":
        still_open = [p for p in picks if p["status"] == "pending"]
        telegram(f"Fresh scan done: no new picks right now. {len(still_open)} pick"
                 f"{'' if len(still_open) == 1 else 's'} still open." + (f"\n{link}" if link else ""))
    print(f"Graded {len(graded)}, new picks {len(new)}")

    note = f"Last run found {len(new)} new pick(s) and graded {len(graded)}."
    save_json(PICKS_FILE, picks)
    save_json(SETTINGS_FILE, s)
    save_json(REPORTS_FILE, reports)
    save_json(CHANGELOG_FILE, changelog)
    save_json(SCANS_FILE, dict(sorted(scans.items(), key=lambda kv: kv[1]["t"])[-6000:]))
    save_json(RESEARCH_FILE, dict(sorted(research_cache.items(), key=lambda kv: kv[1].get("t", ""))[-1500:]))
    if "geo" in ctx:
        ctx["geo"].save()
    write_dashboard(picks, s, reports, changelog, note, taken, engines)


if __name__ == "__main__":
    main()
