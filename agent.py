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
import learn
import model as tm
import odds
import props
import tracker
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
CONTROL_FILE = os.path.join(STATE_DIR, "control.json")  # written by the Cloudflare worker
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

SETTINGS_VERSION = 7
NEW_IN_V3 = {"min_edge": 0.03, "min_price": 0.20, "skip_expert_disagree": True, "min_edge_no_sharp": 0.05,
             "kelly_fraction": 0.125, "max_units": 8.0, "daily_unit_cap": 8.0, "brake_stop": None,
             "paused": False, "min_main_prob": 0.0}
DEFAULT_SETTINGS = {
    "version": SETTINGS_VERSION,
    "min_edge": 0.03,          # minimum edge after fees when a sharp sportsbook price backs the pick
    # Bankroll protection: the 100 units are treated as the owner's entire, irreplaceable bankroll.
    # Survival comes first: small bets, a daily limit, and automatic brakes after losses.
    "kelly_fraction": 0.125,   # eighth-Kelly sizing (half of the usual "cautious" quarter Kelly)
    "max_units": 8.0,          # biggest single pick: no separate limit beyond the daily cap
    "daily_unit_cap": 8.0,     # most units risked per day
    "brake_half": 0.10,        # down 10% from the peak: bet sizes are halved
    "brake_stop": None,        # no automatic pause (set a number like 0.20 to pause picks after a 20% drop)
    "min_main_prob": 0.0,     # no minimum chance: any pick the model rates as good (the 20¢ min price still applies)
    "paused": False,           # set by /pause in Telegram (or the optional brake)
    "line_weights": {},        # starting trust for totals, spreads and both-teams-to-score, per sport
    "league_daily_cap": 4.0,   # most units a day in any one league
    "trend_block": 0.04,       # skip a recommended pick if its price fell this much in the last 12 hours
    "move_alert": 0.05,        # alert when an open pick's price moves 5¢ either way
    "gap_alert": 0.04,         # flag Kalshi prices at least 4¢ below the sharp sportsbook price
    "odds_markets": "h2h,totals,spreads",
    "track_series_per_run": 150,  # other Kalshi sports series watched per run (all of them over about a day)
    "parlays_per_day": 2,      # parlay suggestions a day
    "parlay_units": 1.0,       # stake per parlay
    "parlay_max_legs": 4,
    "parlay_leg_edge": 0.03,   # every leg needs at least this edge
    "parlay_min_prob": 0.05,   # at least a 1-in-20 chance to hit
    "parlay_min_ev": 0.10,     # model says it's worth at least 10% more than it costs
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
    "min_edge_no_sharp": 0.05, # stricter bar when no sharp price is available
    "watch_edge": 0.015,       # smaller edges go on the practice-only watchlist
    "watch_per_day": 20,       # most watchlist picks per day
    "watch_weight": 0.5,       # the watchlist trusts the model this much
    "watch_per_type": 4,       # most watchlist picks a day of any one kind (keeps it varied)
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
    units = s["kelly_fraction"] * f * 100 * s.get("_size_factor", 1.0)
    units = min(s["max_units"], round(units * 2) / 2)
    return units if units >= 0.5 else 0.0


def bankroll_state(picks, s):
    """Current bankroll, its peak, and how far below the peak it is (recommended picks only)."""
    run = peak = s["starting_bankroll"]
    for p in sorted([p for p in picks if is_main(p) and p["status"] in ("win", "loss", "void")],
                    key=lambda p: p.get("graded", "")):
        run += p["pnl"] or 0
        peak = max(peak, run)
    return run, peak, (peak - run) / peak if peak else 0.0


def is_main(p):
    """Real recommendations (the watchlist is tracked separately)."""
    return p.get("tier") != "W"


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


def research(p, cache, force=False):
    """Search the web for news and expert opinion on a match. Returns a dict (never raises)."""
    empty = {"red_flag": False, "expert_lean": "none", "summary": "", "sources": []}
    hit = cache.get(p["event"])
    if hit and not force and parse_time(hit["t"]) > now_utc() - timedelta(hours=6):
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


QTRS = [f"{q}Q{x}" for q in "1234" for x in ("", "SPREAD", "TOTAL", "WINNER")]
LINE_SERIES = {  # every other Kalshi market on the same game (tennis excluded: rarely offered)
    "mlb": lambda g: [f"KXMLB{x}" for x in ("TOTAL", "SPREAD", "TEAMTOTAL", "F5TOTAL", "F5SPREAD")],
    "nfl": lambda g: [f"KXNFL{x}" for x in ["TOTAL", "SPREAD", "TEAMTOTAL", "1HTOTAL", "1HSPREAD", "1HWINNER",
                                            "2HTOTAL", "2HSPREAD"] + QTRS],
    "cfb": lambda g: [f"KXNCAAF{x}" for x in ["TOTAL", "SPREAD", "TEAMTOTAL", "1HTOTAL", "1HSPREAD", "2H"] + QTRS],
    "soccer": lambda g: [f"KX{g[2:-4]}{x}" for x in ("TOTAL", "SPREAD", "BTTS", "TEAMTOTAL", "1HTOTAL", "1HSPREAD",
                                                      "1HBTTS", "2HTOTAL", "2HSPREAD", "2HBTTS", "SCORE", "FTTS")],
}
PROP_SERIES = {"nfl": list(props.NFL_PROPS), "mlb": list(props.MLB_PROPS)}
UNIT = {"soccer": "goals", "mlb": "runs", "nfl": "points", "cfb": "points"}


def mtype_of(series):
    """(market type, part of the game) from a Kalshi series name."""
    if series.endswith("TEAMTOTAL"):
        return "team_total", None
    if series.endswith("SCORE"):
        return "correct_score", None
    if series.endswith("FTTS"):
        return "first_to_score", None
    hit = re.search(r"(1H|2H|F5|[1-4]Q)(TOTAL|SPREAD|BTTS|WINNER)?$", series)
    if hit:
        kind = {"TOTAL": "total", "SPREAD": "spread", "BTTS": "btts"}.get(hit.group(2), "period_winner")
        return kind, hit.group(1).lower()
    kind = "total" if series.endswith("TOTAL") else "spread" if series.endswith("SPREAD") else \
        "btts" if series.endswith("BTTS") else "winner"
    return kind, None


def prop_outcomes(eng, series, markets):
    """Player props on both sides (over and under)."""
    outs = []
    for m, p, label, note in props.outcomes(eng.props, series, markets):
        under = re.sub(r"(\d+)\+$", lambda x: f"under {x.group(1)}", label) if label.endswith("+") else f"No: {label}"
        for yes, prob, lab in ((True, p, label), (False, 1 - p, under)):
            o = common.Outcome(m, prob, lab, f"{note} Model gives \"{lab}\" {prob:.0%}.")
            o.mtype, o.line, o.yes = f"prop_{series[len('KX' + eng.key.upper()):].lower()}", \
                float(m.get("floor_strike") or 0), yes
            outs.append(o)
    return outs


def line_outcomes(eng, cand, series, markets):
    """Model chances for every other market on the game, on both the Yes and the No side
    (so unders and "doesn't cover" count too)."""
    dist = getattr(cand, "dist", None)
    kind, part = mtype_of(series)
    if dist is None:
        return []
    if part:
        dist = dist.part(part) if hasattr(dist, "part") else None
        if dist is None:
            return []
    group = kind + (f"_{part}" if part else "")
    outs = []
    for m in markets:
        try:
            x = float(m["floor_strike"]) if m.get("floor_strike") is not None else None
            code = re.sub(r"\d+$", "", m["ticker"].split("-")[-1])
            side = getattr(cand, "sides", {}).get(code)
            if kind == "total":
                p = dist.total_over(x)
            elif kind == "period_winner" and side:
                p = dist.margin_over(side, 0)
            elif kind == "correct_score" and hasattr(dist, "grid"):
                sc = re.match(r"([A-Z]+?)(\d+)([A-Z]+?)(\d+)$", m["ticker"].split("-")[-1])
                if not sc:
                    continue
                s1, s2 = getattr(cand, "sides", {}).get(sc.group(1)), getattr(cand, "sides", {}).get(sc.group(3))
                g1, g2 = int(sc.group(2)), int(sc.group(4))
                if not s1 or not s2 or s1 == s2 or max(g1, g2) >= 12:
                    continue
                hg, ag = (g1, g2) if s1 == "home" else (g2, g1)
                p = dist.grid[hg][ag]
            elif kind == "first_to_score" and hasattr(dist, "lh"):
                tot = dist.lh + dist.la
                none = math.exp(-tot)
                p = none if code == "NONE" else (dist.lh if side == "home" else dist.la) / tot * (1 - none) if side else None
                if p is None:
                    continue
            elif kind == "spread" and side:
                p = dist.margin_over(side, x)
            elif kind == "team_total" and side and hasattr(dist, "team_over"):
                p = dist.team_over(side, x)
            elif kind == "btts" and hasattr(dist, "btts"):
                p = dist.btts()
            else:
                continue
        except (TypeError, ValueError, KeyError):
            continue
        label = m.get("yes_sub_title") or m["ticker"]
        no_label = ("Under " + label[5:]) if label.lower().startswith("over ") else \
            re.sub(r" over ", " under ", label) if kind == "team_total" else f"No: {label}"
        for yes, prob, lab in ((True, p, label), (False, 1 - p, no_label)):
            o = common.Outcome(m, prob, lab, f"{dist.describe()} Model gives \"{lab}\" {prob:.0%}.")
            o.mtype, o.line, o.yes = group, x, yes
            outs.append(o)
    return outs


def side_prices(o):
    """Ask and bid for the side of the market this outcome buys."""
    if getattr(o, "yes", True):
        return price(o.market, "yes_ask_dollars"), price(o.market, "yes_bid_dollars")
    return price(o.market, "no_ask_dollars"), price(o.market, "no_bid_dollars")


def watch_kind(p):
    """Watchlist variety buckets: all player props for a sport count as one kind."""
    mt = p.get("mtype", "winner")
    return f"{p.get('model_key', 'tennis')}:{'props' if mt.startswith('prop') else mt.split('_')[0]}"


def make_picks(engines, picks, s, research_cache, sharp, obs, learned):
    already_games = {p.get("game") or p["event"] for p in picks}
    today = now_utc().date().isoformat()
    used_today = sum(p["units"] for p in picks if p["made"][:10] == today and is_main(p))
    league_used = defaultdict(float)
    for p in picks:
        if p["made"][:10] == today and is_main(p):
            league_used[p.get("tour")] += p["units"]
    now = now_utc()
    candidates, legs = [], []
    for eng in engines:
        horizon = s["horizon"].get(eng.key, s["horizon_hours"])
        books = open_markets(list(eng.series))
        line_books = {}
        if eng.key in LINE_SERIES:
            names = sorted({ls for g in eng.series for ls in LINE_SERIES[eng.key](g)})
            line_books = open_markets(names)
        prop_books = open_markets(PROP_SERIES[eng.key]) if getattr(eng, "props", None) and eng.key in PROP_SERIES else {}
        by_game = {}
        for series, events in books.items():
            for event, ms in events.items():
                ms = [m for m in ms if m.get("status") == "active"]
                if len(ms) < 2:
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
                game = event.split("-", 1)[1]
                cand.start, cand.end, cand.game = start, end, game
                by_game[game] = (cand, series)
        # every market for each game: winner plus totals, spreads and both-teams-to-score
        for game, (cand, series) in by_game.items():
            groups = []  # (series, outcomes, normalise mids?)
            for o in cand.outcomes:
                o.mtype, o.line, o.yes = "winner", None, True
            groups.append((series, cand.outcomes, True))
            for ls, events in line_books.items():
                for event, ms in events.items():
                    if event.split("-", 1)[1] == game:
                        ms = [m for m in ms if m.get("status") == "active"]
                        groups.append((ls, line_outcomes(eng, cand, ls, ms), False))
            for ps, events in prop_books.items():
                for event, ms in events.items():
                    if event.split("-", 1)[1] == game:
                        ms = [m for m in ms if m.get("status") == "active"]
                        groups.append((ps, prop_outcomes(eng, ps, ms), False))
            best = None
            for gseries, outs, normalise in groups:
                mids = []
                for o in outs:
                    ask, bid = side_prices(o)
                    mids.append(None if ask is None or bid is None else (ask + bid) / 2)
                if not outs or (normalise and None in mids):
                    continue
                if normalise:
                    tot = sum(mids)
                    if tot <= 0:
                        continue
                    mids = [m / tot for m in mids]
                for o, mid in zip(outs, mids):
                    if mid is None:
                        continue
                    m = o.market
                    group = f"{eng.key}:{o.mtype}"
                    oid = m["ticker"] + ("" if o.yes else ":no")
                    if 0.05 <= mid <= 0.95:
                        ob = learn.observe(obs, oid, group, cand.league, cand.sport, gseries, game,
                                           o.prob, mid, cand.start, now)
                    else:
                        ob = obs.get(oid)
                    default = s["weights"].get(eng.key, s["model_weight"]) if o.mtype == "winner" else \
                        s["line_weights"].get(group, s["min_model_weight"])
                    weight = learn.get_trust(learned, group, cand.league, default)
                    q = learn.calibrate(learned, group, weight * o.prob + (1 - weight) * mid)
                    ask, bid = side_prices(o)
                    vol = price(m, "volume_24h_fp") or 0
                    min_vol = s["min_volume_24h"] if o.mtype == "winner" else s["min_volume_24h"] / 5
                    if not (s["min_price"] <= ask <= s["max_price"]) or ask - bid > s["max_spread"] or vol < min_vol:
                        continue
                    p_eff = ask + fee(ask, s["fee_rate"])
                    edge = q - p_eff
                    ww = max(weight, s["watch_weight"])
                    wq = learn.calibrate(learned, group, ww * o.prob + (1 - ww) * mid)
                    wedge = wq - p_eff
                    score = max(edge, wedge)
                    if not (s["watch_edge"] <= score <= s["max_edge"]) or (best and score <= best["_score"]):
                        continue
                    move = learn.trend(ob)
                    if eng.key == "tennis":
                        finfo, flip, surf = o.why
                        why = explain(finfo, flip, surf, o.prob, q, ask)
                        me = cand.names[1] if flip else cand.names[0]
                        opp = cand.names[0] if flip else cand.names[1]
                        teams, side = list(cand.names), me
                    else:
                        why = f"{o.why} Blended with the market that's {q:.0%} vs a {ask:.0%} price."
                        me, opp = o.label, ""
                        teams = [t.strip() for t in re.split(r" vs\.? | at ", cand.title)][:2]
                        if o.mtype == "spread":
                            side = re.sub(r" wins by .*$", "", o.label)
                            side = common.match_name(side, teams, cutoff=0.5) or side
                        else:
                            side = "Draw" if o.label == "Draw" else o.label.replace(" to win", "")
                    if abs(move) >= 0.02:
                        why += f" Price {'up' if move > 0 else 'down'} {abs(move) * 100:.0f}¢ in the last 12 hours."
                    best = {
                        "id": oid, "ticker": m["ticker"], "side": "yes" if o.yes else "no",
                        "event": m["event_ticker"], "game": game, "sport": cand.sport,
                        "model_key": eng.key, "mtype": o.mtype, "line": o.line, "tour": cand.league,
                        "tournament": cand.title, "matchup": cand.title, "surface": cand.extra.get("surface", ""),
                        "pick": me, "opponent": opp, "market": o.label, "price": round(ask, 2),
                        "model_prob": round(q, 3), "raw_prob": round(o.prob, 3), "edge": round(edge, 3),
                        "tier": tier_of(edge), "units": max(size_units(q, p_eff, s), 0.5), "made": now.isoformat(),
                        "expected_end": cand.end.isoformat(), "start_est": cand.start.isoformat(),
                        "status": "pending", "latest_price": round(ask, 2), "close_price": None, "pnl": None,
                        "why": why, "trend": move, "new_market": bool(ob and ob.get("new")), "trust": weight,
                        "_hint": eng.research_hint, "_teams": teams, "_side": side, "_score": score,
                        "_watch": (round(wq, 3), round(wedge, 3)),
                    }
            if best and game not in already_games:  # one pick per game: no stacking correlated bets
                candidates.append(best)
            if best:
                pass

    candidates.sort(key=lambda p: p["_score"], reverse=True)
    new, searched = [], 0
    watch_today = sum(1 for p in picks if p["made"][:10] == today and not is_main(p))
    watch_kinds = defaultdict(int)
    for p in picks:
        if p["made"][:10] == today and not is_main(p):
            watch_kinds[watch_kind(p)] += 1
    for p in candidates:
        wq, wedge = p.pop("_watch")
        p.pop("_score", None)
        # Compare with sharp sportsbook prices: the most reliable sign of a real edge
        found = sharp.prob(p) if sharp else None
        p_eff = p["price"] + fee(p["price"], s["fee_rate"])
        if found:
            sp, src = found
            q = s["sharp_weight"] * sp + (1 - s["sharp_weight"]) * p["model_prob"]
            p.update(sharp_prob=sp, sharp_source=src, model_prob=round(q, 3), edge=round(q - p_eff, 3))
            p["why"] += f" Sharp sportsbook price ({src}) says {sp:.0%}."
            if sp - p_eff >= s["gap_alert"]:
                p["price_gap"] = round(sp - p_eff, 3)
        else:
            p["sharp_prob"] = None
        bar = s["min_edge"] if found else s["min_edge_no_sharp"]
        against = p["trend"] <= -s["trend_block"]  # price has been sliding away from this side
        # new market types need a sharp price behind them until the agent has learned enough about them
        unproven = p.get("mtype", "winner") != "winner" and not found and \
            learned.get("counts", {}).get(f"{p['model_key']}:{p['mtype']}", 0) < learn.MIN_GROUP
        main_ok = bar <= p["edge"] <= s["max_edge"] and not against and not s["paused"] and not unproven \
            and p["model_prob"] >= s["min_main_prob"]
        if main_ok:
            p["tier"] = tier_of(p["edge"])
            p["units"] = max(size_units(p["model_prob"], p_eff, s), 0.5)
            if used_today + p["units"] > s["daily_unit_cap"] or league_used[p["tour"]] + p["units"] > s["league_daily_cap"]:
                main_ok = False
        if not main_ok:
            # Track the model's own view on the practice watchlist so it learns faster
            kind = watch_kind(p)
            if not (s["watch_edge"] <= wedge <= s["max_edge"]) or watch_today >= s["watch_per_day"] \
                    or watch_kinds[kind] >= s["watch_per_type"]:
                continue
            watch_kinds[kind] += 1
            if found and found[0] + 0.02 < p["price"]:
                continue  # sharp books clearly disagree; not worth tracking
            watch_today += 1
            for k in ("_hint", "_teams", "_side"):
                p.pop(k, None)
            p.update(tier="W", units=0.5, model_prob=wq, edge=wedge, research="", sources=[], expert_lean="none")
            new.append(p)
            continue
        if searched < s["max_research"] or p["event"] in research_cache:
            searched += p["event"] not in research_cache
            info = research(p, research_cache)
        else:
            info = {"red_flag": False, "expert_lean": "none", "summary": "", "sources": []}
        p["research"] = info.get("summary", "")
        p["sources"] = info.get("sources", [])
        p["expert_lean"] = info.get("expert_lean", "none")
        p["_research_hint"] = p.get("_hint")
        for k in ("_hint", "_teams", "_side"):
            p.pop(k, None)
        p.pop("_research_hint", None)
        if info.get("red_flag"):
            print(f"Skipped {p['market']}: news red flag ({p['research']})")
            continue
        if s["skip_expert_disagree"] and p["expert_lean"] == "disagrees":
            print(f"Skipped {p['market']}: experts disagree")
            continue
        used_today += p["units"]
        league_used[p["tour"]] += p["units"]
        new.append(p)
    legs = [dict(p) for p in new]  # parlays are built from this run's recommended and watchlist picks
    return new, legs


# ---------------------------------------------------------------- parlays
PARLAYS_FILE = os.path.join(STATE_DIR, "parlays.json")


def build_parlays(legs, parlays, s):
    """Combine the best legs from different games into parlays with a big payout and positive expected value."""
    import itertools
    today = now_utc().date().isoformat()
    made_today = [x for x in parlays if x["made"][:10] == today]
    if len(made_today) >= s["parlays_per_day"]:
        return []
    used = {leg["id"] for x in made_today for leg in x["legs"]}
    pool, seen = [], set()
    for L in sorted(legs, key=lambda L: L["edge"], reverse=True):
        game = L.get("game") or L.get("event")
        start = parse_time(L.get("start_est")) or (parse_time(L["expected_end"]) - timedelta(hours=3))
        if L.get("edge", 0) >= s["parlay_leg_edge"] and game not in seen and L["id"] not in used \
                and start > now_utc() + timedelta(minutes=30):
            pool.append(L)
            seen.add(game)
    pool = pool[:10]
    options = []
    for n in range(2, s["parlay_max_legs"] + 1):
        for combo in itertools.combinations(pool, n):
            prob = math.prod(L["model_prob"] for L in combo)
            cost = math.prod(L["price"] + fee(L["price"], s["fee_rate"]) for L in combo)
            payout = 1 / cost
            ev = prob * payout - 1
            if prob >= s["parlay_min_prob"] and ev >= s["parlay_min_ev"]:
                options.append((ev, prob, cost, combo))
    options.sort(key=lambda x: x[0] * x[1], reverse=True)  # value, but favour ones that actually hit sometimes
    out, taken_legs = [], set()
    for ev, prob, cost, combo in options:
        if len(made_today) + len(out) >= s["parlays_per_day"]:
            break
        if taken_legs & {L["id"] for L in combo}:
            continue
        taken_legs |= {L["id"] for L in combo}
        out.append({
            "id": f"PARLAY-{now_utc().strftime('%Y%m%d%H%M')}-{len(out) + 1}", "tier": "P", "sport": "Parlay",
            "kind": "recommended" if all(L.get("tier") != "W" for L in combo) else "watchlist",
            "made": now_utc().isoformat(), "units": s["parlay_units"], "price": round(cost, 4),
            "model_prob": round(prob, 4), "ev": round(ev, 3), "payout_units": round(s["parlay_units"] * (1 / cost - 1), 1),
            "status": "pending", "pnl": None,
            "legs": [{k: L[k] for k in ("id", "ticker", "side", "market", "matchup", "sport", "price", "model_prob",
                                        "edge", "expected_end")} for L in combo],
        })
    return out


def grade_parlays(parlays, s):
    graded = []
    for x in parlays:
        if x["status"] != "pending":
            continue
        results = []
        for L in x["legs"]:
            if "result" not in L:
                try:
                    m = kalshi_get(f"/markets/{L['ticker']}").get("market", {})
                except requests.RequestException:
                    results.append(None)
                    continue
                if m.get("status") in ("finalized", "settled", "determined"):
                    r = m.get("result")
                    L["result"] = "win" if r == L["side"] else "loss" if r in ("yes", "no") else "void"
            results.append(L.get("result"))
        if "loss" in results:
            x["status"], x["pnl"] = "loss", -x["units"]
        elif all(r in ("win", "void") for r in results):
            live = [L for L in x["legs"] if L["result"] == "win"]
            cost = math.prod(L["price"] + fee(L["price"], s["fee_rate"]) for L in live) if live else 1
            x["status"] = "win" if live else "void"
            x["pnl"] = round(x["units"] * (1 / cost - 1), 2) if live else 0.0
        else:
            continue
        x["graded"] = now_utc().isoformat()
        graded.append(x)
    return graded


def watch_open_picks(picks, obs, s):
    """Price movement alerts and the pre-game news check for open recommended picks."""
    alerts = []
    now = now_utc()
    for p in picks:
        if p["status"] != "pending" or not is_main(p):
            continue
        o = obs.get(p["id"])
        if o and o.get("h"):
            mid = o["h"][-1][1]
            move = mid - p["price"]
            if move >= s["move_alert"] and not p.get("alert_up"):
                p["alert_up"] = True
                alerts.append(f"📈 {html.escape(p['market'])}: price up to {mid * 100:.0f}¢ from {p['price'] * 100:.0f}¢ "
                              f"(the market is moving your way)")
            elif move <= -s["move_alert"] and not p.get("alert_down"):
                p["alert_down"] = True
                alerts.append(f"📉 {html.escape(p['market'])}: price down to {mid * 100:.0f}¢ from {p['price'] * 100:.0f}¢ "
                              f"(the market is moving against it)")
        start = parse_time(p.get("start_est"))
        if start and not p.get("prechecked") and timedelta(minutes=30) <= start - now <= timedelta(minutes=100):
            p["prechecked"] = True
            info = research(dict(p, _hint="confirmed lineups and starters, late scratches, injuries announced "
                                           "today, weather"), {}, force=True)
            if info.get("red_flag"):
                alerts.append(f"⚠️ Pre-game check on {html.escape(p['market'])}: {html.escape(info.get('summary', ''))}")
            elif info.get("summary"):
                p["pregame"] = info["summary"]
    return alerts


def grade_picks(picks, s):
    graded = []
    for p in picks:
        if p["status"] != "pending":
            continue
        ticker = p.get("ticker", p["id"])
        side = p.get("side", "yes")
        try:
            m = kalshi_get(f"/markets/{ticker}").get("market", {})
        except requests.RequestException:
            try:
                m = kalshi_get(f"/historical/markets/{ticker}").get("market", {})
            except requests.RequestException:
                continue
        status, result = m.get("status"), m.get("result")
        ask, bid = price(m, f"{side}_ask_dollars"), price(m, f"{side}_bid_dollars")
        if status == "active" and ask is not None and bid is not None and ask > 0:
            start = parse_time(p.get("start_est")) or (parse_time(p["expected_end"]) - timedelta(hours=3))
            if now_utc() < start:  # keep the last price seen before the game starts: the "closing" price
                p["close_price"] = round((ask + bid) / 2, 3)
                p["latest_price"] = p["close_price"]
        if status in ("finalized", "settled", "determined"):
            pr, u = p["price"], p["units"]
            fee_units = u * s["fee_rate"] * (1 - pr)
            if result == side:
                p["status"], p["pnl"] = "win", round(u * (1 - pr) / pr - fee_units, 2)
            elif result in ("yes", "no"):
                p["status"], p["pnl"] = "loss", round(-u - fee_units, 2)
            else:
                p["status"], p["pnl"] = "void", 0.0
            p["graded"] = now_utc().isoformat()
            graded.append(p)
    return graded


# ---------------------------------------------------------------- stats
def your_version(p, taken, s):
    """A pick as you actually traded it: your own price if you logged one."""
    t = taken.get(p["id"])
    pr = (t.get("price") if isinstance(t, dict) else None) or p["price"]
    un = (t.get("units") if isinstance(t, dict) else None) or p["units"]
    q = dict(p, price=pr, units=un)
    if p["status"] not in ("win", "loss") or (pr == p["price"] and un == p["units"]):
        return q
    fee_units = un * s["fee_rate"] * (1 - pr)
    q["pnl"] = round(un * (1 - pr) / pr - fee_units, 2) if p["status"] == "win" else round(-un - fee_units, 2)
    return q


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
def telegram(text, buttons=None):
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("[telegram skipped]\n" + text)
        return
    for chunk in [text[i:i + 3900] for i in range(0, len(text), 3900)]:
        try:
            body = {"chat_id": chat, "text": chunk, "parse_mode": "HTML", "disable_web_page_preview": True}
            if buttons:
                body["reply_markup"] = {"inline_keyboard": buttons}
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json=body, timeout=20)
            if not r.ok:  # e.g. a wrong token or chat ID in the GitHub secrets
                print(f"Telegram refused the message ({r.status_code}): {r.text[:300]}")
                r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                                  json={"chat_id": chat, "text": re.sub(r"<[^>]+>", "", chunk)}, timeout=20)
                print("Telegram plain-text retry " + ("worked" if r.ok else f"also failed: {r.text[:300]}"))
            else:
                print("Telegram message sent")
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
def weekly(picks, ctx, s, taken):
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
        learned = ctx.get("learned", {})
        n = learned.get("counts", {}).get(f"{eng.key}:winner", 0)
        if n >= learn.MIN_GROUP:
            new_w = learned["trust"][f"{eng.key}:winner"]
            why = f"{n} finished Kalshi markets it tracked"
        s["weights"][eng.key] = new_w
        if abs(new_w - old) >= 0.1:
            changes.append(f"{eng.sport} ({eng.key.upper()}): trust in the model vs the market changed from "
                           f"{old:.0%} to {new_w:.0%} based on {why}")
    # 3. Tighten or loosen the minimum edge, only with enough evidence
    done = [p for p in picks if p["status"] in ("win", "loss")]
    recent = [p for p in done if is_main(p)][-80:]
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
        "yours_week": summarize([your_version(p, taken, s) for p in week if taken.get(p["id"])]),
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


_OBS = []
_PARLAYS = []


def ctx_learned():
    return load_json(learn.LEARNED_FILE, {})


def write_dashboard(picks, s, reports, changelog, status_note, taken, engines):
    all_picks = picks
    picks = [p for p in all_picks if is_main(p)]
    watch = [p for p in all_picks if not is_main(p)]
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
        "protection": dict(zip(("bankroll", "peak", "drawdown"), [round(v, 3) for v in bankroll_state(all_picks, s)]),
                           paused=s["paused"]),
        "settings": s,
        "open": [p for p in picks if p["status"] == "pending"],
        "history": [dict(p, your_pnl=your_version(p, taken, s)["pnl"], your_price=your_version(p, taken, s)["price"],
                         your_units=your_version(p, taken, s)["units"])
                    if taken.get(p["id"]) else p for p in sorted(done, key=lambda p: p.get("graded", ""), reverse=True)],
        "parlays": {"open": [x for x in (_PARLAYS[0] if _PARLAYS else []) if x["status"] == "pending"],
                    "done": sorted([x for x in (_PARLAYS[0] if _PARLAYS else []) if x["status"] != "pending"],
                                   key=lambda x: x.get("graded", ""), reverse=True)[:100],
                    "summary": summarize(_PARLAYS[0] if _PARLAYS else [])},
        "learning": {"groups": {g: {"finished": n, "trust": ctx_learned().get("trust", {}).get(g),
                                    "confidence_fix": g in ctx_learned().get("calib", {})}
                                for g, n in ctx_learned().get("counts", {}).items()},
                     "price_trend": learn.trend_report(_OBS[0]) if _OBS else {}},
        "worker_url": os.getenv("WORKER_URL", "").rstrip("/"),
        "tracker": tracker.movers(),
        "models": {e.key: dict(public(e.info), sport=e.sport, label=LABELS.get(e.key, e.sport),
                               trust=s["weights"].get(e.key, s["model_weight"])) for e in engines},
        "summary": {"all": summarize(picks),
                    "week": summarize([p for p in done if p.get("graded", "") >= week_ago]),
                    "yours": summarize([your_version(p, taken, s) for p in done if taken.get(p["id"])]),
                    "watch": summarize(watch)},
        "by_expert_lean": by_group(picks, "expert_lean"),
        "by_tier": by_group(all_picks, "tier"),
        "watch_open": [p for p in watch if p["status"] == "pending"],
        "watch_history": sorted([p for p in watch if p["status"] in ("win", "loss", "void")],
                                key=lambda p: p.get("graded", ""), reverse=True)[:300],
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
    obs = learn.load_obs()
    control = load_json(CONTROL_FILE, {})  # written by Telegram commands (/pause, /resume, /quiet, /loud)
    s["paused"] = bool(control.get("paused", s["paused"]))
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
    for e in engines:  # player props for the sports that have free player data
        try:
            if e.key == "nfl":
                e.props = props.NFLProps()
            elif e.key == "mlb":
                e.props = props.MLBProps()
        except Exception:
            print(f"props for {e.key} failed:\n" + traceback.format_exc())
    for e in engines:  # first run for a sport: start from what its history says
        mk = (e.info or {}).get("market")
        if e.key not in s["weights"]:
            s["weights"][e.key] = max(mk["best_weight"], s["min_model_weight"]) if mk else s["model_weight"]
    for e in engines:
        print(f"{e.sport} ({e.key}) model check:", json.dumps(public(e.info), default=float))
        tot = (e.info or {}).get("totals")
        for mt in ("total", "spread", "btts"):
            key = f"{e.key}:{mt}"
            if key not in s["line_weights"]:
                s["line_weights"][key] = max(tot["best_weight"], s["min_model_weight"]) if (tot and mt == "total") \
                    else s["min_model_weight"]
    # Learn from every market it has tracked: grade finished ones, then retune trust and confidence
    graded_obs = learn.resolve(obs, kalshi_all)
    learned = learn.learn(obs, {f"{k}:winner": w for k, w in s["weights"].items()} | s["line_weights"],
                          s["min_model_weight"])
    ctx["learned"] = learned
    old = load_json(learn.LEARNED_FILE, {}).get("trust", {})
    for g, w in learned["trust"].items():
        if g in old and abs(old[g] - w) >= 0.1:
            changelog.append({"date": now_utc().date().isoformat(),
                              "change": f"{g}: trust in the model vs the market moved from {old[g]:.0%} to {w:.0%} "
                                        f"after {learned['counts'][g]} finished markets"})
    save_json(learn.LEARNED_FILE, learned)
    print(f"Learning: graded {graded_obs} more markets; tracking {len(obs)}; groups {learned['counts']}")

    graded = grade_picks(picks, s)
    if graded:
        lines = [f"{'✅' if p['status'] == 'win' else '❌' if p['status'] == 'loss' else '➖'} "
                 f"{html.escape(p['market'])} at {p['price'] * 100:.0f}¢: {p['pnl']:+g}u" for p in graded]
        telegram("<b>Results</b>\n" + "\n".join(lines))

    # Weekly report: first run after 9am Eastern (13:00 UTC) on Monday, once per week
    t = now_utc()
    due = t.weekday() == 0 and t.hour >= 13 and not any(r["week_ending"] == t.date().isoformat() for r in reports)
    if mode == "weekly" or due:
        report, changes = weekly(picks, ctx, s, taken)
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

    alerts = watch_open_picks(picks, obs, s)
    if alerts:
        telegram("\n\n".join(alerts))
    # Bankroll protection brakes
    bank, peak, dd = bankroll_state(picks, s)
    s["_size_factor"] = 0.5 if dd >= s["brake_half"] else 1.0
    if s.get("brake_stop") and dd >= s["brake_stop"] and not s["paused"]:
        s["paused"] = True
        changelog.append({"date": now_utc().date().isoformat(),
                          "change": f"Recommended picks paused: bankroll {bank:.1f}u is {dd:.0%} below its peak of {peak:.1f}u"})
        telegram(f"<b>Bankroll protection</b>\nThe bankroll is {bank:.1f}u, {dd:.0%} below its peak. Recommended picks are "
                 "paused to protect what's left. The watchlist keeps running so the agent can keep learning.")
    sharp = odds.Sharp(s["odds_calls_per_day"], s["odds_markets"])
    new, legs = make_picks(engines, picks, s, research_cache, sharp, obs, learned)
    parlays = load_json(PARLAYS_FILE, [])
    p_graded = grade_parlays(parlays, s)
    for x in p_graded:
        telegram(f"{'🎉' if x['status'] == 'win' else '❌' if x['status'] == 'loss' else '➖'} Parlay "
                 f"({len(x['legs'])} legs) {x['status']}: {x['pnl']:+g}u")
    open_legs = [dict(p) for p in picks if p["status"] == "pending" and p.get("model_prob") and p.get("price")]
    new_parlays = build_parlays(legs + open_legs, parlays, s)
    parlays.extend(new_parlays)
    save_json(PARLAYS_FILE, parlays)
    for x in new_parlays:
        telegram(f"<b>{'Parlay (all legs recommended)' if x.get('kind') == 'recommended' else 'Watchlist parlay'}</b> "
                 f"({len(x['legs'])} legs, {x['units']:g}u to win {x['payout_units']:g}u, practice only)\n"
                 + "\n".join(f"• {html.escape(L['market'])} at {L['price'] * 100:.0f}¢ ({html.escape(L['sport'])}: "
                              f"{html.escape(L['matchup'])})" for L in x["legs"])
                 + f"\n\nModel's chance it hits: {x['model_prob']:.1%}. Expected value +{x['ev']:.0%} if the model is right."
                 "\nBuild it in the Kalshi app if combos are offered for these markets.",
                 buttons=[[{"text": "✅ I took this", "callback_data": f"t|{x['id']}|0"},
                           {"text": "Skip", "callback_data": f"s|{x['id']}"}]])
    sharp.save()
    picks.extend(new)
    main_new = [p for p in new if is_main(p)]
    watch_new = [p for p in new if not is_main(p)]
    for p in main_new:  # one message per pick, with buttons to log it
        telegram(f"<b>New pick</b>{' 🆕 new market' if p.get('new_market') else ''}\n" + pick_line(p)
                 + (f"\n\n{link}" if link else ""),
                 buttons=[[{"text": "✅ I took this", "callback_data": f"t|{p['id']}|{round(p['price'] * 100)}"},
                           {"text": "Skip", "callback_data": f"s|{p['id']}"}]])
    if watch_new:
        telegram(f"<b>Watchlist</b> (smaller edges, practice only)\n"
                 + "\n".join(f"• {html.escape(p['market'])} at {p['price'] * 100:.0f}¢, edge +{p['edge']:.1%} "
                              f"({html.escape(p.get('sport', ''))})" for p in watch_new))
    gaps = [p for p in new if p.get("price_gap")]
    if gaps:
        telegram("<b>Price gaps</b> (Kalshi cheaper than sharp sportsbooks)\n" + "\n".join(
            f"• {html.escape(p['market'])}: Kalshi {p['price'] * 100:.0f}¢ vs sharp {p['sharp_prob'] * 100:.0f}%"
            for p in gaps))
    # hourly check-in, even when nothing new turned up
    open_main = [p for p in picks if p["status"] == "pending" and is_main(p)]
    open_watch = [p for p in picks if p["status"] == "pending" and not is_main(p)]
    today_done = [p for p in picks if is_main(p) and p.get("graded", "")[:10] == now_utc().date().isoformat()
                  and p["status"] in ("win", "loss")]
    td = summarize(today_done)
    if mode == "ask" or not control.get("quiet"):
        telegram(("⏸ Recommended picks are paused (send /resume to restart).\n" if s["paused"] else "")
                 + (f"Hourly check: {len(main_new)} new pick{'' if len(main_new) == 1 else 's'}, "
                    f"{len(watch_new)} new on the watchlist." if new else "Hourly check: nothing new this hour.")
                 + f"\nOpen: {len(open_main)} pick{'' if len(open_main) == 1 else 's'}, {len(open_watch)} watchlist."
                 + (f"\nToday: {td['wins']}-{td['losses']}, {td['units']:+g}u." if td["picks"] else "")
                 + f"\nBankroll {bank:.1f}u.")
    print(f"Graded {len(graded)}, new picks {len(new)}")

    note = f"Last run found {len(new)} new pick(s) and graded {len(graded)}."
    save_json(PICKS_FILE, picks)
    s.pop("_size_factor", None)
    save_json(SETTINGS_FILE, s)
    save_json(REPORTS_FILE, reports)
    save_json(CHANGELOG_FILE, changelog)
    learn.save_obs(obs)
    try:  # watch every other Kalshi sports market, building a price and result history for the future
        tracked = tracker.run(kalshi_all, s["track_series_per_run"])
        print(f"Tracker: {tracked}")
    except Exception:
        print("Tracker failed:\n" + traceback.format_exc())
    save_json(RESEARCH_FILE, dict(sorted(research_cache.items(), key=lambda kv: kv[1].get("t", ""))[-1500:]))
    if "geo" in ctx:
        ctx["geo"].save()
    _OBS.append(obs)
    _PARLAYS.append(parlays)
    write_dashboard(picks, s, reports, changelog, note, taken, engines)


if __name__ == "__main__":
    main()
