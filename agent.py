"""
Picks agent (tennis v1)
-----------------------
Runs on GitHub Actions. Each run it:
  1. Downloads every settled Kalshi tennis match and rebuilds player Elo ratings
  2. Grades any open picks that have finished
  3. Scans today's Kalshi tennis markets for edges and makes new picks
  4. Sends new picks and results to Telegram
  5. Writes data.json for the dashboard
With "weekly" it also writes the weekly report, tunes itself and asks Gemini
for a written reflection.

Usage:  python agent.py run      (normal run)
        python agent.py weekly   (weekly report + learning)
"""
import html
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
STATE_DIR = "state"
PICKS_FILE = os.path.join(STATE_DIR, "picks.json")
SETTINGS_FILE = os.path.join(STATE_DIR, "settings.json")
REPORTS_FILE = os.path.join(STATE_DIR, "reports.json")
CHANGELOG_FILE = os.path.join(STATE_DIR, "changelog.json")
SCANS_FILE = os.path.join(STATE_DIR, "scans.json")
DASHBOARD_FILE = "data.json"

# Men's and women's match series on Kalshi (Challengers share the tour's rating pool)
SERIES = {
    "KXATPMATCH": ("M", "ATP"),
    "KXATPCHALLENGERMATCH": ("M", "ATP Challenger"),
    "KXWTAMATCH": ("W", "WTA"),
    "KXWTACHALLENGERMATCH": ("W", "WTA Challenger"),
}

DEFAULT_SETTINGS = {
    "min_edge": 0.03,          # minimum edge after fees to make a pick
    "kelly_fraction": 0.25,    # quarter Kelly sizing
    "max_units": 3.0,          # biggest single pick
    "daily_unit_cap": 10.0,    # most units risked per day
    "min_matches": 12,         # each player needs this many rated matches
    "min_volume_24h": 500,     # skip thin markets
    "max_spread": 0.04,        # skip markets with a wide bid/ask spread
    "max_price": 0.90,         # skip heavy favourites
    "min_price": 0.10,         # skip long shots
    "fee_rate": 0.07,          # Kalshi taker fee multiplier
    "k_scale": 1.0,            # Elo speed, tuned weekly
    "model_weight": 0.5,       # how much to trust Elo vs the market price, tuned weekly
    "max_edge": 0.15,          # bigger "edges" are almost always model errors
    "starting_bankroll": 100.0,
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


def surface_of(tournament):
    t = re.sub(r"^\d{4} (atp|wta)( challenger)? ", "", tournament.lower())
    has = lambda words: any(re.search(r"\b" + re.escape(w) + r"\b", t) for w in words)
    if has(CLAY):
        return "clay"
    if has(GRASS):
        return "grass"
    return "hard"


def competitor(m):
    return (m.get("custom_strike") or {}).get("tennis_competitor") or m.get("yes_sub_title")


def fee(p, rate):
    return rate * p * (1 - p)


# ---------------------------------------------------------------- data
def fetch_results():
    """Every finished match on Kalshi as (time, pool, winner_id, loser_id, surface, names)."""
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
            "t": win.get("settlement_ts") or win.get("close_time"),
            "pool": pool,
            "w": competitor(win), "l": competitor(lose),
            "wn": win.get("yes_sub_title"), "ln": lose.get("yes_sub_title"),
            "surface": surface_of(tourn), "ev": ev_ticker,
        })
    matches.sort(key=lambda x: x["t"])
    print(f"Loaded {len(matches)} finished matches")
    return matches


# ---------------------------------------------------------------- model
class Elo:
    def __init__(self, k_scale=1.0):
        self.k_scale = k_scale
        self.r = defaultdict(lambda: 1500.0)
        self.n = defaultdict(int)
        self.form = defaultdict(list)
        self.names = {}

    def k(self, n):
        return 250.0 / ((n + 5) ** 0.4) * self.k_scale

    def rating(self, pool, pid, surface):
        o, s = self.r[(pool, pid, "all")], self.r[(pool, pid, surface)]
        return 0.5 * o + 0.5 * s

    def prob(self, pool, a, b, surface):
        ra, rb = self.rating(pool, a, surface), self.rating(pool, b, surface)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update(self, m):
        pool, w, l, s = m["pool"], m["w"], m["l"], m["surface"]
        for key in ("all", s):
            kw, kl = (pool, w, key), (pool, l, key)
            exp = 1.0 / (1.0 + 10 ** ((self.r[kl] - self.r[kw]) / 400.0))
            self.r[kw] += self.k(self.n[kw]) * (1 - exp)
            self.r[kl] -= self.k(self.n[kl]) * (1 - exp)
            self.n[kw] += 1
            self.n[kl] += 1
        for pid, res in ((w, 1), (l, 0)):
            self.form[(pool, pid)] = (self.form[(pool, pid)] + [res])[-10:]
        self.names[(pool, w)], self.names[(pool, l)] = m["wn"], m["ln"]

    def played(self, pool, pid):
        return self.n[(pool, pid, "all")]


def build_elo(matches, k_scale):
    elo = Elo(k_scale)
    for m in matches:
        elo.update(m)
    return elo


def log_loss_for(matches, k_scale, holdout=0.3):
    """Train on older matches, score predictions on the newest 30%."""
    elo = Elo(k_scale)
    cut = int(len(matches) * (1 - holdout))
    loss, n = 0.0, 0
    for i, m in enumerate(matches):
        if i >= cut and elo.played(m["pool"], m["w"]) >= 5 and elo.played(m["pool"], m["l"]) >= 5:
            p = min(max(elo.prob(m["pool"], m["w"], m["l"], m["surface"]), 1e-4), 1 - 1e-4)
            loss -= math.log(p)
            n += 1
        elo.update(m)
    return loss / max(n, 1), n


# ---------------------------------------------------------------- picks
def open_markets():
    """Open match markets grouped into events (one event = one match, two markets)."""
    grouped = defaultdict(list)
    for series in SERIES:
        for m in kalshi_all("/markets", {"series_ticker": series, "status": "open"}):
            grouped[m["event_ticker"]].append(m)
    return [{"event_ticker": k, "markets": v} for k, v in grouped.items()]


def size_units(q, p_eff, s):
    f = (q - p_eff) / (1 - p_eff)
    units = s["kelly_fraction"] * f * 100
    units = min(s["max_units"], round(units * 2) / 2)
    return units if units >= 0.5 else 0.0


def tier_of(edge):
    return "A" if edge >= 0.08 else "B" if edge >= 0.05 else "C"


def form_text(elo, pool, pid):
    f = elo.form[(pool, pid)]
    return f"{sum(f)}-{len(f) - sum(f)} in last {len(f)}" if f else "no recent matches"


def make_picks(elo, picks, s, scans):
    taken = {p["event"] for p in picks}
    today = now_utc().date().isoformat()
    used_today = sum(p["units"] for p in picks if p["made"][:10] == today)
    horizon = now_utc() + timedelta(hours=20)
    candidates = []
    for ev in open_markets():
        ms = [m for m in ev.get("markets", []) if m.get("status") == "active"]
        if ev["event_ticker"] in taken or len(ms) != 2:
            continue
        pool, tour = SERIES.get(ev["event_ticker"].split("-")[0], ("M", ""))
        end = parse_time(ms[0].get("expected_expiration_time") or ms[0].get("occurrence_datetime"))
        if not end or end > horizon or end < now_utc():
            continue
        a, b = ms
        ida, idb = competitor(a), competitor(b)
        if min(elo.played(pool, ida), elo.played(pool, idb)) < s["min_matches"]:
            continue
        tourn = tournament_of(a)
        surf = surface_of(tourn)
        qa = elo.prob(pool, ida, idb, surf)
        mid_a = None
        if price(a, "yes_ask_dollars") is not None and price(a, "yes_bid_dollars") is not None:
            mid_a = (price(a, "yes_ask_dollars") + price(a, "yes_bid_dollars")) / 2
            if now_utc() < end - timedelta(hours=3):  # remember what Elo and the market said
                scans[ev["event_ticker"]] = {"pool": pool, "a": ida, "elo": round(qa, 4),
                                            "mid": round(mid_a, 3), "t": now_utc().isoformat()}
        if mid_a is None:
            continue
        w = s["model_weight"]
        blend_a = w * qa + (1 - w) * mid_a
        best = None
        for m, q, me, opp, opp_id, raw in ((a, blend_a, ida, b, idb, qa), (b, 1 - blend_a, idb, a, ida, 1 - qa)):
            ask, bid = price(m, "yes_ask_dollars"), price(m, "yes_bid_dollars")
            vol = price(m, "volume_24h_fp") or 0
            if ask is None or bid is None or not (s["min_price"] <= ask <= s["max_price"]):
                continue
            if ask - bid > s["max_spread"] or vol < s["min_volume_24h"]:
                continue
            p_eff = ask + fee(ask, s["fee_rate"])
            edge = q - p_eff
            if s["min_edge"] <= edge <= s["max_edge"] and (best is None or edge > best["edge"]):
                units = size_units(q, p_eff, s)
                if units <= 0:
                    continue
                ra = elo.rating(pool, me, surf)
                rb = elo.rating(pool, opp_id, surf)
                best = {
                    "id": m["ticker"], "event": ev["event_ticker"], "sport": "Tennis",
                    "tour": tour, "tournament": tourn, "surface": surf,
                    "pick": m.get("yes_sub_title"), "opponent": opp.get("yes_sub_title"),
                    "market": f"{m.get('yes_sub_title')} to win",
                    "price": round(ask, 2), "model_prob": round(q, 3),
                    "edge": round(edge, 3), "tier": tier_of(edge), "units": units,
                    "made": now_utc().isoformat(), "expected_end": end.isoformat(),
                    "status": "pending", "latest_price": round(ask, 2), "pnl": None,
                    "why": (f"Elo {ra:.0f} vs {rb:.0f} on {surf} gives {raw:.0%}; blended with the "
                            f"market that's {q:.0%} vs a {ask:.0%} price. Form: {form_text(elo, pool, me)} "
                            f"vs {form_text(elo, pool, opp_id)}."),
                }
        if best:
            candidates.append(best)
    candidates.sort(key=lambda p: p["edge"], reverse=True)
    new = []
    for p in candidates:
        if used_today + p["units"] > s["daily_unit_cap"]:
            continue
        used_today += p["units"]
        new.append(p)
    return new


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
        last = price(m, "last_price_dollars")
        if status == "active" and last is not None:
            end = parse_time(p["expected_end"])
            if end and now_utc() < end - timedelta(hours=3):  # before the match likely starts
                p["latest_price"] = round(last, 2)
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
    return {
        "picks": len(done), "wins": wins, "losses": len(done) - wins,
        "units": round(pnl, 2), "staked": round(staked, 2),
        "roi": round(pnl / staked, 3) if staked else 0.0,
        "beat_price": round(sum(p["latest_price"] > p["price"] for p in beat) / len(beat), 3) if beat else None,
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
    return (f"<b>{p['tier']}</b>: {html.escape(p['market'])} at {p['price'] * 100:.0f}¢, "
            f"{p['units']:g}u (model {p['model_prob']:.0%}, edge +{p['edge']:.0%})\n"
            f"   <i>{html.escape(p['opponent'] or '')} · {html.escape(p['tournament'])}</i>")


# ---------------------------------------------------------------- weekly learning
def weekly(picks, matches, s, scans):
    changes = []
    # 1. Re-tune how fast Elo reacts, using every finished match (not just our picks)
    scores = {k: log_loss_for(matches, k)[0] for k in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)}
    best_k = min(scores, key=scores.get)
    if best_k != s["k_scale"] and scores[best_k] < scores.get(s["k_scale"], 9) - 0.002:
        changes.append(f"Elo speed changed from {s['k_scale']} to {best_k} "
                       f"(prediction error {scores.get(s['k_scale'], 0):.4f} to {scores[best_k]:.4f} on recent matches)")
        s["k_scale"] = best_k
    # 2. Re-tune how much to trust Elo vs the market, using every match the agent scanned
    winners = {m["ev"]: m["w"] for m in matches}
    resolved = [(sc["elo"], sc["mid"], winners[ev] == sc["a"]) for ev, sc in scans.items() if ev in winners]
    if len(resolved) >= 200:
        def loss(w):
            tot = 0.0
            for e, mid, won in resolved:
                q = min(max(w * e + (1 - w) * mid, 1e-4), 1 - 1e-4)
                tot -= math.log(q if won else 1 - q)
            return tot / len(resolved)
        wl = {w / 10: loss(w / 10) for w in range(0, 11)}
        best_w = min(wl, key=wl.get)
        if abs(best_w - s["model_weight"]) >= 0.1:
            changes.append(f"Trust in Elo vs market changed from {s['model_weight']:.0%} to {best_w:.0%} "
                           f"after checking {len(resolved)} scanned matches")
            s["model_weight"] = best_w
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

    week_ago = (now_utc() - timedelta(days=7)).isoformat()
    week = [p for p in done if p.get("graded", "") >= week_ago]
    report = {
        "week_ending": now_utc().date().isoformat(),
        "week": summarize(week), "all_time": summarize(done),
        "by_tier": by_group(week, "tier"), "by_tour": by_group(week, "tour"),
        "calibration": calibration(done), "changes": changes,
    }
    prompt = (
        "You are the analyst for a tennis prediction-market model that trades on Kalshi using Elo "
        "ratings. Write a short weekly reflection (under 150 words, plain text, no markdown) for the "
        "owner: what went well, what went badly, likely reasons, and one idea worth testing next. "
        "Be honest about small sample sizes. Data:\n" + json.dumps(report)
        + "\nThis week's picks:\n"
        + json.dumps([{k: p[k] for k in ("market", "tour", "surface", "price", "model_prob", "edge",
                                         "tier", "status", "pnl")} for p in week])
    )
    report["ai_note"] = gemini(prompt) or "AI reflection unavailable this week."
    return report, changes


# ---------------------------------------------------------------- dashboard
def write_dashboard(picks, s, reports, changelog, status_note):
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
        "history": sorted(done, key=lambda p: p.get("graded", ""), reverse=True)[:200],
        "summary": {"all": summarize(picks),
                    "week": summarize([p for p in done if p.get("graded", "") >= week_ago])},
        "by_tier": by_group(picks, "tier"),
        "by_tour": by_group(picks, "tour"),
        "calibration": calibration(picks),
        "curve": curve,
        "reports": reports[-12:][::-1],
        "changelog": changelog[-30:][::-1],
    })


# ---------------------------------------------------------------- main
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    s = {**DEFAULT_SETTINGS, **load_json(SETTINGS_FILE, {})}
    picks = load_json(PICKS_FILE, [])
    reports = load_json(REPORTS_FILE, [])
    changelog = load_json(CHANGELOG_FILE, [])
    scans = load_json(SCANS_FILE, {})
    link = dashboard_url()

    matches = fetch_results()
    elo = build_elo(matches, s["k_scale"])

    graded = grade_picks(picks, s)
    if graded:
        lines = [f"{'✅' if p['status'] == 'win' else '❌' if p['status'] == 'loss' else '➖'} "
                 f"{html.escape(p['market'])} at {p['price'] * 100:.0f}¢: {p['pnl']:+g}u" for p in graded]
        telegram("<b>Results</b>\n" + "\n".join(lines))

    if mode == "weekly":
        report, changes = weekly(picks, matches, s, scans)
        reports.append(report)
        for c in changes:
            changelog.append({"date": now_utc().date().isoformat(), "change": c})
        w = report["week"]
        msg = (f"<b>Weekly report</b>\nRecord {w['wins']}-{w['losses']}, {w['units']:+g}u, "
               f"ROI {w['roi']:.1%}\n"
               + ("\n".join("• " + html.escape(c) for c in changes) + "\n" if changes else "No settings changed.\n")
               + "\n" + html.escape(report["ai_note"]))
        telegram(msg + (f"\n\n{link}" if link else ""))
        elo = build_elo(matches, s["k_scale"])

    new = make_picks(elo, picks, s, scans)
    picks.extend(new)
    if new:
        total = sum(p["units"] for p in new)
        telegram(f"<b>{len(new)} new pick{'s' if len(new) > 1 else ''}</b> ({total:g} units)\n\n"
                 + "\n\n".join(pick_line(p) for p in new) + (f"\n\n{link}" if link else ""))
    print(f"Graded {len(graded)}, new picks {len(new)}")

    note = f"Last run found {len(new)} new pick(s) and graded {len(graded)}."
    save_json(PICKS_FILE, picks)
    save_json(SETTINGS_FILE, s)
    save_json(REPORTS_FILE, reports)
    save_json(CHANGELOG_FILE, changelog)
    save_json(SCANS_FILE, dict(sorted(scans.items(), key=lambda kv: kv[1]["t"])[-6000:]))
    write_dashboard(picks, s, reports, changelog, note)


if __name__ == "__main__":
    main()
