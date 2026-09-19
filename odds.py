"""
Sharp price check
-----------------
Compares Kalshi prices with sportsbook prices (Pinnacle first, the sharpest book, otherwise the
average of the books available) using The Odds API free tier (500 requests a month).
A gap between Kalshi and the sharp price is the most reliable kind of edge, so picks lean on it.

Requests are rationed: only sports that have a pick candidate are checked, each sport is cached
for a few hours, and there's a daily cap so the free monthly allowance lasts.
"""
import os
import re
from datetime import datetime, timedelta, timezone

import common as c

API = "https://api.the-odds-api.com/v4"
CACHE = os.path.join(c.STATE_DIR, "odds_cache.json")
USAGE = os.path.join(c.STATE_DIR, "odds_usage.json")

KEYS = {
    "Premier League": "soccer_epl", "La Liga": "soccer_spain_la_liga", "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga", "Ligue 1": "soccer_france_ligue_one",
    "Liga Portugal": "soccer_portugal_primeira_liga", "Eredivisie": "soccer_netherlands_eredivisie",
    "Belgian Pro League": "soccer_belgium_first_div", "EFL Championship": "soccer_efl_champ",
    "Turkish Super Lig": "soccer_turkey_super_league", "Greek Super League": "soccer_greece_super_league",
    "Scottish Premiership": "soccer_spl", "MLS": "soccer_usa_mls", "Liga MX": "soccer_mexico_ligamx",
    "Brasileirao": "soccer_brazil_campeonato", "Argentina Primera": "soccer_argentina_primera_division",
    "J League": "soccer_japan_j_league", "Danish Superliga": "soccer_denmark_superliga",
    "Eliteserien": "soccer_norway_eliteserien", "Allsvenskan": "soccer_sweden_allsvenskan",
    "Ekstraklasa": "soccer_poland_ekstraklasa", "Swiss Super League": "soccer_switzerland_superleague",
    "Champions League": "soccer_uefa_champs_league", "MLB": "baseball_mlb", "NFL": "americanfootball_nfl",
    "College Football": "americanfootball_ncaaf",
}


def _now():
    return datetime.now(timezone.utc)


class Sharp:
    def __init__(self, daily_cap, markets="h2h,totals,spreads"):
        self.markets = markets
        self.key = os.getenv("ODDS_API_KEY")
        self.cache = c.load_json(CACHE, {})
        self.usage = c.load_json(USAGE, {})
        self.daily_cap = daily_cap
        self.tennis_keys = None

    @property
    def enabled(self):
        return bool(self.key)

    def _budget_ok(self):
        today = _now().date().isoformat()
        if self.usage.get("date") != today:
            self.usage.update(date=today, calls=0)
        remaining = self.usage.get("remaining")
        cost = len(self.markets.split(","))
        return self.usage["calls"] + cost <= self.daily_cap and (remaining is None or remaining > cost + 5)

    def _odds(self, sport_key):
        hit = self.cache.get(sport_key)
        if hit and datetime.fromisoformat(hit["t"]) > _now() - timedelta(hours=3):
            return hit["events"]
        if not self.enabled or not self._budget_ok():
            return hit["events"] if hit else []
        try:
            mk = self.markets if not sport_key.startswith("tennis") else "h2h"
            r = c.get(f"{API}/sports/{sport_key}/odds", params={
                "apiKey": self.key, "regions": "eu", "markets": mk, "oddsFormat": "decimal"}, timeout=30, tries=1)
            self.usage["calls"] += len(mk.split(","))
            rem = r.headers.get("x-requests-remaining")
            if rem is not None:
                self.usage["remaining"] = float(rem)
            events = r.json()
        except Exception as e:
            print(f"odds: {sport_key} unavailable ({e})")
            events = []
        self.cache[sport_key] = {"t": _now().isoformat(), "events": events}
        return events

    def _tennis_key(self, tournament):
        """Find the active tennis tournament key (only main-tour events are covered)."""
        if self.tennis_keys is None:
            try:
                sports = c.get(f"{API}/sports", params={"apiKey": self.key}, timeout=30, tries=1).json()
                self.tennis_keys = [s for s in sports if s.get("group") == "Tennis" and s.get("active")]
            except Exception:
                self.tennis_keys = []
        words = {w for w in re.findall(r"[a-z]+", tournament.lower()) if len(w) > 3} - {"challenger", "qualification"}
        for s in self.tennis_keys:
            text = (s.get("title", "") + " " + s.get("description", "")).lower()
            tour = "wta" if "wta" in tournament.lower() else "atp"
            if tour in s["key"] and any(w in text for w in words):
                return s["key"]
        return None

    def prob(self, pick):
        """Fair (no-vig) sharp chance for this pick's outcome, or None if not available."""
        if not self.enabled:
            return None
        league = pick.get("tour", "")
        key = KEYS.get(league)
        if pick.get("sport") == "Tennis":
            if "Challenger" in league or "Exhibition" in league:
                return None
            key = self._tennis_key(pick.get("tournament", ""))
        if not key:
            return None
        teams, side = pick.get("_teams"), pick.get("_side")
        if pick.get("mtype", "winner") not in ("winner", "total", "spread") or \
                (pick.get("mtype") == "spread" and pick.get("side") == "no"):
            return None
        end = datetime.fromisoformat(pick["expected_end"])
        for ev in self._odds(key):
            names = [ev.get("home_team", ""), ev.get("away_team", "")]
            try:
                start = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if abs((end - start).total_seconds()) > 36 * 3600:
                continue
            if not all(c.match_name(t, names, cutoff=0.75) for t in teams):
                continue
            books = {b["key"]: b for b in ev.get("bookmakers", [])}
            chosen = [books["pinnacle"]] if "pinnacle" in books else list(books.values())
            fair = []
            mtype, line = pick.get("mtype", "winner"), pick.get("line")
            api_key = {"winner": "h2h", "total": "totals", "spread": "spreads"}.get(mtype)
            for b in chosen:
                mk = next((m for m in b.get("markets", []) if m["key"] == api_key), None)
                if not mk:
                    continue
                outs = [o for o in mk["outcomes"] if o.get("price", 0) > 1]
                if mtype == "total":
                    outs = [o for o in outs if o.get("point") == line]
                    want = "over" if pick.get("side", "yes") == "yes" else "under"
                    target = next((o for o in outs if o["name"].lower() == want), None)
                elif mtype == "spread" and pick.get("side", "yes") == "yes":
                    team = c.match_name(side, [o["name"] for o in outs], cutoff=0.75)
                    mine = next((o for o in outs if o["name"] == team and o.get("point") == -line), None)
                    other = next((o for o in outs if o["name"] != team and o.get("point") == line), None)
                    outs, target = ([mine, other], mine) if mine and other else ([], None)
                elif side == "Draw":
                    target = next((o for o in outs if o["name"].lower() == "draw"), None)
                else:
                    name = c.match_name(side, [o["name"] for o in outs if o["name"].lower() != "draw"], cutoff=0.75)
                    target = next((o for o in outs if o["name"] == name), None)
                total = sum(1 / o["price"] for o in outs)
                if target and total > 0:
                    fair.append((1 / target["price"]) / total)
            if fair:
                return round(sum(fair) / len(fair), 4), ("Pinnacle" if "pinnacle" in books else f"{len(fair)} books")
        return None

    def save(self):
        c.save_json(CACHE, self.cache)
        c.save_json(USAGE, self.usage)
