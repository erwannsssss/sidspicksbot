"""Shared helpers used by every sport."""
import difflib
import json
import math
import os
import re
import time
import unicodedata

import numpy as np
import requests

STATE_DIR = "state"
UA = {"User-Agent": "Mozilla/5.0 (picks-agent; personal research)"}


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


def fresh(meta_path, hours):
    """True if the cache at meta_path was refreshed within `hours`."""
    return time.time() - load_json(meta_path, {}).get("refreshed", 0) < hours * 3600


def mark_fresh(meta_path):
    save_json(meta_path, {"refreshed": time.time()})


def get(url, params=None, headers=None, timeout=60, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
            if r.status_code == 429:
                time.sleep(3 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ---------------------------------------------------------------- team names
DROP = {"fc", "cf", "afc", "sc", "club", "de", "cd", "ud", "sd", "sv", "vfb", "vfl", "tsg", "rc",
        "ss", "us", "calcio", "the", "fk", "sk", "if", "bk", "ac", "as", "ca", "cr", "ec", "se",
        "and", "ssc", "ogc", "losc", "rcd", "ca", "cp", "sl", "united"}
ALIASES = {
    "man united": "manchester united", "man utd": "manchester united", "man city": "manchester city",
    "spurs": "tottenham", "tottenham hotspur": "tottenham", "wolves": "wolverhampton",
    "wolverhampton wanderers": "wolverhampton", "nottm forest": "nottingham forest",
    "nottingham": "nottingham forest", "newcastle": "newcastle", "west ham united": "west ham",
    "brighton and hove albion": "brighton", "sheffield weds": "sheffield wednesday",
    "ath madrid": "atletico madrid", "atletico": "atletico madrid", "ath bilbao": "athletic bilbao",
    "athletic club": "athletic bilbao", "betis": "real betis", "sociedad": "real sociedad",
    "espanol": "espanyol", "celta": "celta vigo", "vallecano": "rayo vallecano",
    "inter": "inter milan", "internazionale": "inter milan", "milan": "ac milan",
    "m gladbach": "monchengladbach", "mgladbach": "monchengladbach", "gladbach": "monchengladbach",
    "borussia monchengladbach": "monchengladbach", "ein frankfurt": "eintracht frankfurt",
    "frankfurt": "eintracht frankfurt", "fc koln": "koln", "cologne": "koln", "psg": "paris saint germain",
    "paris sg": "paris saint germain", "paris": "paris saint germain", "st etienne": "saint etienne",
    "bayern": "bayern munich", "bayern munchen": "bayern munich", "leverkusen": "bayer leverkusen",
    "dortmund": "borussia dortmund", "leipzig": "rb leipzig", "hertha": "hertha berlin",
    "stuttgart": "vfb stuttgart", "mainz": "mainz 05", "hoffenheim": "tsg hoffenheim",
    "sporting lisbon": "sporting cp", "sp lisbon": "sporting cp", "sporting": "sporting cp",
    "psv": "psv eindhoven", "az": "az alkmaar", "ajax": "ajax amsterdam",
    "club brugge": "club brugge", "brugge": "club brugge", "salzburg": "rb salzburg",
    "red bull salzburg": "rb salzburg", "celtic": "celtic", "rangers": "rangers",
    "america": "club america", "guadalajara": "chivas", "pumas unam": "pumas", "unam": "pumas",
    "la galaxy": "los angeles galaxy", "lafc": "los angeles fc", "ny red bulls": "new york red bulls",
    "nycfc": "new york city", "new york city fc": "new york city",
    "stade rennais": "rennes", "enschede": "twente", "heart of midlothian": "hearts",
    "saint louis": "st louis city", "st louis": "st louis city", "czestochowa": "rakow",
    "alkmaar": "az alkmaar", "eindhoven": "psv eindhoven", "rotterdam": "feyenoord",
    "gliwice": "piast gliwice", "katowice": "gks katowice", "bremen": "werder bremen",
    "vicente barcelos": "gil vicente", "frontale": "kawasaki frontale", "marinos": "yokohama f marinos",
    "avispa": "avispa fukuoka", "fagiano o": "fagiano okayama", "shimizu": "shimizu s pulse",
    "goztepe izmir": "goztepe", "basaksehir": "buyuksehyr", "istanbul basaksehir": "buyuksehyr",
    "kocaeli": "kocaelispor", "erzurum": "erzurumspor", "panaitolikos": "panetolikos",
    "levadiakos": "levadeiakos", "leuven": "oud heverlee leuven", "paranaense": "athletico pr",
    "athletico paranaense": "athletico pr", "atletico mineiro": "atletico mg", "flamengo": "flamengo rj",
    "barracas": "barracas central", "rosario": "rosario central", "gimnasia la plata": "gimnasia l p",
    "san luis": "atl san luis", "new york rb": "new york red bulls", "miami": "inter miami",
    "kansas city": "sporting kansas city", "new england": "new england revolution",
    "rb bragantino": "bragantino", "los angeles g": "los angeles galaxy", "los angeles f": "los angeles fc",
    "tokyo v": "verdy", "sparta": "sparta rotterdam", "schalke": "schalke 04",
}


def canon(name):
    s = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("'", ""))
    s = re.sub(r"\s+", " ", s).strip()
    s = ALIASES.get(s, s)
    toks = [t for t in s.split() if t not in DROP]
    return " ".join(toks) or s


def match_name(name, candidates, cutoff=0.72):
    """Best match for a team name among candidate names (returns the candidate or None)."""
    if not candidates:
        return None
    c = canon(name)
    table = {}
    for cand in candidates:
        table.setdefault(canon(cand), cand)
    if c in table:
        return table[c]
    toks = set(c.split())
    # all tokens of one name contained in the other ("Inter" / "Inter Milan")
    subset = [k for k in table if toks and (toks <= set(k.split()) or set(k.split()) <= toks)]
    if len(subset) == 1:
        return table[subset[0]]
    # one name starts with or contains the other ("Kocaeli" / "Kocaelispor")
    contain = [k for k in table if len(c) >= 4 and (k.startswith(c) or c.startswith(k) or c in k)]
    if len(contain) == 1:
        return table[contain[0]]
    best = difflib.get_close_matches(c, list(table), n=1, cutoff=cutoff)
    return table[best[0]] if best else None


# ---------------------------------------------------------------- models
def fit_logistic(X, y, lam=2.0, iters=15):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(X.shape[1])
        w -= np.linalg.solve(H, g)
    return w


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def evaluate_binary(X, y, mkt, raw_price, feature_names, base_cols=(0,), base_coef=None,
                    min_edge=0.03, max_edge=0.15, min_weight=0.25, lam=2.0, min_price=0.10):
    """
    Shared training + honest testing for two-outcome sports.
    X rows are (team A vs team B) features, y = 1 if A won. mkt = fair market chance for A
    (or None), raw_price = price you'd actually pay for A (or None).
    Trains on the oldest 85%, tests on the newest 15%, compares with the market, backtests,
    then refits on everything. Falls back to the single rating column if the full model
    doesn't beat it.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    info = {"training_rows": len(y)}
    if len(y) < 800:
        coef = np.zeros(X.shape[1])
        if base_coef is not None:
            coef[:len(base_coef)] = base_coef
        return coef, dict(info, using="rating")
    cut = int(len(y) * 0.85)
    w_train = fit_logistic(X[:cut], y[:cut], lam)
    bc = list(base_cols)
    base_train = fit_logistic(X[:cut][:, bc], y[:cut], lam)
    Xh, yh = X[cut:], y[cut:]
    p_full = 1 / (1 + np.exp(-Xh @ w_train))
    p_base = 1 / (1 + np.exp(-Xh[:, bc] @ base_train))
    ll_full, ll_base = log_loss(p_full, yh), log_loss(p_base, yh)
    use_full = ll_full < ll_base - 0.001
    pp = p_full if use_full else p_base
    info["holdout"] = {"matches": int(len(yh)),
                       "rating_log_loss": round(ll_base, 4), "model_log_loss": round(ll_full, 4),
                       "rating_accuracy": round(float(np.mean((p_base > .5) == (yh == 1))), 3),
                       "model_accuracy": round(float(np.mean((p_full > .5) == (yh == 1))), 3)}
    info["using"] = "full" if use_full else "rating"
    idx = [i for i in range(cut, len(y)) if mkt[i] is not None and raw_price[i] is not None]
    if len(idx) > 300:
        pm = np.array([mkt[i] for i in idx])
        price = np.array([raw_price[i] for i in idx])
        pj = pp[[i - cut for i in idx]]
        yy = y[idx]
        grid = {k / 10: log_loss(k / 10 * pj + (1 - k / 10) * pm, yy) for k in range(11)}
        best = min(grid, key=grid.get)
        bw = max(best, min_weight)
        info["market"] = {"market_log_loss": round(log_loss(pm, yy), 4),
                          "model_log_loss": round(log_loss(pj, yy), 4), "best_weight": best,
                          "backtest_weight": bw, "matches": len(idx)}
        info["backtest"] = backtest(bw * pj + (1 - bw) * pm, price, yy, min_edge, max_edge, min_price)
        info["_bt"] = {"model": pj, "market": pm, "price": price, "won": yy}
    if use_full:
        coef = fit_logistic(X, y, lam)
    else:
        coef = np.zeros(X.shape[1])
        coef[bc] = fit_logistic(X[:, bc], y, lam)
    info["weights"] = {f: round(float(c), 3) for f, c in zip(feature_names, coef)}
    return coef, info


def backtest(q, price, won, min_edge, max_edge, min_price=0.10, max_price=0.90):
    """Bet every game the live rules would bet, at the given prices, and total the result."""
    bets = wins = 0
    staked = pnl = 0.0
    for qi, pi, yi in zip(q, price, won):
        edge = qi - pi
        if min_edge <= edge <= max_edge and min_price <= pi <= max_price:
            stake = min(3.0, max(0.5, round(0.25 * edge / (1 - pi) * 100 * 2) / 2))
            bets += 1
            staked += stake
            if yi == 1:
                pnl += stake * (1 - pi) / pi
                wins += 1
            else:
                pnl -= stake
    return {"bets": int(bets), "wins": int(wins), "units": round(float(pnl), 1),
            "roi": round(float(pnl / staked), 3) if staked else 0.0}


class Outcome:
    """One side of a Kalshi event that the agent could buy."""
    def __init__(self, market, prob, label, why):
        self.market, self.prob, self.label, self.why = market, prob, label, why


class Candidate:
    """A Kalshi event the model has an opinion on."""
    def __init__(self, event, sport, league, title, outcomes, extra=None):
        self.event, self.sport, self.league, self.title = event, sport, league, title
        self.outcomes, self.extra = outcomes, extra or {}
