# Sid's Picks Bot

An agent that scans Kalshi sports markets every hour, prices them with its own models, compares them with
sharp sportsbook prices, and sends picks to Telegram. It tracks every pick, learns from every market it sees,
and shows everything on a dashboard.

**Practice first.** Backtests have not shown a proven edge. Let the practice record (closing line value and
ROI over a few hundred picks) prove itself before real money goes in.

---

## What it covers

| Sport | Markets it prices (both Yes and No sides) |
|---|---|
| Tennis | Match winner: ATP, WTA, both Challenger tours, Davis Cup, exhibitions |
| Soccer (22 leagues + Champions League) | Winner/draw, total goals, team totals, spreads, both teams to score, 1st and 2nd half totals/spreads/BTTS, correct score, first team to score |
| NFL | Winner, spread, total, team totals, halves, quarters (spread, total, winner), player props: passing/rushing/receiving yards, receptions, passing TDs, anytime and 2+ TDs |
| College football | Winner, spread, total, team totals, halves, quarters (needs the free CFBD key) |
| MLB | Winner, run line, total, team totals, first 5 innings, player props: pitcher strikeouts, hits, home runs, total bases, hits+runs+RBIs |
| Parlays | Up to 2 a day, built from the strongest legs across different games |
| Everything else on Kalshi | 3,800+ sports series (NBA, NHL, golf, F1, UFC, esports, goalscorers, corners, futures...) are watched and their price/result history recorded, ready for future models |

## Files

| File | Where it goes | What it does |
|---|---|---|
| agent.py | repository main folder | The main program: runs every hour |
| model.py | main folder | Tennis data and model |
| soccer.py | main folder | Soccer data and model (goals, winner, all soccer markets) |
| football.py | main folder | NFL and college football models |
| mlb.py | main folder | Baseball model |
| props.py | main folder | Player prop models (NFL, MLB) |
| learn.py | main folder | Learns from every market it tracks |
| tracker.py | main folder | Watches every other Kalshi sports market |
| odds.py | main folder | Sharp sportsbook price check (The Odds API) |
| common.py | main folder | Shared helpers |
| index.html | main folder | The dashboard |
| requirements.txt | main folder | Python libraries to install |
| daily.yml | `.github/workflows/` | The hourly schedule |
| worker.js | Cloudflare (picks-helper) | Telegram commands, buttons, dashboard checkboxes |

## Secrets

GitHub → Settings → Secrets and variables → Actions:

| Name | Where to get it | Required? |
|---|---|---|
| TELEGRAM_BOT_TOKEN | Telegram @BotFather → /mybots → your bot → API Token | Yes |
| TELEGRAM_CHAT_ID | Telegram @userinfobot | Yes |
| GEMINI_API_KEY | aistudio.google.com → Get API key | Yes (web research) |
| WORKER_URL | Your Cloudflare worker address (ends in .workers.dev) | Yes (dashboard checkboxes) |
| ODDS_API_KEY | the-odds-api.com, free Starter plan | Strongly recommended |
| CFBD_API_KEY | collegefootballdata.com, free | Only for college football |

Cloudflare → Workers & Pages → picks-helper → Settings → Variables and Secrets:
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, GITHUB_TOKEN (fine-grained, sidspicksbot only, Actions + Contents read/write),
GITHUB_REPO (`erwannsssss/sidspicksbot`), WEBHOOK_SECRET (made up, letters and numbers), DASHBOARD_PASSCODE (made up).

## Telegram commands

| Command | What it does |
|---|---|
| any new picks? | Open picks now, then a fresh scan |
| /today | Open picks (numbered) and today's results |
| /week | This week, all time, your trades, bankroll, closing line value |
| /soccer /football /baseball /tennis | Open picks for one sport |
| /why 2 | Full reasoning for pick 2 |
| /paid 47 | Fix the price you paid on the last pick you logged |
| /pause, /resume | Stop or restart recommended picks |
| /quiet, /loud | Hourly check-ins off or on |

Buttons under each pick: **✅ I took this** (logs it at the listed price) and **Skip**.

## Pick types

- **Recommended picks (tiers A, B, C):** need a 3%+ edge backed by a sharp sportsbook price, or 5%+ without one,
  at least a 40% chance, no injury red flags, and no strong price slide. New market types (props, quarters,
  correct score...) need a sharp price behind them until the agent has graded 150 of that type.
- **Watchlist (W):** smaller edges, practice only, never counted in the bankroll. This is where most learning happens.
- **Parlays (P):** tracked separately. Only take one if Kalshi's combo pays at least the listed amount.

## Key settings (state/settings.json)

| Setting | Default | Meaning |
|---|---|---|
| min_edge / min_edge_no_sharp | 0.03 / 0.05 | Edge needed with / without a sharp price |
| min_price | 0.20 | Skip long shots below 20¢ |
| daily_unit_cap / max_units | 8 / 8 | Most units a day / on one pick |
| league_daily_cap | 4 | Most units a day in one league |
| kelly_fraction | 0.125 | Bet sizing (eighth Kelly) |
| brake_half | 0.10 | Halve bet sizes 10% below the peak |
| brake_stop | null | Set to 0.20 to pause picks after a 20% drop |
| parlays_per_day / parlay_units | 2 / 1 | Parlay suggestions and stake |
| watch_per_day | 20 | Most watchlist picks a day |
| odds_calls_per_day | 15 | Keeps the free Odds API plan from running out |

Edit a value on GitHub (pencil icon), commit, and it applies on the next run.

## Troubleshooting

- **Red X on a run:** open the run → Run agent → copy the last lines of the log.
- **No Telegram messages:** run the workflow in **ask** mode; the log says whether Telegram accepted the message.
- **Dashboard not updating:** press Ctrl+Shift+R; check Actions for a failed "pages build and deployment".
- **Buttons or commands don't respond:** check the Cloudflare worker has the latest worker.js and its secrets,
  and that `getWebhookInfo` shows your `.workers.dev/telegram` address.
- **GitHub token expired** (Telegram commands stop working): make a new fine-grained token and paste it into the
  worker's GITHUB_TOKEN secret, then Deploy.
