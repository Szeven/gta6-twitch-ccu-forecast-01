# gta6-twitch-ccu-forecast-01 — notes

Entry for make.fun competition `gta6-twitch-ccu-forecast-01` (Forecast GTA VI's
launch-day Twitch viewers hour by hour, $250). Account @luthfi-007, payout
wallet `GrHtSvTcmjHHWwZM7VsNPgZWoTwPnLxgpMDYfH6iUPXe`
(see `.makefun/owner.json`).

## Model (`main.py`, Python 3.13 stdlib only, no network)

Per-hour, scale-local ridge regression on the log scale. For each forecast
hour h = 2..24, fit over the case's comparable past launches:

    log v_h  ~  a_h + b_h * log(p1) + c_h * log(pre24max)

- `p1` = the target launch's hour-1 print; `pre24max` = largest hourly viewer
  count in the 24 h before the release day (from the case's own
  `history_hourly`). Hour 1 (00:00–01:00 UTC) is a depressed hour — late night
  in Europe, US unlocks still hours away — while pre-launch hype (streams
  from New Zealand/Australia, where the game already unlocked) sets the day's
  scale. The regression learns per hour how much weight each anchor deserves:
  early hours load almost entirely on p1, late hours spread weight onto the
  pre-launch peak and the intercept (small launches grow many-fold by
  evening, huge ones stay near their hour-1 level).
- The fit is **local in launch scale**: training launches are kernel-weighted
  by log-distance between their hour-1 print and the target's (Gaussian
  kernel, bandwidth 2.0 log units), so a huge launch is predicted mostly from
  huge launches. A global fit is dragged around by simultaneous-global-unlock
  launches whose hour 1 is a near-zero pre-launch baseline and whose day then
  grows 100–1000x — a regime GTA VI, a rolling-midnight console launch, is
  not in. Early-access-contaminated and weak-tier comparables are
  down-weighted (0.5 / 0.3, parsed from the `unlock` text).
- The GTA VI category's own pre-release history (a comparable in every case)
  is excluded from training: it is not a launch day.
- Spread: the fit's weighted residual sd per hour, floored at 0.15 log units,
  capped at 1.5, shrunk by 0.7 (the raw residuals mix in the global-unlock
  launches and overstate the spread for big rolling launches). Quantiles are
  `exp(mu_h + z_tau * sd_h)`, z = standard normal quantiles at
  0.05/0.25/0.5/0.75/0.95, rounded to 2 decimals half-up.
- Fallbacks: if a case has too few usable comparables, or any step fails, the
  program writes the house persistence forecast for that case instead of
  crashing — a case is never left invalid. Deterministic: pure arithmetic, no
  randomness; two runs write byte-identical files.

Things tried in leave-one-out validation over the 37 usable comparables and
rejected: plain median-ratio curves, empirical quantiles, diurnal carryover
(previous-day same-hour anchor — bad, pre-launch hype is a different regime),
OLS without the scale kernel, extra predictors (pre-7d max, channels, pre
slope — no gain), blending with the ratio model.

## Practice-case results

`makefun grade-local` cannot grade this competition locally: the kit's
program/v1 scorer and reference data are not published, and the CLI answers
only "it is graded on the server only" (RULES.txt agrees: "There is no local
grade of this competition: the server grades every entry"). So the practice
answers were reconstructed from the same public source the kit cites:
SullyGnome's hourly "Average viewers" series for the four practice launches
(the endpoint pattern in the kit's `input/NOTICE.txt`), read via the browser.
Each reconstruction is anchored on the case file itself — the 6 pre-checkpoint
hourly values plus the hour-1 print match the case's `history_hourly` /
`known` exactly. Cross-check: scoring the house baseline starter against
these answers gives loss **0.387097**, typical miss 82.7%, coverage 0.783 —
the kit's stated baseline figures (0.387 / 82.8% / 0.783).

Mean pinball loss on the log scale (lower is better), all 23 hours x 5 levels:

| case | launch | this entry | house baseline |
|------|--------|-----------|----------------|
| p00 | Marvel's Wolverine (2026-09-15) | 0.171330 | 0.743671 |
| p01 | Red Dead Redemption 2 (2018-10-26) | 0.175904 | 0.286813 |
| p02 | God of War Ragnarök (2022-11-09) | 0.111835 | 0.214579 |
| p03 | Marvel's Spider-Man 2 (2023-10-20) | 0.179305 | 0.303324 |
| **mean** | | **0.159593** | **0.387097** |

This entry: typical miss 28.1%, 90% coverage 0.935. House baseline: typical
miss 82.7%, coverage 0.783. The entry beats the house baseline on every
practice case and by 0.228 on the mean.

Leave-one-out validation over the 37 usable comparables (same model, each
comparable as target, trained on the other 36): mean loss ~0.39 overall,
~0.35 on rolling-midnight launches, ~0.21 on launches with hour-1 >= 100k,
vs ~0.93 / ~0.71 / ~0.26 for the house baseline on the same splits.

## How to run / verify

```sh
# all practice cases (from the repo root; needs the kit's input/ dir)
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py

# one case at a time, as the grader does (identical output for that case)
mkdir -p out1
MAKEFUN_INPUT_DIR=<dir with only p02.in> MAKEFUN_OUTPUT_DIR=out1 python3 main.py
```

Checks performed 2026-11-18 (local Python 3.12; grader uses 3.13, code uses
no version-specific features):

- Format: every `.out` is `{"hours": [...]}` with exactly 23 rows, hours
  2–24 each once, each row exactly `{"hour", "q"}` with 5 finite quantiles,
  each > 0 and <= 1e9, non-decreasing. Max file size 1,555 bytes (limit
  65,536).
- Determinism: two full runs produce byte-identical outputs (`diff -r`
  clean).
- Speed: ~0.9 s for all 4 practice cases together (limit 8 s per case).
- Single-case run produces output identical to the same case in the full run.
- Degenerate input (comparables removed) still yields a valid forecast via
  the persistence fallback (hour-2 median = hour-1 print).

## Commits

- Code commit (main.py, LICENSE, .makefun/owner.json):
  `6b7d2e68a0b981665f2796ba0fef179e28c8e171`
- This NOTES.md + README.md were added in the follow-up commit; see
  `git log` for the final HEAD.

## Files

- `main.py` — the entry (Python 3.13, stdlib only).
- `LICENSE` — MIT.
- `.makefun/owner.json` — `{"wallet": "GrHtSvTcmjHHWwZM7VsNPgZWoTwPnLxgpMDYfH6iUPXe"}`.
- `NOTES.md` — this file.
- `README.md` — short entry summary.
