[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Szeven/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Szeven/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01

Forecast GTA VI's launch-day Twitch concurrent viewers, hour by hour
(hours 2–24), as five quantiles per hour, from the hour-1 print plus the
case's history and 38–42 comparable past launches.

Entry for the make.fun competition `gta6-twitch-ccu-forecast-01`.

## Program

`main.py` (Python 3.13, standard library only, no network). Run:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<input dir> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` in the input directory and writes one `<id>.out` per
case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Model (summary)

Per-hour ridge regression on the log scale,
`log v_h ~ a_h + b_h·log(p1) + c_h·log(pre24max)`, fit on the comparables
with a Gaussian kernel in launch scale (bandwidth 2.0 log units of the
hour-1 print), so a huge launch is predicted from huge launches. Spread =
shrunk weighted residual sd. See `NOTES.md` for the full description and
the practice results (mean pinball loss **0.160** vs house baseline
**0.387**).

## License

MIT — see `LICENSE`.

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
