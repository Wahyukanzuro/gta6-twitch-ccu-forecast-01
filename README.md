# GTA VI launch-day Twitch CCU forecaster

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Wahyukanzuro/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Wahyukanzuro/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) on [make.fun](https://make.fun).

## What it does

`main.py` (Python 3.13, standard library only) reads a case file
(`/input/<id>.in`) and writes a forecast (`/output/<id>.out`): for each of
hours 2–24 of the release day, five quantiles (5/25/50/75/95%) of the Twitch
category's concurrent viewers.

## Method — ratio-curve ensemble

The program fits itself at runtime from the case's own comparables (past big
launches); nothing is hardcoded:

1. Each comparable launch contributes its release-day curve normalised by its
   own hour-0 viewers: `r_h = viewers(h) / viewers(0)`.
2. Each comparable is weighted by similarity to the target: a Gaussian kernel
   on log hour-0 viewers (bandwidth 1.25), ×1.5 for rolling-midnight unlocks,
   ×1.5 for PS5-only launches. The target's own pre-release history
   ("Not a launch") is excluded.
3. Per hour, the forecast is the hour-1 print × the weighted median ratio;
   the five quantiles are the weighted empirical quantiles of the ratio
   distribution at that hour.
4. An hour with fewer than 3 usable ratio curves falls back to the house
   persistence baseline.

## Validation (honest, leave-one-out)

- Leave-one-out over the 37 comparable launches, mean pinball loss on the log
  scale: **0.41** over all launches, **0.26** over big launches (hour-0 ≥
  20k). Unweighted median: 0.72 / 0.37. Tested and rejected: hype weighting
  (no gain), a per-hour bias correction (hurts: 0.30), lognormal spreads
  (worse on big launches).
- Practice (4 real launches): mean loss **0.246**, vs the house baseline's
  0.387; typical miss 47%, 90% coverage 0.935.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs on the same case write identical bytes.

## License

MIT — see [LICENSE](LICENSE).
