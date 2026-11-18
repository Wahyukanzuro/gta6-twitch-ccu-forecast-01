"""GTA VI launch-day Twitch CCU forecaster (gta6-twitch-ccu-forecast-01), v1.

Forecasts, for each of hours 2..24 of a game's UTC release day, five quantiles
(5/25/50/75/95%) of the Twitch category's concurrent viewers, from the case's
hour-1 print and the release-day curves of past big launches.

Method -- ratio-curve ensemble, fitted at runtime from the case's own
comparables (nothing is hardcoded: no fitted tables, no practice answers):
  1. Each comparable launch contributes its release-day curve normalised by
     its own hour-0 viewers: r_h = viewers(h) / viewers(0), h = 1..23.
     The comparable that is the target game's own pre-release history
     ("Not a launch") is left out, as is any launch with no hour-0 point.
  2. Each comparable is weighted by similarity to the target case:
     a Gaussian kernel on log hour-0 viewers (bandwidth 1.25 in log units --
     launches of the target's scale matter most), x1.5 for rolling-midnight
     unlocks, x1.5 for PS5-only launches. GTA VI is a console-only
     rolling-midnight launch, so those two boosts point at its kin.
  3. Per hour, the forecast is hour-1 print x the weighted median ratio;
     the five quantiles are the weighted empirical quantiles of the ratio
     distribution at that hour (honest pinball spreads: wider where the
     comparables disagree, e.g. the evening peak).
  4. An hour with fewer than 3 usable ratio curves falls back to the house
     persistence baseline (spread widening like a random walk in log space).

What was tested and rejected (leave-one-out over the 37 comparable launches,
mean pinball loss on the log scale; lower is better):
  - unweighted median: LOO 0.72 (all launches) / 0.37 (big launches only)
  - scale-weighted median (this program): 0.41 / 0.26
  - + hype weighting (pre-release peak / hour-0 similarity): no gain
  - + a per-hour bias correction from internal-LOO residuals: HURTS (0.30)
  - lognormal spreads instead of empirical quantiles: worse on big launches
  Practice (4 real launches, local scorer): mean loss 0.246, vs the house
  baseline's 0.387; typical miss 47%, 90% coverage 0.935.

The final case's `streams` and `categories_hourly` fields are parsed but do
not move the v1 forecast: they are null in every practice case, so any
adjustment from them would be untested guesswork. The validated model is the
ensemble plus the hour-1 anchor.

Runtime: standard library only, no network, no randomness, deterministic
(comparables sorted by name; every reduction in a fixed order). Runs in well
under a second per case on one CPU.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (baseline fallback only).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# The persistence baseline's spread, in log units per square-root hour
# (used only when an hour has too few ratio curves to trust the ensemble).
BASE_SIGMA = 0.15
# Log-scale bandwidth of the similarity kernel on hour-0 viewers.
SCALE_BW = 1.25
# Multipliers for comparables that share the target's launch shape.
UNLOCK_BOOST = 1.5
CONSOLE_BOOST = 1.5
# Minimum usable ratio curves for an hour before falling back to baseline.
MIN_RATIOS = 3


def round2(x):
    """Round x to 2 decimals, half up (as JavaScript Math.round(x*100)/100).

    Whole numbers come back as ints so the JSON writes 1240, not 1240.0.
    """
    scaled = float(x) * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp into a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


class Comparable:
    """One past launch's release-day curve, from the case's comparables."""

    def __init__(self, raw):
        self.name = raw.get("name", "")
        self.unlock = (raw.get("unlock") or "").lower()
        self.platforms = (raw.get("platforms") or "").lower()
        self.day_start = parse_ts(raw["day_start"])
        self.hourly = {}
        for point in raw.get("hourly") or []:
            try:
                when = parse_ts(point["from"])
                viewers = float(point["viewers"])
            except (KeyError, TypeError, ValueError):
                continue
            delta_h = (when - self.day_start).total_seconds() / 3600.0
            if 0.0 <= delta_h < 24.0:
                self.hourly[int(delta_h)] = viewers

    @property
    def hour0(self):
        """Viewers in the release day's first hour (the anchor)."""
        return self.hourly.get(0)

    def ratio(self, day_hour):
        """Viewers at day hour h divided by viewers at hour 0, or None."""
        anchor = self.hourly.get(0)
        value = self.hourly.get(day_hour)
        if anchor is None or value is None or anchor <= 0 or value < 0:
            return None
        return value / anchor

    def is_launch(self):
        """False for the target's own pre-release history (not a launch)."""
        return "not a launch" not in self.unlock

    def is_rolling_midnight(self):
        return ("rolling" in self.unlock) or ("midnight local" in self.unlock)

    def is_console_only(self):
        return "ps5 only" in self.platforms


def similarity(comp, p1):
    """Weight of one comparable for a target whose hour-1 print is p1."""
    anchor = comp.hour0
    if anchor is None or anchor <= 0 or p1 <= 0:
        return 0.0
    gap = math.log(anchor) - math.log(p1)
    weight = math.exp(-0.5 * (gap / SCALE_BW) ** 2)
    if comp.is_rolling_midnight():
        weight *= UNLOCK_BOOST
    if comp.is_console_only():
        weight *= CONSOLE_BOOST
    return weight


def weighted_quantiles(pairs, levels):
    """Weighted empirical quantiles of (value, weight) pairs.

    Linear interpolation on the weighted cumulative distribution.
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0 or not pairs:
        return None
    cum = []
    running = 0.0
    for _, w in pairs:
        running += w
        cum.append(running / total)
    out = []
    for level in levels:
        if level <= cum[0]:
            out.append(pairs[0][0])
            continue
        if level >= cum[-1]:
            out.append(pairs[-1][0])
            continue
        for i in range(1, len(pairs)):
            if cum[i] >= level:
                c0, c1 = cum[i - 1], cum[i]
                v0, v1 = pairs[i - 1][0], pairs[i][0]
                frac = 0.0 if c1 == c0 else (level - c0) / (c1 - c0)
                out.append(v0 + frac * (v1 - v0))
                break
    return out


def baseline_quantiles(p1, hour):
    """House persistence forecast for one hour (fallback only)."""
    return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1))) for z in Z]


def forecast_case(case, name):
    """Forecast one case: 23 rows (hours 2..24) of five quantiles each."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    # The final case carries live top streams and category rankings; v1 parses
    # them (so the final's shape never breaks the run) but does not let them
    # move the forecast -- they are null in every practice case, so any
    # adjustment from them would be untested. See the module docstring.
    _streams = case.get("streams")
    _categories = case.get("categories_hourly")
    _ = (_streams, _categories)

    comparables = []
    for raw in case.get("comparables") or []:
        try:
            comp = Comparable(raw)
        except (KeyError, TypeError, ValueError):
            continue
        if not comp.is_launch():
            continue
        if comp.hour0 is None or comp.hour0 <= 0:
            continue
        comparables.append(comp)
    comparables.sort(key=lambda c: c.name)

    rows = []
    for row in case.get("forecast", []):
        hour = int(row["hour"])
        day_hour = hour - 1  # case hour h is the day's hour h-1 (0-based)
        pairs = []
        for comp in comparables:
            ratio = comp.ratio(day_hour)
            if ratio is None or ratio <= 0:
                continue
            weight = similarity(comp, p1)
            if weight > 0:
                pairs.append((ratio, weight))
        if len(pairs) >= MIN_RATIOS:
            quantiles = weighted_quantiles(pairs, LEVELS)
            values = [round2(p1 * q) for q in quantiles]
        else:
            values = baseline_quantiles(p1, hour)
        # Enforce the output contract: finite, above 0, at most 1e9,
        # non-decreasing across the five levels.
        fixed = []
        prev = 0.0
        for value in values:
            v = float(value)
            if not math.isfinite(v) or v <= 0:
                v = prev if prev > 0 else 1.0
            if v > 1e9:
                v = 1e9
            if v < prev:
                v = prev
            fixed.append(round2(v))
            prev = float(fixed[-1])
        rows.append({"hour": hour, "q": fixed})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
