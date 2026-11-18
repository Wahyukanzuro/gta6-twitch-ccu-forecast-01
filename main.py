"""Hour-by-hour launch-day viewership forecaster (v4).

The program reads one case file at a time from the input directory and
writes a 23-row quantile forecast (hours 2..24) for each. Everything is
fitted at run time from the case's own comparable launches; nothing about
any specific launch is baked in.

Model, in brief:
  * Each comparable launch contributes a ratio curve: viewers at day hour
    h divided by viewers at day hour 0. The target's hour-1 print anchors
    every forecast, so the whole job is predicting ratios.
  * Two predictors are blended per day hour. A weighted-median ensemble
    over the ratio curves (similarity weights: log-scale kernel times a
    rolling-midnight bonus, a console-only bonus, and an early-access
    penalty). And a two-predictor weighted least-squares regression of
    log-ratio on log hour-0 viewers plus a pre-release momentum feature
    (log growth of the category in the 24 h before launch versus the
    48 h before that), with spreads taken from the fit residuals.
  * The blend leans on the median early in the day and on the regression
    in the evening, following a smooth ramp. The target's log hour-0
    viewers are capped at the largest comparable's (never floored), so a
    record print cannot extrapolate the regression off a cliff.
  * Pre-release momentum for the target is measured by combining two
    sources in hourly bins: the case's own history (minute reads) and
    the target game's own pre-release history among the comparables,
    which reaches much further back. Bins prefer the case history; the
    comparable fills bins the history does not cover. If momentum cannot
    be measured, each hour's regression falls back to that hour's
    training weighted mean momentum.
  * The blended ratio curve is lightly smoothed across neighbouring
    day hours (the true curve is smooth apart from the unlock jump),
    and any hour the ensemble cannot predict falls back to the
    persistence baseline (hour-1 print with a widening log spread).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.15  # persistence spread, log units per sqrt hour

SCALE_BW = 1.5       # log-scale kernel bandwidth for similarity weights
BLEND_W = 0.75       # regression weight in the blend (evening end)
MOM_LO, MOM_HI = -1.0, 4.0   # winsor bounds on momentum
MIN_REG_N = 10       # launches needed to fit one hour's regression
MIN_MED_N = 3        # ratio points needed for one hour's median
MAX_Q = 1e9          # largest quantile the format allows


def parse_ts(text):
    """Parse an ISO timestamp; None when the text is malformed.

    A bad timestamp never aborts a case: the point is skipped and the
    momentum/fallback logic deals with the thinner data.
    """
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except Exception:
        return None


def round2(x):
    """Round to 2 decimals, halves away from zero; whole values as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def wquantiles(pairs, levels):
    """Weighted empirical quantiles of (value, weight) pairs.

    Sorts by value, walks the cumulative weight, and linearly
    interpolates between the two bracketing points at each level.
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
    for lv in levels:
        if lv <= cum[0]:
            out.append(pairs[0][0])
            continue
        if lv >= cum[-1]:
            out.append(pairs[-1][0])
            continue
        for i in range(1, len(pairs)):
            if cum[i] >= lv:
                c0, c1 = cum[i - 1], cum[i]
                v0, v1 = pairs[i - 1][0], pairs[i][0]
                t = 0.0 if c1 == c0 else (lv - c0) / (c1 - c0)
                out.append(v0 + t * (v1 - v0))
                break
    return out


def persistence_row(p1, hour):
    """House baseline row for case hour `hour`: the hour-1 print with a
    log spread widening like a random walk."""
    spread = SIGMA * math.sqrt(hour - 1)
    return [round2(p1 * math.exp(z * spread)) for z in Z]


def clamp_row(q):
    """Force a quantile row valid and non-decreasing: finite, in range,
    each value at least the previous one."""
    fixed = []
    prev = 0.01
    for v in q:
        if v is None or not math.isfinite(v):
            v = prev
        v = min(max(v, 0.01), MAX_Q)
        if v < prev:
            v = prev
        fixed.append(v)
        prev = v
    return fixed


class Launch:
    """One comparable launch: hourly viewers on release day plus the
    pre-release series that feeds the momentum feature."""

    def __init__(self, comp):
        self.name = comp.get("name") or ""
        self.day_start = parse_ts(comp.get("day_start"))
        self.unlock = comp.get("unlock") or ""
        self.platforms = comp.get("platforms") or ""
        self.hourly = {}
        self.pre = {}
        for h in comp.get("hourly") or []:
            t = parse_ts(h.get("from"))
            if t is None or self.day_start is None:
                continue
            dh = (t - self.day_start).total_seconds() / 3600.0
            v = h.get("viewers")
            if not v:
                continue
            if 0 <= dh < 24:
                self.hourly[int(dh)] = v
            elif -336 <= dh < 0:
                self.pre[int(dh)] = v
        self.momentum = self._momentum()

    @property
    def h0(self):
        return self.hourly.get(0)

    def ratio(self, day_hour):
        """Viewers at day hour `day_hour` over viewers at hour 0."""
        if 0 not in self.hourly or day_hour not in self.hourly:
            return None
        v0 = self.hourly[0]
        v = self.hourly[day_hour]
        if not v0 or v0 <= 0 or not v or v <= 0:
            return None
        return v / v0

    def _momentum(self):
        """Log growth of the category into launch: mean viewers in the
        24 h before day start over mean viewers in the 48 h before that.
        None when either window is too thin to trust."""
        late = [v for dh, v in self.pre.items() if -24 <= dh < 0]
        early = [v for dh, v in self.pre.items() if -72 <= dh < -24]
        if len(late) < 12 or len(early) < 24:
            return None
        m_late = sum(late) / len(late)
        m_early = sum(early) / len(early)
        if m_late <= 0 or m_early <= 0:
            return None
        return math.log(m_late / m_early)


def find_target_history(case):
    """The comparable that measures the TARGET game's own pre-release
    category (not a launch): its name contains the target game's name,
    or its unlock notes mark it as the target's history. None when the
    case carries no such comparable."""
    game = (case.get("game") or {}).get("name") or ""
    for comp in case.get("comparables") or []:
        cname = comp.get("name") or ""
        if game and game.lower() in cname.lower():
            return comp
        if "target-history" in (comp.get("unlock") or "").lower():
            return comp
    return None


def load_pool(case):
    """Training launches: every comparable except the target's own
    history. Momenta are winsorised to [MOM_LO, MOM_HI]."""
    target_hist = find_target_history(case)
    pool = []
    for comp in case.get("comparables") or []:
        if comp is target_hist:
            continue
        launch = Launch(comp)
        if launch.h0 is None or launch.h0 <= 0:
            continue
        if launch.momentum is not None:
            launch.momentum = min(max(launch.momentum, MOM_LO), MOM_HI)
        pool.append(launch)
    return pool, target_hist


def _binned_mean(hist_pts, comp_pts, day_start, lo, hi):
    """Time-weighted mean over the hourly bins in [lo, hi) hours before
    day start. Each bin takes the case history's mean when it has one,
    else the comparable's mean. Returns (mean, bins covered) or
    (None, 0) when no bin has data."""
    bins = {}
    for pts, tag in ((hist_pts, "h"), (comp_pts, "c")):
        for t, v in pts:
            if not v:
                continue
            dh = (t - day_start).total_seconds() / 3600.0
            if lo <= dh < hi:
                bins.setdefault(math.floor(dh), {}).setdefault(tag, []).append(v)
    vals = []
    for k in range(int(lo), int(hi)):
        b = bins.get(k)
        if not b:
            continue
        if "h" in b:
            vals.append(sum(b["h"]) / len(b["h"]))
        else:
            vals.append(sum(b["c"]) / len(b["c"]))
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def target_momentum(case, target_hist):
    """The target's pre-release momentum, combining the case history
    with the target game's own pre-release history. The late window
    (last 24 h) comes from the history; the early window (72 h to 24 h
    before launch) is filled bin by bin from whichever source covers
    it. The history alone is often truncated to the last couple of
    days, which biases the early mean up and the momentum down; the
    comparable reaches back to the premiere and removes that bias.
    None when either window is too thin (late < 12 of 24 bins, early
    < 24 of 48 bins)."""
    day_start = parse_ts(case.get("day_start"))
    if day_start is None:
        return None
    hist_pts = []
    for h in case.get("history") or []:
        t = parse_ts(h.get("at"))
        if t is not None and h.get("viewers"):
            hist_pts.append((t, h["viewers"]))
    comp_pts = []
    if target_hist is not None:
        for h in target_hist.get("hourly") or []:
            t = parse_ts(h.get("from"))
            if t is not None and h.get("viewers"):
                comp_pts.append((t, h["viewers"]))
    late, n_late = _binned_mean(hist_pts, comp_pts, day_start, -24, 0)
    early, n_early = _binned_mean(hist_pts, comp_pts, day_start, -72, -24)
    if late is None or early is None or late <= 0 or early <= 0:
        return None
    if n_late < 12 or n_early < 24:
        return None
    return math.log(late / early)


def similarity(launch, target_h0):
    """Similarity weight of a comparable for a target of hour-0 size
    `target_h0`: a Gaussian kernel on log scale, times a bonus for
    rolling-midnight unlocks, a bonus for console-only launches, and a
    penalty for early-access-contaminated ones."""
    if launch.h0 is None or launch.h0 <= 0:
        return 0.0
    d = abs(math.log(launch.h0) - math.log(max(target_h0, 1)))
    w = math.exp(-0.5 * (d / SCALE_BW) ** 2)
    u = (launch.unlock or "").lower()
    if "rolling" in u or "midnight local" in u:
        w *= 1.5
    if "ps5 only" in (launch.platforms or "").lower():
        w *= 1.5
    if "early-access-contaminated" in u:
        w *= 0.3
    return w


def median_forecast(pool, p1, target_h0):
    """Per day hour: weighted empirical quantiles of the comparables'
    ratio curves, scaled by the target's hour-1 print."""
    out = {}
    for d in range(0, 24):
        pairs = []
        for launch in pool:
            r = launch.ratio(d)
            if r is None or r <= 0:
                continue
            w = similarity(launch, target_h0)
            if w <= 0:
                continue
            pairs.append((r, w))
        if len(pairs) < MIN_MED_N:
            out[d] = None
            continue
        qs = wquantiles(pairs, LEVELS)
        out[d] = [p1 * q for q in qs] if qs else None
    return out


def _solve(matrix):
    """Gaussian elimination with partial pivoting on an augmented
    square matrix. Returns the solution vector, or None when the
    system is singular."""
    n = len(matrix)
    m = [row[:] for row in matrix]
    for i in range(n):
        piv = max(range(i, n), key=lambda k: abs(m[k][i]))
        m[i], m[piv] = m[piv], m[i]
        if abs(m[i][i]) < 1e-9:
            return None
        for k in range(i + 1, n):
            f = m[k][i] / m[i][i]
            for j in range(i, n + 1):
                m[k][j] -= f * m[i][j]
    coef = [0.0] * n
    for i in range(n - 1, -1, -1):
        coef[i] = (m[i][n] - sum(m[i][j] * coef[j] for j in range(i + 1, n))) / m[i][i]
    return coef


def regress_forecast(pool, p1, target_h0, mom_target):
    """Per day hour: weighted least squares of log-ratio on log hour-0
    viewers plus momentum, evaluated at the target. Launches missing
    momentum get that hour's training weighted mean imputed. When the
    target's momentum is unknown, the same weighted mean is used as
    the target's value. The target's log size is capped at the largest
    comparable's, never floored. Spreads are the weighted quantiles of
    the fit residuals, so the interval follows the data, not a guess."""
    xt = math.log(max(target_h0, 1))
    out = {}
    for d in range(0, 24):
        pts = []
        for launch in pool:
            r = launch.ratio(d)
            if r is None or r <= 0 or launch.h0 is None or launch.h0 <= 0:
                continue
            w = similarity(launch, target_h0)
            if w <= 0:
                continue
            pts.append((math.log(launch.h0), launch.momentum, math.log(r), w))
        if len(pts) < MIN_REG_N:
            out[d] = None
            continue
        known = [m for _, m, _, _ in pts if m is not None]
        if known:
            swk = sum(w for _, m, _, w in pts if m is not None)
            mmean = sum(w * m for _, m, _, w in pts if m is not None) / swk
        else:
            mmean = 0.0
        pts = [(x, m if m is not None else mmean, y, w) for x, m, y, w in pts]
        xs = [x for x, _, _, _ in pts]
        xt_c = min(xt, max(xs))
        sw = sum(w for _, _, _, w in pts)
        mx = sum(w * x for x, _, _, w in pts) / sw
        mm = sum(w * m for _, m, _, w in pts) / sw
        my = sum(w * y for _, _, y, w in pts) / sw
        mt = mm if mom_target is None else mom_target
        a_mat = [[0.0] * 3 for _ in range(3)]
        b_vec = [0.0] * 3
        for x, m, y, w in pts:
            row = (1.0, x - mx, m - mm)
            for i in range(3):
                b_vec[i] += w * row[i] * y
                for j in range(3):
                    a_mat[i][j] += w * row[i] * row[j]
        coef = _solve([a_mat[i] + [b_vec[i]] for i in range(3)])
        if coef is None:
            out[d] = None
            continue
        mu = coef[0] + coef[1] * (xt_c - mx) + coef[2] * (mt - mm)
        resid = [(y - (coef[0] + coef[1] * (x - mx) + coef[2] * (m - mm)), w)
                 for x, m, y, w in pts]
        qs = wquantiles(resid, LEVELS)
        if qs is None:
            out[d] = None
            continue
        ratios = sorted(math.exp(mu + e) for e in qs)
        out[d] = [p1 * q for q in ratios]
    return out


def blend_weight(day_hour):
    """Regression weight in the blend: the median rules the morning,
    the regression rules the evening, with a smooth ramp between."""
    if day_hour <= 10:
        return 0.5
    if day_hour >= 20:
        return 0.9
    return 0.5 + 0.4 * (day_hour - 10) / 10


def blend_forecast(reg, med):
    """Log-space blend of the two predictors, per day hour."""
    out = {}
    for d in sorted(set(reg) | set(med)):
        a, b = reg.get(d), med.get(d)
        if a is None and b is None:
            out[d] = None
        elif b is None:
            out[d] = a
        elif a is None:
            out[d] = b
        else:
            w = blend_weight(d)
            out[d] = [math.exp(w * math.log(x) + (1.0 - w) * math.log(y))
                      for x, y in zip(a, b)]
    return out


def smooth_curve(preds):
    """Lightly smooth each quantile level across neighbouring day
    hours (kernel .25/.5/.25 on log values). The ratio curve is
    smooth apart from the unlock jump, and per-hour fits are noisy;
    the smoothing trades a little jump sharpness for a steadier curve.
    Missing hours are skipped, edge hours renormalise."""
    out = {}
    for d in range(0, 24):
        if preds.get(d) is None:
            out[d] = None
            continue
        seg = [d]
        if d - 1 >= 0 and preds.get(d - 1) is not None:
            seg.append(d - 1)
        if d + 1 <= 23 and preds.get(d + 1) is not None:
            seg.append(d + 1)
        wts = {d: 0.5}
        for s in seg:
            if s != d:
                wts[s] = 0.25
        tot = sum(wts.values())
        smoothed = []
        for qi in range(5):
            v = sum(wts[s] * math.log(preds[s][qi]) for s in seg) / tot
            smoothed.append(math.exp(v))
        out[d] = sorted(smoothed)
    return out


def forecast_case(case):
    """Full forecast for one case: 23 rows for case hours 2..24."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])
    pool, target_hist = load_pool(case)
    mom = target_momentum(case, target_hist)
    if mom is not None:
        mom = min(max(mom, MOM_LO), MOM_HI)
    reg = regress_forecast(pool, p1, p1, mom)
    med = median_forecast(pool, p1, p1)
    preds = smooth_curve(blend_forecast(reg, med))
    rows = []
    for hour in range(2, 25):
        q = preds.get(hour - 1)
        if q is None:
            rows.append({"hour": hour, "q": persistence_row(p1, hour)})
        else:
            rows.append({"hour": hour, "q": clamp_row([round2(x) for x in q])})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without a valid file: fall back to the
            # persistence baseline around whatever hour-1 print we can find.
            p1 = 1
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                first = next((k for k in case.get("known") or []
                              if k.get("hour") == 1), None)
                if first is not None:
                    p1 = max(1, first["viewers"])
            except Exception:
                pass
            forecast = {"hours": [{"hour": h, "q": persistence_row(p1, h)}
                                  for h in range(2, 25)]}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
