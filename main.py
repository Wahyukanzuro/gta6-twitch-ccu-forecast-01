"""Launch-day Twitch concurrent-viewer forecaster (v3).

Reads each case file in $MAKEFUN_INPUT_DIR (default /input) and writes one
forecast per case to $MAKEFUN_OUTPUT_DIR (default /output): 23 rows, hours
2..24, five quantiles each (0.05/0.25/0.5/0.75/0.95).

Model (fitted at runtime from the case's own comparables; nothing about any
specific game, launch or answer is baked in):
  * Each comparable launch is normalised by its hour-1 print into a ratio
    curve r_h = viewers(h) / viewers(0).
  * Two predictors per day-hour, blended in log space with a smoothstep
    ramp (median early in the day, regression late, where the scale effect
    is strongest):
      - a weighted-median ensemble of the ratio curves. Weights: Gaussian
        kernel on log hour-1 viewers (bw 1.5) x1.5 rolling-midnight unlock,
        x1.5 PS5-only, x0.3 early-access-contaminated. Spreads are weighted
        empirical quantiles of the ratios.
      - a local two-predictor weighted least-squares fit of log-ratio on
        log hour-1 viewers and pre-release momentum (the category's viewer
        growth over the three days before launch, winsorised to [-1, +4]),
        evaluated at the target's hour-1 print. The target x is capped at
        the largest training hour-1 print (one-sided clamp: a record print
        never drives the line off a cliff, while small launches extrapolate
        freely). Spreads are weighted quantiles of the fit residuals.
  * Fallbacks per hour: regression -> median -> house persistence baseline.
  * Deterministic, standard library only.
"""
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
BW = 1.5            # scale-kernel bandwidth on log hour-1 viewers
MOM_LO, MOM_HI = -1.0, 4.0   # pre-release momentum winsor range
MIN_REG_N = 10      # min points for the regression at one day-hour
MIN_MED_N = 3       # min points for the median ensemble at one day-hour


def parse_ts(s):
    """Tolerant ISO-8601 parser: returns an aware datetime or None.
    Accepts a non-zero-padded hour ('T0:00:00') as well as proper input."""
    if not isinstance(s, str):
        return None
    t = s.strip()
    for cand in (t, re.sub(r"T(\d):", r"T0\1:", t)):
        try:
            dt = datetime.fromisoformat(cand.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def wquantiles(pairs, levels):
    """Weighted empirical quantiles; pairs = [(value, weight)].
    Linear interpolation on the cumulative weight (type-7 style)."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0 or not pairs:
        return None
    cum = []
    s = 0.0
    for _, w in pairs:
        s += w
        cum.append(s / tot)
    res = []
    for lv in levels:
        if lv <= cum[0]:
            res.append(pairs[0][0])
            continue
        if lv >= cum[-1]:
            res.append(pairs[-1][0])
            continue
        for i in range(1, len(pairs)):
            if cum[i] >= lv:
                c0, c1 = cum[i - 1], cum[i]
                v0, v1 = pairs[i - 1][0], pairs[i][0]
                f = 0.0 if c1 == c0 else (lv - c0) / (c1 - c0)
                res.append(v0 + f * (v1 - v0))
                break
    return res


def is_rolling_midnight(unlock):
    u = (unlock or "").lower()
    return ("rolling" in u) or ("midnight local" in u)


def is_early_access(unlock):
    return "early-access-contaminated" in (unlock or "").lower()


def clip_mom(m):
    return min(max(m, MOM_LO), MOM_HI)


class Launch:
    """One comparable launch: release-day hourly viewers + pre-release history."""

    def __init__(self, c):
        self.name = c.get("name") or ""
        self.day_start = parse_ts(c.get("day_start") or "")
        self.unlock = c.get("unlock") or ""
        self.platforms = c.get("platforms") or ""
        self.hourly = {}
        self.pre = []  # (hours before day_start, viewers), up to 14 days back
        if self.day_start is not None:
            for h in c.get("hourly") or []:
                t = parse_ts(h.get("from") or "")
                v = h.get("viewers")
                if t is None or not v or v <= 0:
                    continue
                dh = (t - self.day_start).total_seconds() / 3600.0
                if 0 <= dh < 24:
                    self.hourly[int(dh)] = v
                elif -336 <= dh < 0:
                    self.pre.append((dh, v))
        self.mom = self._momentum()

    @property
    def h0(self):
        return self.hourly.get(0)

    def ratio(self, h):
        if 0 not in self.hourly or h not in self.hourly:
            return None
        return self.hourly[h] / self.hourly[0]

    def _momentum(self):
        """Log growth of the category's viewers into launch: mean of the
        24h before day_start over mean of the 72h-24h window before it.
        None when the history is too thin to say."""
        late = [v for dh, v in self.pre if -24 <= dh < 0]
        early = [v for dh, v in self.pre if -72 <= dh < -24]
        if len(late) < 12 or len(early) < 24:
            return None
        ml = sum(late) / len(late)
        me = sum(early) / len(early)
        if ml <= 0 or me <= 0:
            return None
        return math.log(ml / me)


def target_momentum(case):
    """Pre-release momentum of the case's own category, from its `history`
    field (points before the checkpoint). Same windows as Launch._momentum."""
    day_start = parse_ts(case.get("day_start") or "")
    if day_start is None:
        return None
    late, early = [], []
    for h in case.get("history") or []:
        t = parse_ts(h.get("at") or "")
        v = h.get("viewers")
        if t is None or not v or v <= 0:
            continue
        dh = (t - day_start).total_seconds() / 3600.0
        if -24 <= dh < 0:
            late.append(v)
        elif -72 <= dh < -24:
            early.append(v)
    if len(late) < 12 or len(early) < 24:
        return None
    ml = sum(late) / len(late)
    me = sum(early) / len(early)
    if ml <= 0 or me <= 0:
        return None
    return math.log(ml / me)


def weight(launch, target_h0):
    """Similarity weight of a comparable launch for a target hour-1 print."""
    h0 = launch.h0
    if h0 is None or h0 <= 0:
        return 0.0
    d = abs(math.log(h0) - math.log(max(target_h0, 1)))
    w = math.exp(-0.5 * (d / BW) ** 2)
    if is_rolling_midnight(launch.unlock):
        w *= 1.5
    if "ps5 only" in launch.platforms.lower():
        w *= 1.5
    if is_early_access(launch.unlock):
        w *= 0.3
    return w


def predict_median(train, p1, target_h0):
    """Weighted-median ratio ensemble + empirical-quantile spreads."""
    out = {}
    for d in range(0, 24):
        pairs = []
        for launch in train:
            r = launch.ratio(d)
            if r is None or r <= 0:
                continue
            w = weight(launch, target_h0)
            if w <= 0:
                continue
            pairs.append((r, w))
        if len(pairs) < MIN_MED_N:
            out[d] = None
            continue
        qs = wquantiles(pairs, LEVELS)
        out[d] = [p1 * q for q in qs] if qs else None
    return out


def predict_regress(train, p1, target_h0, mom_target):
    """Per day-hour WLS: log-ratio ~ a + b1*log(h0) + b2*momentum.
    Momenta winsorised; target x capped at the largest training log-h0
    (one-sided clamp). Spreads are weighted quantiles of fit residuals."""
    xt = math.log(max(target_h0, 1))
    out = {}
    for d in range(0, 24):
        raw = []
        for launch in train:
            r = launch.ratio(d)
            if r is None or r <= 0 or launch.h0 is None or launch.h0 <= 0:
                continue
            w = weight(launch, target_h0)
            if w <= 0:
                continue
            raw.append((math.log(launch.h0), launch.mom, math.log(r), w))
        if len(raw) < MIN_REG_N:
            out[d] = None
            continue
        known = [clip_mom(m) for _, m, _, _ in raw if m is not None]
        if known:
            swk = sum(w for _, m, _, w in raw if m is not None)
            mmean = sum(w * clip_mom(m) for _, m, _, w in raw
                        if m is not None) / swk
        else:
            mmean = 0.0
        pts = [(x, clip_mom(m) if m is not None else mmean, y, w)
               for x, m, y, w in raw]
        xs = [x for x, _, _, _ in pts]
        xt_c = min(xt, max(xs))  # one-sided clamp (high only)
        sw = sum(w for _, _, _, w in pts)
        mx = sum(w * x for x, _, _, w in pts) / sw
        mm = sum(w * m for _, m, _, w in pts) / sw
        my = sum(w * y for _, _, y, w in pts) / sw
        # 3x3 weighted normal equations for [1, x-mx, m-mm]
        A = [[0.0] * 3 for _ in range(3)]
        bv = [0.0] * 3
        for x, m, y, w in pts:
            row = (1.0, x - mx, m - mm)
            for i in range(3):
                bv[i] += w * row[i] * y
                for j in range(3):
                    A[i][j] += w * row[i] * row[j]
        M = [A[i] + [bv[i]] for i in range(3)]
        ok = True
        for i in range(3):
            piv = max(range(i, 3), key=lambda k: abs(M[k][i]))
            M[i], M[piv] = M[piv], M[i]
            if abs(M[i][i]) < 1e-9:
                ok = False
                break
            for k in range(i + 1, 3):
                f = M[k][i] / M[i][i]
                for j in range(i, 4):
                    M[k][j] -= f * M[i][j]
        if not ok:
            out[d] = None
            continue
        coef = [0.0] * 3
        for i in (2, 1, 0):
            coef[i] = (M[i][3] - sum(M[i][j] * coef[j]
                                     for j in range(i + 1, 3))) / M[i][i]
        mu = coef[0] + coef[1] * (xt_c - mx) + coef[2] * (mom_target - mm)
        resid = [(y - (coef[0] + coef[1] * (x - mx) + coef[2] * (m - mm)), w)
                 for x, m, y, w in pts]
        qs = wquantiles(resid, LEVELS)
        if qs is None:
            out[d] = None
            continue
        out[d] = sorted(p1 * math.exp(mu + e) for e in qs)
    return out


def ramp_w(d):
    """Regression weight by day-hour: median-dominated early, regression
    late, where the scale/momentum effects are strongest."""
    if d <= 10:
        return 0.5
    if d >= 20:
        return 0.9
    return 0.5 + 0.4 * (d - 10) / 10.0


def blend(reg, med):
    """Log-space blend of the two predictors, per day-hour."""
    out = {}
    for d in set(reg) | set(med):
        a, b = reg.get(d), med.get(d)
        if a is None and b is None:
            out[d] = None
        elif b is None:
            out[d] = a
        elif a is None:
            out[d] = b
        else:
            w = ramp_w(d)
            out[d] = [math.exp(w * math.log(x) + (1.0 - w) * math.log(y))
                      for x, y in zip(a, b)]
    return out


def baseline_row(p1, case_hour):
    """House persistence baseline (starter formula) as last-resort fallback."""
    return [p1 * math.exp(z * 0.15 * math.sqrt(case_hour - 1)) for z in Z]


def forecast_one(case):
    known = [k for k in case.get("known") or [] if k.get("hour") == 1]
    if not known:
        return None
    p1 = max(1, known[0].get("viewers") or 1)
    train = []
    for c in case.get("comparables") or []:
        launch = Launch(c)
        if launch.h0 is None:
            continue
        train.append(launch)
    mom = target_momentum(case)
    if mom is None:
        # no usable pre-release history: fall back to the training launches'
        # similarity-weighted mean momentum (clipped, as in the fit)
        num = den = 0.0
        for launch in train:
            if launch.mom is None:
                continue
            w = weight(launch, p1)
            num += w * clip_mom(launch.mom)
            den += w
        mom = (num / den) if den > 0 else 0.0
    else:
        mom = clip_mom(mom)
    reg = predict_regress(train, p1, p1, mom)
    med = predict_median(train, p1, p1)
    preds = blend(reg, med)
    rows = []
    for h in range(2, 25):
        q = preds.get(h - 1)
        if q is None:
            q = baseline_row(p1, h)
        q = sorted(min(1e9, max(1e-9, float(x))) for x in q)
        rows.append({"hour": h, "q": [round(x, 2) for x in q]})
    return {"hours": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result = forecast_one(case)
        if result is None:
            continue
        (out_dir / (path.stem + ".out")).write_text(
            json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
