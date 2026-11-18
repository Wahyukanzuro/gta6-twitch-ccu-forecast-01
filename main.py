"""Launch-day Twitch-viewer forecaster (v2).

Reads one case file (schema makefun/twitch-ccu-case/v1) from the input dir
and writes one forecast file per case to the output dir: 23 rows (hours 2-24),
five quantiles each (levels 0.05/0.25/0.5/0.75/0.95).

Model (fitted at runtime from the case's own comparables only; nothing is
hardcoded, no table is baked in):
- Each comparable launch contributes a ratio curve r_d = viewers(day hour d) /
  viewers(day hour 0). A comparable with no day-hour-0 value (e.g. a
  pre-release history that is not a launch day) is skipped automatically.
- Similarity weight per comparable: Gaussian kernel on log hour-0 viewers
  (bandwidth 1.5) x 1.5 for rolling-midnight unlocks x 1.5 for console-only
  launches x 0.3 for early-access-contaminated launches.
- Two estimators per day-hour, blended 75/25 in log space:
  (a) a local weighted least-squares fit of log ratio on log hour-0 viewers,
      evaluated at the target's log hour-1 print (the target's log print is
      clamped to the fitted launches' range, so a record print cannot drive
      the line off a cliff), with spreads from the weighted quantiles of the
      fit residuals;
  (b) the weighted median ratio and weighted empirical quantiles.
  Either side falls back to the other when it has too few points.
- If a day-hour has no usable comparable at all, the house persistence
  baseline is used for that hour.
- The live streams / categories fields of a case are intentionally NOT used:
  they are null in every practice case, so no adjustment based on them could
  be validated; the forecast anchors on the hour-1 print and the comparables.

Standard library only. Deterministic: fixed iteration orders, no randomness.
"""
import json
import math
import os
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
BW = 1.5
BLEND_W = 0.75
MIN_REGRESS = 8
MIN_MEDIAN = 3


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_rolling(unlock):
    u = (unlock or "").lower()
    return ("rolling" in u) or ("midnight local" in u)


def is_early(unlock):
    return "early-access-contaminated" in (unlock or "").lower()


def is_console_only(platforms):
    return "ps5 only" in (platforms or "").lower()


def weight(launch, target_h0):
    h0 = launch["h0"]
    if h0 is None or h0 <= 0:
        return 0.0
    d = abs(math.log(h0) - math.log(max(target_h0, 1)))
    w = math.exp(-0.5 * (d / BW) ** 2)
    if is_rolling(launch["unlock"]):
        w *= 1.5
    if is_console_only(launch["platforms"]):
        w *= 1.5
    if is_early(launch["unlock"]):
        w *= 0.3
    return w


def wquantiles(pairs, levels):
    """Weighted empirical quantiles (type-7 style on the weighted CDF)."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0 or not pairs:
        return None
    cum = []
    s = 0.0
    for v, w in pairs:
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
                t = 0.0 if c1 == c0 else (lv - c0) / (c1 - c0)
                res.append(v0 + t * (v1 - v0))
                break
    return res


def load_launches(case):
    """Comparables as {h0, hourly {day-hour: viewers}, unlock, platforms}.
    Only day hours 0..23 count; a comparable without a day-hour-0 value is
    skipped (this also drops pre-release histories, which are not launches)."""
    out = []
    for c in case.get("comparables") or []:
        ds = parse_ts(c["day_start"])
        hourly = {}
        for h in c.get("hourly") or []:
            t = parse_ts(h["from"])
            dh = (t - ds).total_seconds() / 3600.0
            if 0 <= dh < 24:
                v = h.get("viewers")
                if v is not None:
                    hourly[int(dh)] = v
        h0 = hourly.get(0)
        if h0 is None or h0 <= 0:
            continue
        out.append({"h0": h0, "hourly": hourly,
                    "unlock": c.get("unlock") or "",
                    "platforms": c.get("platforms") or ""})
    return out


def predict_day(launches, p1):
    """{day-hour: [q05..q95] ratios} for day hours 0..23, or None per hour."""
    xt = math.log(max(p1, 1))
    res = {}
    for d in range(0, 24):
        pts = []
        for launch in launches:
            if d not in launch["hourly"]:
                continue
            v = launch["hourly"][d]
            if v is None or v <= 0:
                continue
            w = weight(launch, p1)
            if w <= 0:
                continue
            pts.append((math.log(launch["h0"]), math.log(v / launch["h0"]), w))
        reg = None
        if len(pts) >= MIN_REGRESS:
            xs = [x for x, _, _ in pts]
            xc = min(max(xt, min(xs)), max(xs))
            sw = sum(w for _, _, w in pts)
            mx = sum(w * x for x, _, w in pts) / sw
            my = sum(w * y for _, y, w in pts) / sw
            sxx = sum(w * (x - mx) ** 2 for x, _, w in pts)
            b = 0.0
            if sxx > 1e-12:
                sxy = sum(w * (x - mx) * (y - my) for x, y, w in pts)
                b = sxy / sxx
            a = my - b * mx
            mu = a + b * xc
            resid = [(y - (a + b * x), w) for x, y, w in pts]
            qs = wquantiles(resid, LEVELS)
            if qs is not None:
                reg = sorted(math.exp(mu + e) for e in qs)
        pairs = []
        for launch in launches:
            if d not in launch["hourly"]:
                continue
            v = launch["hourly"][d]
            if v is None or v <= 0:
                continue
            w = weight(launch, p1)
            if w <= 0:
                continue
            pairs.append((v / launch["h0"], w))
        med = None
        if len(pairs) >= MIN_MEDIAN:
            qs = wquantiles(pairs, LEVELS)
            if qs is not None:
                med = sorted(qs)
        if reg is None and med is None:
            res[d] = None
        elif med is None:
            res[d] = reg
        elif reg is None:
            res[d] = med
        else:
            res[d] = sorted(math.exp(BLEND_W * math.log(r) + (1.0 - BLEND_W) * math.log(m))
                            for r, m in zip(reg, med))
    return res


def forecast(case):
    p1 = 1
    for k in case.get("known") or []:
        if k.get("hour") == 1 and k.get("viewers"):
            p1 = max(1, k["viewers"])
    launches = load_launches(case)
    day = predict_day(launches, p1)
    rows = []
    for h in range(2, 25):
        ratios = day.get(h - 1)
        if ratios is None:
            qs = [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]
        else:
            qs = [p1 * r for r in ratios]
        qs = [round(max(min(q, 1e9), 0.01), 2) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
