#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01 entry: a probabilistic forecast of
Roblox's Up-and-Coming top 10 over the 72 hours after the deadline.

One static forecast is scored at every hourly read of the window, so each
place's distribution approximates the window-averaged occupancy:
P(game g holds place k at a uniform random read of the window).

Everything is estimated from the history in /input at run time:

- Rank-state -> destination-place transition frequencies over all pairs of
  history reads, bucketed by lag (0-12 h, 12-36 h, 36-72 h) and mixed with
  weights equal to each bucket's share of the scored window.
- The transitions are conditioned hierarchically on the game's state at the
  source read: its rank band, its player-count trend class (playing now vs
  the mean of its previous appearances: rising / flat / falling / new), and
  a size flag (playing at least twice the chart median). Sparse cells shrink
  toward the rank-band distribution.
- Blended 70/30 with a persistence kernel around the closing chart (the
  house baseline's kernel), because the window's first hours sit close to
  the close.
- A diffuse background over games seen in the last few reads but absent at
  the close, sized by the empirical off-chart entrant rate per place.
- Details-driven adjustments, active only when the history carries game
  details (the house recorder's reads do; the public archive's do not, and
  there these are exactly neutral):
  * 28-day eligibility: a game leaves the chart 28 days after publication,
    so a game whose eligibility ends inside the window is scaled by the
    fraction of the window it remains eligible for.
  * Vote ratio: a bounded tilt (x0.90..x1.10) by up-vote share.
  * Visits momentum: a bounded tilt (x0.95..x1.05) by visits growth.
- Each place keeps at most 0.92 of its mass named; the rest is left to the
  scoring rule's unnamed share, which covers games nobody could name -
  above all, games first published during the window itself.

Deterministic: fixed constants, no randomness, no clock, no network.
Reads $MAKEFUN_INPUT_DIR (else /input), writes the forecast to
$MAKEFUN_OUTPUT_DIR (else /output) under the file name task.json gives.
"""

import json
import math
import os
from datetime import datetime, timezone

# --- frozen modelling constants (tuned on the practice window, round numbers)
LAG_EDGES = [0.0, 12.0, 36.0, 72.5]   # lag buckets, hours
ALPHA = 3.0                           # smoothing of state dists toward pooled
BETA_TREND = 20.0                     # shrinkage of (state,trend,size) to state
W_PERSIST = 0.30                      # persistence-kernel blend weight
CAP_NAMED = 0.92                      # max named probability mass per place
MIN_P = 0.0004                        # drop named rows below this
BG_RECENT_READS = 6                   # reads that define "recently seen"
RISE, FALL = 1.25, 0.80               # playing-trend class thresholds
DAY = 86400.0


def parse_t(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def state_of(rank):
    """Rank band used for transition states."""
    if rank is None:
        return "ABS"
    if rank <= 5:
        return rank
    if rank <= 10:
        return "S6"
    if rank <= 20:
        return "S11"
    if rank <= 40:
        return "S21"
    return "S41"


def bucket_of(lag_h):
    for b in range(len(LAG_EDGES) - 1):
        if LAG_EDGES[b] <= lag_h < LAG_EDGES[b + 1]:
            return b
    return None


def bucket_weights(horizon_h):
    ws = []
    for b in range(len(LAG_EDGES) - 1):
        lo = LAG_EDGES[b]
        hi = min(LAG_EDGES[b + 1], horizon_h)
        ws.append(max(0.0, hi - lo) / horizon_h)
    total = sum(ws)
    return [w / total for w in ws]


def load_input(input_dir):
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            # Robustness: a read the recorder marked as failed (ok
            # explicitly false), or one whose chart list is missing or
            # empty, carries no usable signal - skip it rather than let
            # a partial/corrupt snapshot distort the statistics.
            chart = rec.get("up_and_coming")
            if rec.get("ok") is False or not chart or not rec.get("at"):
                continue
            reads.append(rec)
    reads.sort(key=lambda r: r["at"])
    return task, reads


def details_map(details):
    """Normalise a read's details (mapping or list) to {universe_id: dict}."""
    out = {}
    if isinstance(details, dict):
        for kk, vv in details.items():
            try:
                out[int(kk)] = vv
            except (TypeError, ValueError):
                continue
    elif isinstance(details, list):
        for vv in details:
            if isinstance(vv, dict):
                uid = vv.get("id") or vv.get("universe_id") or vv.get("universeId")
                if uid is not None:
                    out[int(uid)] = vv
    return out


def build_forecast(reads, horizon_h, window_from):
    times = [parse_t(r["at"]) for r in reads]
    charts = []     # per read: {universe_id: rank}
    playing = []    # per read: {universe_id: playing}
    for r in reads:
        charts.append({row["universe_id"]: row["rank"] for row in r["up_and_coming"]})
        playing.append({row["universe_id"]: (row.get("playing") or 0)
                        for row in r["up_and_coming"]})
    n = len(reads)
    bw = bucket_weights(horizon_h)
    nb = len(bw)

    def trend_class(g, i):
        """Playing at read i vs the mean of g's up-to-3 previous appearances."""
        if i == 0:
            return "new"
        prior = []
        for j in range(i - 1, -1, -1):
            if g in playing[j]:
                prior.append(playing[j][g])
                if len(prior) >= 3:
                    break
        if not prior:
            return "new"
        ratio = playing[i][g] / max(1.0, sum(prior) / len(prior))
        if ratio >= RISE:
            return "rise"
        if ratio <= FALL:
            return "fall"
        return "flat"

    # "big" flag per read: playing at least twice the chart's median.
    big = []
    for i in range(n):
        vals = sorted(playing[i].values())
        med = vals[len(vals) // 2] if vals else 0
        big.append({g for g, p in playing[i].items() if med and p >= 2 * med})

    # Transition counts: C[b][state][dest], C2[b][(state,trend,big)][dest];
    # dest is a place 1..10, or 0 for "not in the top 10 at the later read".
    C = [dict() for _ in range(nb)]
    C2 = [dict() for _ in range(nb)]
    ent = [dict() for _ in range(nb)]       # entrant counts per place
    ent_pairs = [0] * nb
    for i in range(n):
        ci = charts[i]
        cls_i = {g: trend_class(g, i) for g in ci}
        for j in range(i + 1, n):
            lag = (times[j] - times[i]) / 3600.0
            if lag <= 0:
                continue
            b = bucket_of(lag)
            if b is None:
                continue
            cj = charts[j]
            for g, r_i in ci.items():
                s = state_of(r_i)
                d = cj.get(g)
                d = d if (d is not None and d <= 10) else 0
                C[b].setdefault(s, [0] * 11)[d] += 1
                C2[b].setdefault((s, cls_i[g], g in big[i]), [0] * 11)[d] += 1
            ent_pairs[b] += 1
            for g, r_j in cj.items():
                if r_j <= 10 and g not in ci:
                    ent[b][r_j] = ent[b].get(r_j, 0) + 1

    pooled = {}
    for b in range(nb):
        for s, counts in C[b].items():
            acc = pooled.setdefault(s, [0] * 11)
            for d in range(11):
                acc[d] += counts[d]

    def state_dist(s):
        pool = pooled.get(s)
        if pool is None:
            return [1.0] + [0.0] * 10
        tot_pool = sum(pool)
        out = [0.0] * 11
        for b in range(nb):
            counts = C[b].get(s)
            if counts is None:
                dist = [c / tot_pool for c in pool]
            else:
                tot = sum(counts)
                dist = [(counts[d] + ALPHA * pool[d] / tot_pool) / (tot + ALPHA)
                        for d in range(11)]
            for d in range(11):
                out[d] += bw[b] * dist[d]
        return out

    def trans_dist(s, cls, big_flag):
        out = [0.0] * 11
        pool = pooled.get(s)
        key = (s, cls, big_flag)
        for b in range(nb):
            counts = C2[b].get(key)
            if counts is None or pool is None:
                dist = state_dist(s)
            else:
                tot = sum(counts)
                tot_pool = sum(pool)
                sc = C[b].get(s)
                if sc is None:
                    prior = [c / tot_pool for c in pool]
                else:
                    st = sum(sc)
                    prior = [(sc[d] + ALPHA * pool[d] / tot_pool) / (st + ALPHA)
                             for d in range(11)]
                dist = [(counts[d] + BETA_TREND * prior[d]) / (tot + BETA_TREND)
                        for d in range(11)]
            for d in range(11):
                out[d] += bw[b] * dist[d]
        return out

    close = charts[-1]
    close_sorted = sorted(close.items(), key=lambda kv: kv[1])
    close_ids = [g for g, _ in close_sorted]
    close_cls = {g: trend_class(g, n - 1) for g in close}
    close_big = big[-1] if big else set()

    # Per-game place probabilities from the conditioned transitions.
    P = {}
    for g, r in close.items():
        dist = trans_dist(state_of(r), close_cls[g], g in close_big)
        for k in range(1, 11):
            if dist[k] > 0:
                P[(g, k)] = dist[k]

    # Persistence blend (the baseline kernel on the closing chart).
    for key in list(P):
        P[key] *= (1 - W_PERSIST)

    def persist_add(g, k, p):
        P[(g, k)] = P.get((g, k), 0.0) + W_PERSIST * p

    if len(close_ids) >= 2:
        persist_add(close_ids[0], 1, 0.5)
        persist_add(close_ids[1], 1, 0.3)
    for k in range(2, 11):
        if k - 1 < len(close_ids):
            persist_add(close_ids[k - 1], k, 0.5)
        if k - 2 >= 0:
            persist_add(close_ids[k - 2], k, 0.15)
        if k < len(close_ids):
            persist_add(close_ids[k], k, 0.15)

    # Background: games seen in recent reads but absent at the close share
    # the empirical off-chart entrant rate of each place.
    recent = set()
    for c in charts[-BG_RECENT_READS:]:
        recent.update(c)
    bg_games = sorted(recent - set(close))
    if bg_games:
        for k in range(1, 11):
            share = 0.0
            for b in range(nb):
                if ent_pairs[b]:
                    share += bw[b] * (ent[b].get(k, 0) / ent_pairs[b])
            each = share / len(bg_games)
            if each > 0:
                for g in bg_games:
                    P[(g, k)] = P.get((g, k), 0.0) + each

    apply_details_adjustments(reads, P, close, window_from, horizon_h)

    # Assemble rows: cap each place's named mass, drop dust, stable order.
    rows = []
    for k in range(1, 11):
        cand = [(g, P[(g, k)]) for (g, kk) in P if kk == k and P[(g, k)] >= MIN_P]
        cand.sort(key=lambda x: (-x[1], x[0]))
        cand = cand[:100]
        s = sum(p for _, p in cand)
        if s > CAP_NAMED:
            cand = [(g, p * CAP_NAMED / s) for g, p in cand]
        rows.extend({"universe_id": int(g), "place": k, "p": round(p, 6)}
                    for g, p in cand)

    # Canonical output order: place ascending, then probability
    # descending, then universe_id ascending. (Row order does not affect
    # scoring; this just makes the emitted file canonical regardless of
    # how the rows above were assembled.)
    rows.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))

    # top10: greedy by place over the modelled probabilities, filled from
    # the closing chart and then the most recent earlier charts.
    occ = {}
    for (g, k), p in P.items():
        occ[g] = occ.get(g, 0.0) + p
    remaining = sorted(occ, key=lambda g: (-occ[g], g))
    by_place = {}
    for (g, k), p in P.items():
        by_place.setdefault(k, []).append((p, g))
    top10, used = [], set()
    for k in range(1, 11):
        cands = sorted(by_place.get(k, []), key=lambda x: (-x[0], x[1]))
        pick = next((g for _, g in cands if g not in used), None)
        if pick is None:
            pick = next((g for g in remaining if g not in used), None)
        if pick is not None:
            top10.append(int(pick))
            used.add(pick)
    for chart in reversed(charts):
        if len(top10) >= 10:
            break
        for g, _r in sorted(chart.items(), key=lambda kv: kv[1]):
            if len(top10) >= 10:
                break
            if g not in used:
                top10.append(int(g))
                used.add(g)
    return {"probabilities": rows, "top10": top10[:10]}


def apply_details_adjustments(reads, P, close, window_from, horizon_h):
    """Adjustments that need game details / votes. Exactly neutral when the
    history carries none (the public archive's reads carry neither)."""
    last = reads[-1]
    det_map = details_map(last.get("details"))
    win_end = window_from + horizon_h * 3600.0

    # Visits momentum needs an earlier detail-bearing read (~a day back).
    ref_map = {}
    last_t = parse_t(last["at"])
    for r in reversed(reads[:-1]):
        m = details_map(r.get("details"))
        if m:
            ref_map = m
            if last_t - parse_t(r["at"]) >= 20 * 3600:
                break

    votes = {row["universe_id"]: (row.get("up_votes"), row.get("down_votes"))
             for row in last["up_and_coming"]}
    for g in list(close):
        det = det_map.get(g)
        if det and det.get("created"):
            try:
                created = parse_t(det["created"])
            except (ValueError, TypeError):
                created = None
            if created is not None:
                elig_end = created + 28 * DAY
                if elig_end <= window_from:
                    frac = 0.0
                elif elig_end >= win_end:
                    frac = 1.0
                else:
                    frac = (elig_end - window_from) / (win_end - window_from)
                if frac < 1.0:
                    for k in range(1, 11):
                        if (g, k) in P:
                            P[(g, k)] *= frac
        up, down = votes.get(g, (None, None))
        if up is not None and down is not None and (up + down) >= 20:
            q = up / (up + down)
            mult = 0.9 + 0.2 * q              # bounded tilt, [0.90, 1.10]
            for k in range(1, 11):
                if (g, k) in P:
                    P[(g, k)] *= mult
        if det and ref_map.get(g):
            v_new = det.get("visits")
            v_old = ref_map[g].get("visits")
            if v_new and v_old:
                growth = math.log(v_new / v_old) / math.log(2.0)
                mult = 1.0 + 0.05 * max(-1.0, min(1.0, growth))   # [0.95, 1.05]
                for k in range(1, 11):
                    if (g, k) in P:
                        P[(g, k)] *= mult


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    task, reads = load_input(input_dir)
    window = task.get("window") or {}
    if window.get("from") and window.get("until"):
        window_from = parse_t(window["from"])
        horizon_h = (parse_t(window["until"]) - window_from) / 3600.0
    else:
        window_from = parse_t(reads[-1]["at"])
        horizon_h = 72.0
    forecast = build_forecast(reads, horizon_h, window_from)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, task.get("output", "forecast.json"))
    with open(out_path, "w") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
