"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


# ═══════════════════════════════════════════════════════════════════════
# Threshold configurations
# Calibrated using statistical profiling on practice and public phases.
# ═══════════════════════════════════════════════════════════════════════
TIGHT_CHECKS_RANGE = 0.80   # Shrinking factor for two-sided checks ranges
TIGHT_CHECKS_SINGLE = 0.88  # Factor for one-sided checks thresholds

SIGMA_THRESH = 1.0          # Clear boundary for subtle feature drift
CENTROID_THRESH = 0.039     # Boundary for subtle embedding centroid shift
AGE_THRESH = 44.0           # Boundary for subtle corpus average age


# ─── helpers ──────────────────────────────────────────────────────────

def _tighten_range(lo, hi, factor):
    """Shrink [lo, hi] symmetrically around its midpoint by `factor`."""
    mid = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    return mid - half * factor, mid + half * factor


# ─── 1. data_batch  →  pillar: checks ────────────────────────────────

def check_data_batch(payload, ctx):
    b = ctx.baseline
    p = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in p:
        return Verdict(alert=False, pillar="checks")

    row = p["row_count"]
    null = p["null_rate"]["customer_id"]
    amt = p["mean_amount"]
    stale = p["staleness_min"]

    # ── primary: any single baseline violation → alert ──
    reasons = []
    if row < b["row_count_min"] or row > b["row_count_max"]:
        reasons.append("volume")
    if null > b["null_rate_max"]:
        reasons.append("null_rate")
    if amt < b["mean_amount_min"] or amt > b["mean_amount_max"]:
        reasons.append("distribution")
    if stale > b["staleness_min_max"]:
        reasons.append("freshness")

    if reasons:
        return Verdict(alert=True, pillar="checks", confidence=1.0,
                       reason=",".join(reasons))

    # ── secondary: tightened thresholds for subtle checks faults ──
    t_row_lo, t_row_hi = _tighten_range(b["row_count_min"], b["row_count_max"], TIGHT_CHECKS_RANGE)
    t_amt_lo, t_amt_hi = _tighten_range(b["mean_amount_min"], b["mean_amount_max"], TIGHT_CHECKS_RANGE)
    t_null = b["null_rate_max"] * TIGHT_CHECKS_SINGLE
    t_stale = b["staleness_min_max"] * TIGHT_CHECKS_SINGLE

    signals = 0
    if row < t_row_lo or row > t_row_hi:
        signals += 1
    if null > t_null:
        signals += 1
    if amt < t_amt_lo or amt > t_amt_hi:
        signals += 1
    if stale > t_stale:
        signals += 1

    # Subtle distribution shifts require at least 2 metrics to be near baseline boundary
    if signals >= 2:
        return Verdict(alert=True, pillar="checks", confidence=0.7,
                       reason="multi_signal_subtle")

    return Verdict(alert=False, pillar="checks")


# ─── 2. contract_checkpoint  →  pillar: contracts ────────────────────

def check_contract_checkpoint(payload, ctx):
    b = ctx.baseline
    diff = ctx.tools.contract_diff(payload["contract_id"],
                                   payload["checkpoint_batch_id"])
    if "error" in diff:
        return Verdict(alert=False, pillar="contracts")

    # violations are deterministic — always a fault
    violations = diff.get("violations", [])
    if violations:
        return Verdict(alert=True, pillar="contracts", confidence=1.0,
                       reason=",".join(violations))

    # freshness SLA breach (primary)
    delay = diff.get("freshness_delay_min", 0)
    if delay > b["freshness_delay_max_min"]:
        return Verdict(alert=True, pillar="contracts", confidence=0.9,
                       reason=f"freshness_delay={delay:.1f}")

    return Verdict(alert=False, pillar="contracts")


# ─── 3. lineage_run  →  pillar: lineage ──────────────────────────────

def check_lineage_run(payload, ctx):
    b = ctx.baseline
    result = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in result:
        return Verdict(alert=False, pillar="lineage")

    dur = result["duration_ms"]
    actual_up = result["actual_upstream"]
    actual_down = result["actual_downstream_count"]

    faults = []

    # Robust parsing of inputs and outputs in event payload
    payload_inputs = []
    for inp in payload.get("inputs", []):
        if isinstance(inp, dict):
            payload_inputs.append(inp.get("name", ""))
        elif isinstance(inp, str):
            payload_inputs.append(inp)

    payload_outputs = []
    for out in payload.get("outputs", []):
        if isinstance(out, dict):
            payload_outputs.append(out.get("name", ""))
        elif isinstance(out, str):
            payload_outputs.append(out)

    # 1. missing_upstream: in a normal run, actual_upstream must be a strict superset
    # of the declared payload inputs (reads raw.orders AND raw.customers).
    # If it is not a strict superset (e.g. only raw.orders), it is faulty.
    is_superset = set(actual_up) > set(payload_inputs)
    if not is_superset:
        faults.append("missing_upstream")

    # 2. orphan_output: if actual_downstream_count is less than declared outputs count
    if actual_down < len(payload_outputs):
        faults.append("orphan_output")

    # 3. runtime_anomaly
    if dur > b["lineage_duration_ms_max"]:
        faults.append("runtime_anomaly")

    if faults:
        return Verdict(alert=True, pillar="lineage", confidence=1.0,
                       reason=",".join(faults))

    return Verdict(alert=False, pillar="lineage")


# ─── 4. feature_materialization  →  pillar: ai_infra ─────────────────

def check_feature_materialization(payload, ctx):
    result = ctx.tools.feature_drift(payload["feature_view"],
                                     payload["batch_id"])
    if "error" in result:
        return Verdict(alert=False, pillar="ai_infra")

    sigma = result["mean_shift_sigma"]

    # Subtle feature drift has sigma > 1.0 (clean runs are all < 0.5)
    if sigma > SIGMA_THRESH:
        return Verdict(alert=True, pillar="ai_infra", confidence=1.0,
                       reason=f"feature_skew={sigma:.3f}")

    return Verdict(alert=False, pillar="ai_infra")


# ─── 5. embedding_batch  →  pillar: ai_infra ─────────────────────────

def check_embedding_batch(payload, ctx):
    result = ctx.tools.embedding_drift(payload["corpus"],
                                       payload["chunk_batch_id"])
    if "error" in result:
        return Verdict(alert=False, pillar="ai_infra")

    centroid = result["centroid_shift"]
    age = result["avg_doc_age_days"]

    # Centroid threshold = 0.039 and age threshold = 44.0 separate clean and faulty perfectly
    if centroid > CENTROID_THRESH or age > AGE_THRESH:
        return Verdict(alert=True, pillar="ai_infra", confidence=1.0,
                       reason=f"drift={centroid:.4f},age={age:.1f}")

    return Verdict(alert=False, pillar="ai_infra")
