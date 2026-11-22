"""Rate-distortion optimization contracts for a block-based video encoder.

Units and conventions used throughout the module
------------------------------------------------
* ``bits`` always means a non-negative integer number of encoded bits (never
  bytes and never a rate normalised per pixel).
* ``distortion`` is a non-negative sum of squared differences (SSD), in the
  same sample-domain units as the input pixels.
* ``lam`` is the non-negative Lagrangian multiplier in units of
  distortion per bit.
* Floating-point comparisons in the tests use a tolerance of ``1e-9``.
"""

import math

VALID_MODES = ("skip", "intra", "inter")


def _is_finite_real(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_non_negative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def ssd(a, b):
    """Return the sum of squared differences between two equal-length signals."""
    if len(a) != len(b):
        raise ValueError("signals must have equal length")
    return float(sum((x - y) ** 2 for x, y in zip(a, b)))


def lambda_for_qp(qp):
    """Map an integer quantiser step to the Lagrangian multiplier.

    ``lam = 0.85 * 2 ** ((qp - 12) / 3)`` for integer ``qp`` in ``[0, 51]``.
    """
    if isinstance(qp, bool) or not isinstance(qp, int) or not 0 <= qp <= 51:
        raise ValueError("qp must be an integer in [0, 51]")
    return 0.85 * 2.0 ** ((qp - 12) / 3.0)


def rd_cost(distortion, bits, lam):
    """Return the rate-distortion cost ``J = D + lam * R``."""
    if not _is_finite_real(distortion) or distortion < 0:
        raise ValueError("distortion must be a finite non-negative real")
    if not _is_non_negative_int(bits):
        raise ValueError("bits must be a non-negative integer")
    if not _is_finite_real(lam) or lam <= 0:
        raise ValueError("lam must be a finite positive real")
    return float(distortion) + float(lam) * bits


def _check_candidate(candidate):
    if not hasattr(candidate, "__getitem__"):
        raise ValueError("candidate must be a mapping")
    mode = candidate["mode"]
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode: {mode!r}")
    distortion = candidate["distortion"]
    if not _is_finite_real(distortion) or distortion < 0:
        raise ValueError("distortion must be a finite non-negative real")
    bits = candidate["bits"]
    if not _is_non_negative_int(bits):
        raise ValueError("bits must be a non-negative integer")
    return mode, float(distortion), bits


def choose_mode(candidates, lam):
    """Choose the candidate mode with the smallest rate-distortion cost.

    Ties are resolved by the lexicographically smallest mode name,
    independently of the input order.
    """
    if not _is_finite_real(lam) or lam <= 0:
        raise ValueError("lam must be a finite positive real")
    candidates = list(candidates)
    if not candidates:
        raise ValueError("at least one candidate is required")
    checked = [_check_candidate(candidate) for candidate in candidates]
    modes = [mode for mode, _, _ in checked]
    if len(set(modes)) != len(modes):
        raise ValueError("duplicate modes are not allowed")
    best_mode, best_cost = min(
        ((mode, distortion + lam * bits) for mode, distortion, bits in checked),
        key=lambda item: (item[1], item[0]),
    )
    return {"mode": best_mode, "cost": float(best_cost)}


def mode_bits(mode, block_size):
    """Estimate the number of bits needed to code one block in a given mode."""
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode: {mode!r}")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    if mode == "skip":
        return 1
    n = math.ceil(math.log2(block_size)) if block_size > 1 else 0
    if mode == "intra":
        return 4 + 2 * n
    return 2 + 2 * n


def refine_lambda(cost_curve):
    """Estimate lambda from the slope between two rate-distortion attempts.

    ``lam = (D0 - D1) / (R1 - R0)`` for strictly increasing rates and
    strictly decreasing distortion.
    """
    points = list(cost_curve)
    if len(points) != 2:
        raise ValueError("cost_curve must contain exactly two observations")
    r0, d0 = points[0]["bits"], points[0]["distortion"]
    r1, d1 = points[1]["bits"], points[1]["distortion"]
    for value in (r0, d0, r1, d1):
        if not _is_finite_real(value):
            raise ValueError("bits and distortion must be finite reals")
    if not r1 > r0:
        raise ValueError("rates must be strictly increasing")
    if not d1 < d0:
        raise ValueError("distortion must strictly decrease")
    return float((d0 - d1) / (r1 - r0))


def validate_candidates(cands):
    """Return a list of human-readable validation errors for RD candidates."""
    errors = []
    for index, candidate in enumerate(cands):
        try:
            _check_candidate(candidate)
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"candidate {index}: {exc}")
    checked = []
    for index, candidate in enumerate(cands):
        try:
            checked.append(_check_candidate(candidate))
        except (ValueError, KeyError, TypeError):
            continue
    for i in range(len(checked)):
        for j in range(len(checked)):
            _, d_i, b_i = checked[i]
            _, d_j, b_j = checked[j]
            if b_i < b_j and d_i + 1e-9 < d_j:
                errors.append(
                    f"candidates {i} and {j}: more bits ({b_j} > {b_i}) "
                    f"but more distortion ({d_j} > {d_i})"
                )
    return errors


def rd_batch(blocks, config):
    """Optimise a batch of independent blocks, each with its own lambda."""
    config = dict(config)
    allowed_keys = {"lambda_scale"}
    unknown = set(config) - allowed_keys
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    lambda_scale = config.get("lambda_scale", 1.0)
    if not _is_finite_real(lambda_scale) or lambda_scale <= 0:
        raise ValueError("lambda_scale must be a finite positive real")

    results = []
    total_cost = 0.0
    for block in blocks:
        lam = lambda_for_qp(block["qp"]) * lambda_scale
        chosen = choose_mode(block["candidates"], lam)
        results.append(
            {
                "block_id": block["block_id"],
                "mode": chosen["mode"],
                "cost": chosen["cost"],
                "lambda": lam,
            }
        )
        total_cost += chosen["cost"]
    return {"blocks": results, "total_cost": float(total_cost)}
