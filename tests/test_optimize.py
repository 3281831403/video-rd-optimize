"""Behavioural tests for the rate-distortion optimization stubs.

All tests are black-box checks of the contracts in ``rd/optimize.py``.  They
must remain unmodified: the initial repository intentionally fails every test
because the implementation is stubbed out.
"""

import math

import pytest

from rd.optimize import (
    choose_mode,
    lambda_for_qp,
    mode_bits,
    rd_batch,
    rd_cost,
    refine_lambda,
    ssd,
    validate_candidates,
)

TOL = 1e-9


def _j(candidate, lam):
    """Reference rate-distortion cost used by exhaustive comparisons."""
    return candidate["distortion"] + lam * candidate["bits"]


def test_ssd_is_sum_of_squared_errors():
    a = [1.0, 2.0, 4.0]
    b = [1.0, 4.0, 0.0]

    # Squared differences are 0, 4, 16; SAD would incorrectly give 6.
    assert ssd(a, b) == pytest.approx(20.0, rel=TOL, abs=TOL)
    assert ssd([], []) == pytest.approx(0.0, rel=TOL, abs=TOL)


def test_lambda_for_qp_matches_exponential_contract():
    assert lambda_for_qp(12) == pytest.approx(0.85, rel=TOL, abs=TOL)
    assert lambda_for_qp(18) == pytest.approx(3.4, rel=TOL, abs=TOL)


def test_rd_cost_adds_lambda_times_rate():
    # J = 4 + 2 * 10 = 24, not 4 + 10 / 2 = 9.
    assert rd_cost(4.0, 10, 2.0) == pytest.approx(24.0, rel=TOL, abs=TOL)
    assert rd_cost(0.0, 0, 2.0) == pytest.approx(0.0, rel=TOL, abs=TOL)


def test_choose_mode_minimizes_rd_cost_not_bits():
    candidates = [
        {"mode": "skip", "distortion": 20.0, "bits": 2},   # J = 24
        {"mode": "inter", "distortion": 4.0, "bits": 4},   # J = 12
        {"mode": "intra", "distortion": 1.0, "bits": 10},  # J = 21
    ]

    result = choose_mode(candidates, 2.0)

    assert set(result) == {"mode", "cost"}
    assert result["mode"] == "inter"
    assert result["cost"] == pytest.approx(12.0, rel=TOL, abs=TOL)


def test_skip_mode_includes_flag_bits():
    assert isinstance(mode_bits("skip", 16), int)
    assert mode_bits("skip", 16) == 1
    assert mode_bits("skip", 64) == 1


def test_intra_mode_bits_grow_sublinearly():
    sizes = (4, 16, 64)
    observed = [mode_bits("intra", size) for size in sizes]
    expected = [4 + 2 * math.ceil(math.log2(size)) for size in sizes]

    assert all(isinstance(value, int) for value in observed)
    assert observed == expected
    assert mode_bits("intra", 64) - mode_bits("intra", 16) <= 4


def test_refine_lambda_recovers_positive_slope():
    curve = [
        {"bits": 2, "distortion": 10.0},
        {"bits": 8, "distortion": 2.0},
    ]

    # Positive RD slope magnitude: (10 - 2) / (8 - 2) = 4/3.
    # The sign-corrected reciprocal would be 6/8 = 0.75, so this also catches
    # an implementation that inverts the slope.
    estimate = refine_lambda(curve)

    assert estimate == pytest.approx(4.0 / 3.0, rel=TOL, abs=TOL)
    assert estimate > 0.0


def test_choose_mode_breaks_ties_by_mode_name():
    candidates = [
        {"mode": "skip", "distortion": 8.0, "bits": 1},   # J = 9
        {"mode": "inter", "distortion": 6.0, "bits": 3},  # J = 9
    ]

    forward = choose_mode(candidates, 1.0)
    backward = choose_mode(list(reversed(candidates)), 1.0)

    assert forward["mode"] == "inter"
    assert backward["mode"] == "inter"
    assert forward["cost"] == pytest.approx(9.0, rel=TOL, abs=TOL)
    assert backward["cost"] == pytest.approx(9.0, rel=TOL, abs=TOL)


def test_validate_candidates_rejects_non_monotonic_rate_distortion():
    valid = [
        {"mode": "skip", "distortion": 9.0, "bits": 1},
        {"mode": "inter", "distortion": 2.0, "bits": 4},
        {"mode": "intra", "distortion": 0.5, "bits": 12},
    ]
    assert validate_candidates(valid) == []

    invalid = [
        {"mode": "skip", "distortion": 1.0, "bits": 1},
        # More bits but also more distortion: dominated and invalid.
        {"mode": "inter", "distortion": 5.0, "bits": 8},
    ]
    errors = validate_candidates(invalid)

    assert errors
    assert all(isinstance(error, str) for error in errors)


def test_rd_batch_recomputes_lambda_per_block_qp():
    blocks = [
        {
            "block_id": "low-qp",
            "qp": 18,
            "candidates": [
                {"mode": "skip", "distortion": 90.0, "bits": 1},
                {"mode": "intra", "distortion": 0.0, "bits": 20},
            ],
        },
        {
            "block_id": "high-qp",
            "qp": 30,
            "candidates": [
                {"mode": "skip", "distortion": 90.0, "bits": 1},
                {"mode": "intra", "distortion": 0.0, "bits": 20},
            ],
        },
    ]

    result = rd_batch(blocks, {"lambda_scale": 1.0})

    assert [entry["mode"] for entry in result["blocks"]] == ["intra", "skip"]
    assert result["blocks"][0]["lambda"] == pytest.approx(3.4, rel=TOL, abs=TOL)
    assert result["blocks"][1]["lambda"] == pytest.approx(54.4, rel=TOL, abs=TOL)
    assert result["blocks"][0]["cost"] == pytest.approx(68.0, rel=TOL, abs=TOL)
    assert result["blocks"][1]["cost"] == pytest.approx(144.4, rel=TOL, abs=TOL)
    assert result["total_cost"] == pytest.approx(212.4, rel=TOL, abs=TOL)


def test_lambda_quadruples_when_qp_increases_by_six():
    for qp in (0, 6, 12, 24, 36, 45):
        base = lambda_for_qp(qp)
        higher = lambda_for_qp(qp + 6)
        assert higher == pytest.approx(4.0 * base, rel=TOL)


def test_choose_mode_matches_exhaustive_minimum():
    lam = 2.5
    candidates = [
        {"mode": "skip", "distortion": 15.0, "bits": 1},   # J = 17.5
        {"mode": "inter", "distortion": 2.0, "bits": 6},   # J = 17.0
        {"mode": "intra", "distortion": 0.0, "bits": 14},  # J = 35.0
    ]
    exhaustive = min(candidates, key=lambda candidate: _j(candidate, lam))

    result = choose_mode(candidates, lam)

    assert exhaustive["mode"] == "inter"
    assert result["mode"] == exhaustive["mode"]
    assert result["cost"] == pytest.approx(_j(exhaustive, lam), rel=TOL, abs=TOL)


def test_choose_mode_is_order_independent():
    candidates = [
        {"mode": "skip", "distortion": 18.0, "bits": 1},   # J = 20
        {"mode": "inter", "distortion": 6.0, "bits": 5},   # J = 16
        {"mode": "intra", "distortion": 0.0, "bits": 14},  # J = 28
    ]
    orders = [
        candidates,
        list(reversed(candidates)),
        [candidates[1], candidates[2], candidates[0]],
        [candidates[2], candidates[0], candidates[1]],
    ]

    chosen = {choose_mode(order, 2.0)["mode"] for order in orders}

    assert chosen == {"inter"}


def test_skip_mode_selected_on_exact_match():
    candidates = [
        {"mode": "intra", "distortion": 0.0, "bits": 12},
        {"mode": "inter", "distortion": 0.0, "bits": 8},
        {"mode": "skip", "distortion": 0.0, "bits": 1},
    ]

    result = choose_mode(candidates, 4.0)

    assert result["mode"] == "skip"
    assert result["cost"] == pytest.approx(4.0, rel=TOL, abs=TOL)


def test_two_blocks_are_optimized_independently():
    qp = 24
    blocks = [
        {
            "block_id": "residual-block",
            "qp": qp,
            "candidates": [
                {"mode": "skip", "distortion": 1000.0, "bits": 1},
                {"mode": "intra", "distortion": 0.0, "bits": 16},
            ],
        },
        {
            "block_id": "exact-block",
            "qp": qp,
            "candidates": [
                {"mode": "skip", "distortion": 0.0, "bits": 1},
                {"mode": "intra", "distortion": 0.0, "bits": 16},
            ],
        },
    ]

    result = rd_batch(blocks, {})

    assert [entry["mode"] for entry in result["blocks"]] == ["intra", "skip"]
    for entry, block in zip(result["blocks"], blocks):
        lam = lambda_for_qp(block["qp"])
        expected = choose_mode(block["candidates"], lam)
        assert entry["block_id"] == block["block_id"]
        assert entry["mode"] == expected["mode"]
        assert entry["cost"] == pytest.approx(expected["cost"], rel=TOL, abs=TOL)
        assert entry["lambda"] == pytest.approx(lam, rel=TOL, abs=TOL)
    assert result["total_cost"] == pytest.approx(
        sum(entry["cost"] for entry in result["blocks"]), rel=TOL, abs=TOL
    )
