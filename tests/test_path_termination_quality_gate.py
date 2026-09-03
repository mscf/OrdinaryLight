from tests.gates.path_termination_quality import evaluate_termination


def _quality(*, error=1.0, temporal=1.0, low_frequency=1.0, bias=0.0):
    return {
        "relative_rmse_mean": error,
        "temporal_residual_rmse_mean": temporal,
        "low_frequency_energy_ratio_mean": low_frequency,
        "bias_mean": bias,
    }


def test_roulette_allows_bounded_variance_without_bias():
    failures = evaluate_termination(
        _quality(), _quality(error=1.2, temporal=1.2, low_frequency=1.1),
        max_error_ratio=1.25, max_temporal_ratio=1.25,
        max_low_frequency_ratio=1.15, max_abs_bias=0.01,
    )
    assert failures == []


def test_roulette_rejects_bias_and_excessive_variance():
    failures = evaluate_termination(
        _quality(), _quality(error=1.3, temporal=1.3,
                            low_frequency=1.2, bias=0.02),
        max_error_ratio=1.25, max_temporal_ratio=1.25,
        max_low_frequency_ratio=1.15, max_abs_bias=0.01,
    )
    assert len(failures) == 4
