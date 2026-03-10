from arknova_rl.trainer import _restore_metrics


def test_restore_metrics_defaults_missing_timing_fields():
    restored = _restore_metrics(
        [
            {
                "step_count": 10,
                "episode_count": 2,
                "avg_completed_rounds": 5.0,
                "avg_terminal_score_diff_abs": 3.0,
                "terminal_reason_counts": {"x": 1},
                "policy_loss": 1.0,
                "value_loss": 2.0,
                "entropy": 3.0,
                "total_loss": 4.0,
            }
        ]
    )

    assert len(restored) == 1
    metrics = restored[0]
    assert metrics.episode_time_avg_sec == 0.0
    assert metrics.episode_time_min_sec == 0.0
    assert metrics.episode_time_max_sec == 0.0
    assert metrics.update_time_sec == 0.0
    assert metrics.model_update_time_sec == 0.0
