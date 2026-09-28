import numpy as np

from examples.study_navigation import OracleMap, evaluate_routes, obstacle_labels, sample_map


def test_known_wall_and_opening_labels():
    points = np.array([[0.0, 0.0], [0.0, 0.6], [0.0, 1.2], [1.0, 0.0]])
    np.testing.assert_array_equal(obstacle_labels(points), [1, 0, 1, 0])


def test_sample_map_is_balanced_and_reproducible():
    x1, y1 = sample_map(7, n=200)
    x2, y2 = sample_map(7, n=200)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(np.bincount(y1), [100, 100])


def test_exact_map_reference_reaches_goal_without_wall_breach():
    result = evaluate_routes(OracleMap())
    assert result["collision_free_success"] == 1.0
    assert result["true_wall_breaches"] == 0.0
