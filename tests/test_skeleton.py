from poselab.skeleton import (
    LANDMARK_NAMES,
    NUM_LANDMARKS,
    SKELETON_EDGES,
    landmark_side,
)


def test_landmark_count():
    assert NUM_LANDMARKS == 33
    assert len(set(LANDMARK_NAMES)) == 33


def test_edges_reference_valid_indices():
    for a, b in SKELETON_EDGES:
        assert 0 <= a < NUM_LANDMARKS
        assert 0 <= b < NUM_LANDMARKS
        assert a != b


def test_edges_unique():
    normalized = {tuple(sorted(e)) for e in SKELETON_EDGES}
    assert len(normalized) == len(SKELETON_EDGES)


def test_landmark_side():
    assert landmark_side("left_wrist") == "left"
    assert landmark_side("right_knee") == "right"
    assert landmark_side("nose") == "center"
    assert landmark_side("mouth_left") == "left"
