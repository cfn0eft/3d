from poselab.skeleton import LANDMARK_NAMES
from poselab.tracking import PersonTracker
from poselab.types import Keypoint, PersonPose


def _person(x: float, y: float = 100.0) -> PersonPose:
    keypoints = [
        Keypoint(
            index=i, name=name, x_norm=0.5, y_norm=0.5, z=0.0,
            visibility=0.9, presence=0.9, x_px=x, y_px=y,
        )
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    return PersonPose(person_index=0, keypoints=keypoints)


def test_ids_stable_when_detection_order_flips():
    tracker = PersonTracker()
    a, b = _person(100.0), _person(500.0)
    tracker.assign([a, b], 640, 480)
    assert (a.person_index, b.person_index) == (0, 1)

    # 次フレームで検出順が入れ替わっても ID は位置に追従する
    a2, b2 = _person(505.0), _person(102.0)
    out = tracker.assign([a2, b2], 640, 480)
    assert b2.person_index == 0  # x=102 は前フレームの x=100 と同一人物
    assert a2.person_index == 1
    assert [p.person_index for p in out] == [0, 1]  # ID 順に整列


def test_new_person_gets_new_id():
    tracker = PersonTracker()
    a = _person(100.0)
    tracker.assign([a], 640, 480)
    b = _person(100.0)
    c = _person(600.0)
    tracker.assign([b, c], 640, 480)
    assert b.person_index == 0
    assert c.person_index == 1


def test_id_survives_short_gap():
    tracker = PersonTracker(max_missed=5)
    a = _person(100.0)
    tracker.assign([a], 640, 480)
    for _ in range(3):  # 3 フレーム未検出
        tracker.assign([], 640, 480)
    back = _person(110.0)
    tracker.assign([back], 640, 480)
    assert back.person_index == 0


def test_id_dropped_after_long_gap():
    tracker = PersonTracker(max_missed=2)
    tracker.assign([_person(100.0)], 640, 480)
    for _ in range(3):
        tracker.assign([], 640, 480)
    fresh = _person(100.0)
    tracker.assign([fresh], 640, 480)
    assert fresh.person_index == 1  # 古いトラックは破棄され新 ID


def test_far_person_not_matched():
    tracker = PersonTracker(max_distance=0.1)
    tracker.assign([_person(0.0, 0.0)], 640, 480)
    far = _person(600.0, 400.0)
    tracker.assign([far], 640, 480)
    assert far.person_index == 1


def test_reset():
    tracker = PersonTracker()
    tracker.assign([_person(100.0)], 640, 480)
    tracker.reset()
    p = _person(100.0)
    tracker.assign([p], 640, 480)
    assert p.person_index == 0
