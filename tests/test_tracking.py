import numpy as np

from poselab.skeleton import LANDMARK_NAMES
from poselab.tracking import PersonTracker, format_warning
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


def _person_with_torso(cx: float, cy: float = 200.0, half: float = 40.0) -> PersonPose:
    """胴体矩形 (肩・腰) が (cx, cy) を中心に広がる人物。"""
    person = _person(cx, cy)
    # 両肩 (11, 12) と両腰 (23, 24) を矩形に配置
    coords = {
        11: (cx - half, cy - half), 12: (cx + half, cy - half),
        23: (cx - half, cy + half), 24: (cx + half, cy + half),
    }
    for idx, (x, y) in coords.items():
        person.keypoints[idx].x_px = x
        person.keypoints[idx].y_px = y
    return person


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


def test_ids_survive_crossing():
    """逆方向に移動する 2 人が交差しても ID が入れ替わらない。"""
    tracker = PersonTracker()
    final = {}
    for t in range(41):
        a = _person(100.0 + 20.0 * t, y=100.0)   # 左から右へ
        b = _person(900.0 - 20.0 * t, y=130.0)   # 右から左へ
        persons = [b, a] if t % 2 else [a, b]    # 検出順も揺らす
        tracker.assign(persons, 1280, 720, frame_index=t)
        for p in persons:
            final[p.person_index] = p.keypoints[0].x_px
    # P0 (左発) は右側へ、P1 (右発) は左側へ抜けている
    assert final[0] > 700
    assert final[1] < 200


def test_crossing_warning_recorded():
    tracker = PersonTracker()
    for t in range(41):
        a = _person_with_torso(100.0 + 20.0 * t, cy=200.0)
        b = _person_with_torso(900.0 - 20.0 * t, cy=230.0)
        tracker.assign([a, b], 1280, 720, frame_index=t, timestamp_ms=t * 33.3)
    warnings = tracker.get_warnings()
    crossing = [w for w in warnings if w["type"] == "crossing"]
    assert crossing, "交差時に警告が記録されるべき"
    w = crossing[0]
    assert w["ids"] == [0, 1]
    assert w["frame_start"] <= 20 <= w["frame_end"]  # 交差はフレーム 20 付近
    assert "接近" in format_warning(w)


def test_reappear_warning():
    tracker = PersonTracker(max_missed=30, reappear_threshold=5)
    tracker.assign([_person(100.0)], 640, 480, frame_index=0)
    for t in range(1, 11):  # 10 フレーム未検出
        tracker.assign([], 640, 480, frame_index=t)
    back = _person(105.0)
    tracker.assign([back], 640, 480, frame_index=11, timestamp_ms=366.3)
    assert back.person_index == 0
    warnings = tracker.get_warnings()
    assert any(w["type"] == "reappear" and w["ids"] == [0] for w in warnings)
    assert "再出現" in format_warning(warnings[0])


def test_appearance_disambiguates():
    """服装 (胴体色) が異なれば、位置が紛らわしくても正しく対応付く。"""
    h, w = 480, 1280
    frame1 = np.zeros((h, w, 3), dtype=np.uint8)
    frame1[:, :640] = (0, 0, 255)    # 左半分: 赤い服の人
    frame1[:, 640:] = (255, 0, 0)    # 右半分: 青い服の人
    tracker = PersonTracker(appearance_weight=1.0)
    a = _person_with_torso(300.0)
    b = _person_with_torso(900.0)
    tracker.assign([a, b], w, h, frame=frame1, frame_index=0)
    assert (a.person_index, b.person_index) == (0, 1)

    # 次フレーム: 両者がほぼ同じ位置に来てしまった (位置では曖昧)。
    # 赤背景側に居るのが P0 のはず
    frame2 = np.zeros((h, w, 3), dtype=np.uint8)
    frame2[:, :600] = (0, 0, 255)
    frame2[:, 600:] = (255, 0, 0)
    red_person = _person_with_torso(540.0)   # 赤領域
    blue_person = _person_with_torso(680.0)  # 青領域
    tracker.assign([blue_person, red_person], w, h, frame=frame2, frame_index=1)
    assert red_person.person_index == 0
    assert blue_person.person_index == 1
