import numpy as np

from src.data.risk_dataset import RiskAwareFallDataset
from src.features.context_features import extract_scene_features


def make_sequence(vertical_drop=0.0, sway=0.0, lean=0.0):
    seq = np.zeros((30, 17, 3), dtype=np.float32)
    for t in range(30):
        frac = t / 29.0
        cx = 0.5 + sway * np.sin(frac * np.pi)
        cy = 0.45 + vertical_drop * frac
        shoulder_dx = 0.08 + lean * frac
        hip_dx = 0.06
        ankle_dx = 0.10

        points = {
            5: (cx - shoulder_dx, cy - 0.18),
            6: (cx + shoulder_dx, cy - 0.18),
            11: (cx - hip_dx, cy),
            12: (cx + hip_dx, cy),
            15: (cx - ankle_dx, cy + 0.26),
            16: (cx + ankle_dx, cy + 0.26),
            13: (cx - 0.08, cy + 0.14),
            14: (cx + 0.08, cy + 0.14),
            0: (cx, cy - 0.28),
        }
        for idx, (x, y) in points.items():
            seq[t, idx, 0] = x
            seq[t, idx, 1] = y
            seq[t, idx, 2] = 1.0
    return seq


def test_scene_features_are_not_zero_for_valid_sequence():
    seq = make_sequence(vertical_drop=0.15, sway=0.05, lean=0.05)
    feat = extract_scene_features(seq, metadata={"scene": "bathroom"})
    assert feat.shape == (18,)
    assert np.count_nonzero(feat) > 4


def test_risk_aware_dataset_produces_higher_score_for_unstable_sequence():
    stable = {"keypoints": make_sequence(), "label": 0, "metadata": {"action": "ADL_001"}}
    unstable = {"keypoints": make_sequence(vertical_drop=0.22, sway=0.08, lean=0.12), "label": 1, "metadata": {"action": "Fall_forward"}}
    dataset = RiskAwareFallDataset([stable, unstable], risk_mode=True)
    _, stable_score, _ = dataset[0]
    _, unstable_score, _ = dataset[1]
    assert unstable_score.item() > stable_score.item()


def test_multimodal_dataset_returns_real_scene_features():
    sample = {"keypoints": make_sequence(vertical_drop=0.10, sway=0.04), "label": 1, "metadata": {"scene": "kitchen"}}
    dataset = RiskAwareFallDataset([sample], multimodal=True)
    _, scene_feat, _, _ = dataset[0]
    assert scene_feat.shape[0] == 18
    assert float(scene_feat.abs().sum()) > 0.0
