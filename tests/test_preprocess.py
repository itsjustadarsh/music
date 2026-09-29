import numpy as np

from src.config import CLIP_LENGTH_SAMPLES, HOP_LENGTH, N_MELS, SAMPLE_RATE
from src.data.preprocess import AST_SAMPLE_RATE, audio_to_logmel


def _sine_wave(freq: float = 440.0, duration: float = 3.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_logmel_shape_matches_expected_mel_bins():
    clip = _sine_wave(duration=CLIP_LENGTH_SAMPLES / SAMPLE_RATE)
    logmel = audio_to_logmel(clip)
    expected_frames = 1 + len(clip) // HOP_LENGTH
    assert logmel.shape[0] == N_MELS
    assert abs(logmel.shape[1] - expected_frames) <= 1


def test_logmel_is_finite_and_float32():
    clip = _sine_wave()
    logmel = audio_to_logmel(clip)
    assert logmel.dtype == np.float32
    assert np.isfinite(logmel).all()


def test_logmel_of_silence_is_bounded():
    silence = np.zeros(CLIP_LENGTH_SAMPLES, dtype=np.float32)
    logmel = audio_to_logmel(silence)
    assert np.isfinite(logmel).all()


def test_resample_to_ast_rate_preserves_duration():
    import librosa

    clip = _sine_wave(duration=3.0)
    resampled = librosa.resample(clip, orig_sr=SAMPLE_RATE, target_sr=AST_SAMPLE_RATE)
    expected_len = int(3.0 * AST_SAMPLE_RATE)
    assert abs(len(resampled) - expected_len) <= 1


def _fake_manifest(tmp_path):
    import pandas as pd

    rows = [
        {"logmel_path": f"logmel/{split}/Rock/{tid:06d}_{i}.npy",
         "waveform_path": f"waveform16k/{split}/Rock/{tid:06d}_{i}.npy",
         "track_id": tid, "genre": "Rock", "split": split, "subclip_idx": i}
        for split in ("train", "val", "test")
        for tid in range(3)
        for i in range(10)
    ]
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_manifest_filters_to_one_split(tmp_path):
    from src.training.dataset import load_manifest

    df = load_manifest(_fake_manifest(tmp_path), "val")
    assert len(df) == 30
    assert set(df["split"]) == {"val"}


def test_load_manifest_caps_subclips_per_track(tmp_path):
    from src.training.dataset import load_manifest

    df = load_manifest(_fake_manifest(tmp_path), "train", max_subclips_per_track=2)
    assert len(df) == 6  # 3 tracks x 2 sub-clips
    assert df["subclip_idx"].max() == 1
    assert sorted(df["track_id"].unique()) == [0, 1, 2]
