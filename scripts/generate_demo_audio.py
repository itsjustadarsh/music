#!/usr/bin/env python3
"""Generate synthetic demo audio files for the walkthrough.

Creates short WAV clips with distinct sonic characters so each one
classifies differently when fed to the CNN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

DEST = Path(__file__).resolve().parent.parent / "demo-files"
DEST.mkdir(exist_ok=True)
SR = 22050
DUR = 5  # seconds
N = SR * DUR


def _norm(y: np.ndarray) -> np.ndarray:
    peak = np.abs(y).max()
    return (y / peak * 0.8).astype(np.float32) if peak > 0 else y.astype(np.float32)


def sine_chord(freqs: list[float], dur: float = DUR) -> np.ndarray:
    t = np.linspace(0, dur, int(SR * dur), dtype=np.float64)
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return _norm(y)


def make_electronic():
    """Fast repeating synth pattern with sub-bass."""
    t = np.linspace(0, DUR, N, dtype=np.float64)
    bass = np.sin(2 * np.pi * 55 * t)
    # Square wave lead at 440 Hz, gated at 8 Hz
    lead = np.sign(np.sin(2 * np.pi * 440 * t))
    gate = (np.sin(2 * np.pi * 8 * t) > 0).astype(float)
    y = 0.5 * bass + 0.4 * lead * gate
    # Add noise hi-hat
    hh = np.random.default_rng(42).normal(0, 0.15, N)
    hh_gate = (np.sin(2 * np.pi * 16 * t) > 0.7).astype(float)
    y += hh * hh_gate
    return _norm(y)


def make_hiphop():
    """Boomy kick pattern + snappy noise hits."""
    t = np.linspace(0, DUR, N, dtype=np.float64)
    rng = np.random.default_rng(7)
    # Kick at ~2.67 Hz (160 BPM)
    kick_env = np.exp(-30 * (t % 0.375))
    kick = np.sin(2 * np.pi * 60 * t) * kick_env
    # Snare (noise burst) on off-beats
    snare_env = np.exp(-40 * ((t + 0.1875) % 0.375))
    snare = rng.normal(0, 1, N) * snare_env * 0.3
    # Simple bass line
    bass = 0.3 * np.sin(2 * np.pi * 80 * t) * (np.sin(2 * np.pi * 1.33 * t) > 0).astype(float)
    return _norm(kick + snare + bass)


def make_rock():
    """Distorted power chord + steady beat."""
    t = np.linspace(0, DUR, N, dtype=np.float64)
    rng = np.random.default_rng(99)
    # Power chord E5 (E2=82.4 Hz + B2=123.5 Hz + E3=164.8 Hz)
    chord = (np.sin(2 * np.pi * 82.4 * t) +
             np.sin(2 * np.pi * 123.5 * t) +
             np.sin(2 * np.pi * 164.8 * t))
    # Distortion (hard clip)
    chord = np.clip(chord * 3, -1, 1)
    # Steady 4/4 kick
    kick_env = np.exp(-25 * (t % 0.5))
    kick = np.sin(2 * np.pi * 80 * t) * kick_env * 0.5
    # Crash noise
    crash = rng.normal(0, 0.05, N)
    return _norm(chord + kick + crash)


def make_folk():
    """Arpeggiated acoustic-style chord (clean harmonics)."""
    t = np.linspace(0, DUR, N, dtype=np.float64)
    # C major arpeggio with harmonics
    notes = [261.6, 329.6, 392.0, 523.3]  # C4 E4 G4 C5
    y = np.zeros(N)
    for i, f in enumerate(notes):
        env = np.exp(-3 * ((t - i * 0.15) % 0.6))
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t)
        y += tone * env * 0.4
    return _norm(y)


def make_pop():
    """Bright synth melody with steady four-on-the-floor."""
    t = np.linspace(0, DUR, N, dtype=np.float64)
    # Melody: simple repeating phrase
    melody_freqs = [523.3, 587.3, 659.3, 587.3]  # C5 D5 E5 D5
    y = np.zeros(N)
    for i, f in enumerate(melody_freqs):
        start = int(i * 0.5 * SR)
        end = min(start + int(0.45 * SR), N)
        seg_t = np.linspace(0, 0.45, end - start)
        env = np.exp(-2 * seg_t)
        y[start:end] += np.sin(2 * np.pi * f * seg_t) * env
    # Loop the pattern
    pattern_len = int(2.0 * SR)
    if pattern_len > 0:
        full = np.tile(y[:pattern_len], N // pattern_len + 1)[:N]
    else:
        full = y
    # Add kick
    kick_env = np.exp(-20 * (t % 0.5))
    kick = np.sin(2 * np.pi * 100 * t) * kick_env * 0.4
    return _norm(full + kick)


def make_experimental():
    """Atonal textures, glitches, spectral noise."""
    rng = np.random.default_rng(13)
    t = np.linspace(0, DUR, N, dtype=np.float64)
    # Frequency sweep
    sweep = np.sin(2 * np.pi * (100 + 2000 * t / DUR) * t)
    # Random amplitude modulation
    n_chunks = (N + 999) // 1000
    am = rng.uniform(0.2, 1.0, n_chunks)
    am = np.repeat(am, 1000)[:N]
    # Granular noise bursts
    noise = rng.normal(0, 0.3, N)
    burst = (rng.random(N) > 0.95).astype(float)
    y = sweep * am * 0.5 + noise * burst
    return _norm(y)


def make_instrumental():
    """Piano-like tones, no percussion."""
    t = np.linspace(0, DUR, N, dtype=np.float64)
    # Piano-ish envelope: fast attack, slow decay
    notes = [
        (0.0, 261.6), (0.8, 293.7), (1.6, 329.6),
        (2.4, 349.2), (3.2, 392.0), (4.0, 440.0),
    ]
    y = np.zeros(N)
    for onset, freq in notes:
        start = int(onset * SR)
        seg_len = min(int(1.5 * SR), N - start)
        seg_t = np.linspace(0, 1.5, seg_len)
        env = np.exp(-2.5 * seg_t) * (1 - np.exp(-100 * seg_t))
        tone = (np.sin(2 * np.pi * freq * seg_t) +
                0.5 * np.sin(2 * np.pi * 2 * freq * seg_t) +
                0.25 * np.sin(2 * np.pi * 3 * freq * seg_t))
        y[start:start + seg_len] += tone * env * 0.3
    return _norm(y)


def make_international():
    """Microtonal-ish melody with drone (sitar/oud style)."""
    t = np.linspace(0, DUR, N, dtype=np.float64)
    # Drone on Sa (C3 = 130.8 Hz)
    drone = np.sin(2 * np.pi * 130.8 * t) + 0.3 * np.sin(2 * np.pi * 261.6 * t)
    # Melody with quarter-tones
    melody_freqs = [293.7, 311.1, 349.2, 370.0, 415.3, 370.0, 311.1]
    y = np.zeros(N)
    for i, f in enumerate(melody_freqs):
        start = int(i * 0.6 * SR)
        seg_len = min(int(0.55 * SR), N - start)
        seg_t = np.linspace(0, 0.55, seg_len)
        # Vibrato
        vib = np.sin(2 * np.pi * 5 * seg_t) * 3
        env = np.exp(-1.5 * seg_t)
        tone = np.sin(2 * np.pi * (f + vib) * seg_t) * env
        y[start:start + seg_len] += tone * 0.5
    pattern_len = int(4.2 * SR)
    full = np.tile(y[:pattern_len], N // pattern_len + 1)[:N] if pattern_len > 0 else y
    return _norm(drone * 0.4 + full)


GENERATORS = {
    "electronic_synth_demo": make_electronic,
    "hiphop_beat_demo": make_hiphop,
    "rock_powerchord_demo": make_rock,
    "folk_arpeggio_demo": make_folk,
    "pop_melody_demo": make_pop,
    "experimental_glitch_demo": make_experimental,
    "instrumental_piano_demo": make_instrumental,
    "international_drone_demo": make_international,
}


def main():
    print(f"Generating {len(GENERATORS)} demo audio files in {DEST}/\n")
    for name, gen in GENERATORS.items():
        path = DEST / f"{name}.wav"
        y = gen()
        sf.write(str(path), y, SR)
        dur_s = len(y) / SR
        print(f"  {path.name:<35} {dur_s:.1f}s  {len(y):>7,} samples")
    print(f"\nDone. Upload these files via the frontend or /classify API endpoint.")


if __name__ == "__main__":
    main()
