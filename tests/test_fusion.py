"""Tests for Epoch-Scheduled Conditional Fusion."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fusion.data import TrackBatch
from src.fusion.escf import ESCF, confidence_features
from src.fusion.schedules import SCHEDULES, get_schedule, supervisor


def make_batch(n_tracks=4, n_eps=10, dim_a=16, dim_b=32, n_classes=8, ragged=True):
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(n_tracks, n_eps, dtype=torch.bool)
    if ragged:
        mask[0, 6:] = False  # a short track, as real 30s tracks sometimes are
    return TrackBatch(
        emb_a=torch.randn(n_tracks, n_eps, dim_a, generator=g),
        emb_b=torch.randn(n_tracks, n_eps, dim_b, generator=g),
        logits_a=torch.randn(n_tracks, n_eps, n_classes, generator=g),
        logits_b=torch.randn(n_tracks, n_eps, n_classes, generator=g),
        mask=mask,
        labels=torch.randint(0, n_classes, (n_tracks,), generator=g),
        track_ids=np.arange(n_tracks),
    )


# --- schedules ------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(SCHEDULES))
def test_schedules_stay_in_unit_interval(name):
    fn = get_schedule(name)
    assert all(0.0 <= fn(e, 20) <= 1.0 for e in range(1, 21))


@pytest.mark.parametrize("name", ["linear", "cosine"])
def test_smooth_schedules_span_prior_to_learned(name):
    fn = get_schedule(name)
    assert fn(1, 20) == pytest.approx(0.0)
    assert fn(20, 20) == pytest.approx(1.0)


def test_supervisor_schedule_reproduces_the_original_curve():
    """The originally suggested max(0.8, t/T): flat at 0.8 for most of training,
    reaching 1.0 only at the final epoch. Kept so the paper can compare against it
    rather than argue about it."""
    assert supervisor(1, 10) == pytest.approx(0.8)
    assert supervisor(8, 10) == pytest.approx(0.8)
    assert supervisor(10, 10) == pytest.approx(1.0)


def test_unknown_schedule_rejected():
    with pytest.raises(ValueError, match="Unknown schedule"):
        get_schedule("nope")


# --- fusion mechanics -----------------------------------------------------

def test_confidence_features_are_ordered_by_certainty():
    confident = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    unsure = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    c_conf, c_unsure = confidence_features(confident), confidence_features(unsure)
    assert c_conf[0, 0] > c_unsure[0, 0]        # max prob
    assert c_conf[0, 1] > c_unsure[0, 1]        # margin
    assert c_conf[0, 2] > c_unsure[0, 2]        # negative entropy


@pytest.mark.parametrize("variant", ESCF.VARIANTS)
def test_every_variant_produces_track_logits(variant):
    batch = make_batch()
    model = ESCF(dim_a=16, dim_b=32, variant=variant)
    logits = model(batch, epoch=5, total_epochs=10)
    assert logits.shape == (len(batch), 8)
    assert torch.isfinite(logits).all()


def test_attention_never_weights_padding():
    batch = make_batch()
    model = ESCF(dim_a=16, dim_b=32, variant="conditional", use_attention=True)
    _, weights = model(batch, epoch=5, total_epochs=10, return_weights=True)
    beta = weights["beta"]
    assert torch.allclose(beta.sum(dim=1), torch.ones(len(batch)), atol=1e-5)
    assert beta[0, 6:].abs().max().item() == pytest.approx(0.0, abs=1e-8)


def test_uniform_attention_is_the_mean_over_real_episodes():
    batch = make_batch()
    model = ESCF(dim_a=16, dim_b=32, variant="fixed", use_attention=False)
    _, weights = model(batch, epoch=1, total_epochs=1, return_weights=True)
    beta = weights["beta"]
    assert beta[0, 0].item() == pytest.approx(1 / 6)   # 6 real episodes
    assert beta[1, 0].item() == pytest.approx(1 / 10)
    assert beta[0, 6:].sum().item() == pytest.approx(0.0)


def test_single_backbone_variants_ignore_the_other_backbone():
    batch = make_batch()
    cnn_only = ESCF(dim_a=16, dim_b=32, variant="cnn_only", use_attention=False)
    out_before = cnn_only(batch, epoch=1, total_epochs=1)

    perturbed = TrackBatch(
        batch.emb_a, batch.emb_b, batch.logits_a,
        batch.logits_b + 100.0,  # wreck backbone B
        batch.mask, batch.labels, batch.track_ids,
    )
    out_after = cnn_only(perturbed, epoch=1, total_epochs=1)
    assert torch.allclose(out_before, out_after)


def test_curriculum_moves_alpha_from_prior_to_learned_gate():
    """lambda_t = 0 must reproduce the prior exactly; lambda_t = 1 the learned gate."""
    batch = make_batch()
    model = ESCF(dim_a=16, dim_b=32, variant="conditional", schedule="linear", alpha_prior=0.5)

    _, first = model(batch, epoch=1, total_epochs=10, return_weights=True)
    assert first["lambda"] == pytest.approx(0.0)
    assert torch.allclose(first["alpha"], torch.full_like(first["alpha"], 0.5))

    _, last = model(batch, epoch=10, total_epochs=10, return_weights=True)
    assert last["lambda"] == pytest.approx(1.0)
    assert not torch.allclose(last["alpha"], torch.full_like(last["alpha"], 0.5))


def test_conditional_gate_varies_across_episodes():
    """The whole claim: alpha is per-episode, not one number per track."""
    batch = make_batch(ragged=False)
    model = ESCF(dim_a=16, dim_b=32, variant="conditional", schedule="constant")
    _, weights = model(batch, epoch=1, total_epochs=1, return_weights=True)
    assert weights["alpha"].std(dim=1).min().item() > 0.0


def test_global_variant_uses_one_weight_everywhere():
    batch = make_batch(ragged=False)
    model = ESCF(dim_a=16, dim_b=32, variant="global")
    _, weights = model(batch, epoch=1, total_epochs=1, return_weights=True)
    assert weights["alpha"].std().item() == pytest.approx(0.0, abs=1e-6)


def test_alpha_stays_a_convex_weight():
    batch = make_batch()
    model = ESCF(dim_a=16, dim_b=32, variant="conditional", schedule="constant")
    _, weights = model(batch, epoch=1, total_epochs=1, return_weights=True)
    assert weights["alpha"].min().item() >= 0.0
    assert weights["alpha"].max().item() <= 1.0


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="Unknown variant"):
        ESCF(dim_a=16, dim_b=32, variant="nope")
