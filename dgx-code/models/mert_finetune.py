"""MERT fine-tuning models for genre classification.

Three fine-tuning strategies, from lightest to heaviest:

1. LoRA (default) — inject low-rank adapters into attention layers, freeze everything else.
   ~2M trainable params for rank=16. Best accuracy/overfitting tradeoff on 6.4k tracks.

2. Partial unfreeze — freeze early layers, unfreeze last N transformer layers + classifier.
   More expressive but higher overfit risk.

3. Full fine-tune — unfreeze everything. Only viable with strong regularisation.

All variants use gradient checkpointing to fit in GPU memory and process raw 24kHz audio.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, HubertConfig, HubertModel, Wav2Vec2FeatureExtractor

log = logging.getLogger(__name__)

MERT_95M_CHECKPOINT = "m-a-p/MERT-v1-95M"
MERT_330M_CHECKPOINT = "m-a-p/MERT-v1-330M"
MERT_SAMPLE_RATE = 24000


def load_mert_backbone(checkpoint: str, gradient_checkpointing: bool = True):
    """Load MERT as a HubertModel (see laptop pipeline's mert_embed.py for rationale)."""
    src_config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
    src_config.conv_pos_batch_norm = False

    allowed = set(HubertConfig().to_dict())
    config = HubertConfig(**{k: v for k, v in src_config.to_dict().items() if k in allowed})

    model = HubertModel.from_pretrained(checkpoint, config=config)

    if gradient_checkpointing:
        # Non-reentrant checkpointing: the frozen conv front-end yields inputs without
        # requires_grad, which with reentrant checkpointing silently blocks gradients
        # to LoRA / unfrozen layers.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        checkpoint, trust_remote_code=True
    )
    return model, extractor


class MertFinetune(nn.Module):
    """End-to-end MERT fine-tuning for genre classification.

    Takes raw 24kHz waveforms, runs through MERT, pools hidden states,
    and classifies. Supports LoRA, partial unfreeze, and full fine-tuning.

    Args:
        checkpoint: HF model name for MERT variant
        num_classes: number of output genres
        pool_mode: how to aggregate MERT's hidden states
            - "last": use only the last transformer layer
            - "mean_all": mean over all layers (default, matches frozen probe insight)
            - "weighted": learnable weighted sum of all layers
            - "concat_last_4": concatenate last 4 layers
        unfreeze_layers: number of transformer layers to unfreeze from the top.
            0 = freeze all (probe-only), -1 = unfreeze all
        use_lora: apply LoRA adapters to attention layers
        lora_rank: rank for LoRA adapters
        lora_alpha: alpha scaling for LoRA
        classifier_hidden: hidden size for the classification head
        dropout: dropout rate
    """

    def __init__(
        self,
        checkpoint: str = MERT_330M_CHECKPOINT,
        num_classes: int = 8,
        pool_mode: str = "weighted",
        unfreeze_layers: int = 0,
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        classifier_hidden: int = 512,
        dropout: float = 0.3,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.use_lora = use_lora

        # Load backbone
        self.backbone, self.feature_extractor = load_mert_backbone(
            checkpoint, gradient_checkpointing=gradient_checkpointing
        )

        hidden_size = self.backbone.config.hidden_size
        num_layers = self.backbone.config.num_hidden_layers + 1  # +1 for conv projection

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)
        elif unfreeze_layers != 0:
            self._unfreeze_top_layers(unfreeze_layers)

        # Pool mode determines input dim to classifier
        if pool_mode == "last":
            cls_input_dim = hidden_size
        elif pool_mode == "mean_all":
            cls_input_dim = hidden_size
        elif pool_mode == "weighted":
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
            cls_input_dim = hidden_size
        elif pool_mode == "concat_last_4":
            cls_input_dim = hidden_size * 4
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(cls_input_dim),
            nn.Linear(cls_input_dim, classifier_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, classifier_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden // 2, num_classes),
        )

    def _apply_lora(self, rank: int, alpha: int):
        """Inject LoRA adapters into all attention Q/V projections."""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=rank,
                lora_alpha=alpha,
                target_modules=["q_proj", "v_proj", "k_proj"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[],
            )
            # PEFT tries enable_input_require_grads() when checkpointing is on, which
            # HubertModel doesn't support. Wrap first, then re-enable non-reentrant
            # checkpointing (which doesn't need input grads).
            gc_on = self.backbone.is_gradient_checkpointing
            if gc_on:
                self.backbone.gradient_checkpointing_disable()
            self.backbone = get_peft_model(self.backbone, lora_config)
            if gc_on:
                self.backbone.base_model.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            log.info(
                "LoRA applied: %d trainable params",
                sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            )
        except ImportError:
            log.warning("peft not installed, falling back to manual LoRA")
            self._apply_manual_lora(rank, alpha)

    def _apply_manual_lora(self, rank: int, alpha: int):
        """Manual LoRA implementation when peft is not available."""
        self.lora_layers = nn.ModuleList()
        scaling = alpha / rank

        for layer in self.backbone.encoder.layers:
            attn = layer.attention.self

            # LoRA for Q
            lora_q = LoRALayer(attn.query.in_features, attn.query.out_features, rank, scaling)
            # LoRA for V
            lora_v = LoRALayer(attn.value.in_features, attn.value.out_features, rank, scaling)

            self.lora_layers.append(lora_q)
            self.lora_layers.append(lora_v)

            # Monkey-patch the forward
            original_q = attn.query
            original_v = attn.value
            attn.query = LoRALinear(original_q, lora_q)
            attn.value = LoRALinear(original_v, lora_v)

    def _unfreeze_top_layers(self, n: int):
        """Unfreeze the last n transformer layers."""
        layers = list(self.backbone.encoder.layers)
        if n == -1:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            for layer in layers[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True
        # Always unfreeze the final layer norm
        if hasattr(self.backbone.encoder, "layer_norm"):
            for param in self.backbone.encoder.layer_norm.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        log.info("Unfroze top %d layers: %d trainable backbone params", n, trainable)

    def _pool_hidden_states(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Pool multi-layer hidden states into a single feature vector per input."""
        # Each hidden state: (batch, time, hidden)
        if self.pool_mode == "last":
            return hidden_states[-1].mean(dim=1)

        elif self.pool_mode == "mean_all":
            stacked = torch.stack(hidden_states, dim=0)  # (layers, batch, time, hidden)
            return stacked.mean(dim=(0, 2))  # (batch, hidden)

        elif self.pool_mode == "weighted":
            stacked = torch.stack(hidden_states, dim=0)  # (layers, batch, time, hidden)
            # Mean-pool over time first
            pooled = stacked.mean(dim=2)  # (layers, batch, hidden)
            weights = torch.softmax(self.layer_weights, dim=0)
            # Weighted sum over layers
            weighted = (pooled * weights.view(-1, 1, 1)).sum(dim=0)  # (batch, hidden)
            return weighted

        elif self.pool_mode == "concat_last_4":
            last4 = hidden_states[-4:]
            pooled = [h.mean(dim=1) for h in last4]  # each (batch, hidden)
            return torch.cat(pooled, dim=-1)  # (batch, hidden*4)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, samples) raw 24kHz audio

        Returns:
            logits: (batch, num_classes)
        """
        # Process through MERT feature extractor
        # Note: for batch processing, all waveforms must be same length (handled by dataset)
        device = waveforms.device

        # The feature extractor expects numpy-like inputs; we handle it on the model side
        # For batched processing, we skip the HF feature extractor and directly normalize
        # (MERT's Wav2Vec2FeatureExtractor just does zero-mean unit-variance normalization)
        waveforms = (waveforms - waveforms.mean(dim=-1, keepdim=True)) / (
            waveforms.std(dim=-1, keepdim=True) + 1e-7
        )

        outputs = self.backbone(
            input_values=waveforms,
            output_hidden_states=True,
            return_dict=True,
        )

        pooled = self._pool_hidden_states(outputs.hidden_states)
        logits = self.classifier(pooled)
        return logits

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LoRALayer(nn.Module):
    """Single LoRA adapter: W' = W + BA * scaling."""

    def __init__(self, in_features: int, out_features: int, rank: int, scaling: float):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """Wraps an existing Linear layer with a LoRA adapter."""

    def __init__(self, original: nn.Linear, lora: LoRALayer):
        super().__init__()
        self.original = original
        self.lora = lora
        # Keep original frozen
        for param in self.original.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora(x)


class MertMultiCrop(nn.Module):
    """Multi-crop MERT: processes a track as multiple overlapping windows
    and aggregates predictions.

    During training: randomly samples `n_crops` windows from the track.
    During eval: uses evenly-spaced windows and averages logits.

    This gives the model both local detail and global context without
    requiring memory for the full 30s at once.
    """

    def __init__(
        self,
        checkpoint: str = MERT_330M_CHECKPOINT,
        num_classes: int = 8,
        crop_seconds: float = 5.0,
        n_crops_train: int = 6,
        n_crops_eval: int = 10,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.crop_samples = int(crop_seconds * MERT_SAMPLE_RATE)
        self.n_crops_train = n_crops_train
        self.n_crops_eval = n_crops_eval

        # Use MertFinetune as the per-crop model
        self.model = MertFinetune(
            checkpoint=checkpoint,
            num_classes=num_classes,
            pool_mode="weighted",
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            classifier_hidden=512,
            dropout=dropout,
        )

    def _extract_crops(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Extract crops from full waveforms.

        Args:
            waveforms: (batch, samples)

        Returns:
            crops: (batch * n_crops, crop_samples)
            n_crops: number of crops per track
        """
        batch_size, total_samples = waveforms.shape
        n_crops = self.n_crops_train if self.training else self.n_crops_eval

        if self.training:
            # Random crops
            max_start = total_samples - self.crop_samples
            if max_start <= 0:
                # Track too short, just repeat
                crops = waveforms[:, : self.crop_samples].unsqueeze(1).expand(-1, n_crops, -1)
            else:
                starts = torch.randint(0, max_start, (batch_size, n_crops))
                crops = torch.stack([
                    torch.stack([w[s : s + self.crop_samples] for s in starts[i]])
                    for i, w in enumerate(waveforms)
                ])
        else:
            # Evenly-spaced crops
            stride = max(1, (total_samples - self.crop_samples) // max(1, n_crops - 1))
            starts = [i * stride for i in range(n_crops)]
            starts = [min(s, total_samples - self.crop_samples) for s in starts]
            crops = torch.stack([
                torch.stack([w[s : s + self.crop_samples] for s in starts])
                for w in waveforms
            ])

        # (batch, n_crops, crop_samples) -> (batch * n_crops, crop_samples)
        return crops.reshape(-1, self.crop_samples), n_crops

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: (batch, samples) raw 24kHz audio (full 30s tracks)

        Returns:
            logits: (batch, num_classes) — averaged over crops
        """
        batch_size = waveforms.shape[0]
        crops, n_crops = self._extract_crops(waveforms)

        # Forward all crops through the model
        all_logits = self.model(crops)  # (batch * n_crops, num_classes)

        # Average logits across crops for each track
        all_logits = all_logits.reshape(batch_size, n_crops, -1)
        return all_logits.mean(dim=1)  # (batch, num_classes)
