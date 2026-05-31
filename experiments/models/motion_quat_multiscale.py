"""CN-XXL with quaternion transport inside the multi-scale block.

The earlier QSC branch was residual after multi_scale, so the trained
scale filters could ignore it. This variant puts the quaternion operation
inside scale aggregation itself:

  fine scale feature --unit quaternion transport--> coarse scale feature

Each 32-channel scale feature is treated as 8 quaternion channels. For each
adjacent scale pair, a small conv predicts a unit quaternion per channel group
from the source/target pair. Hamilton transport is injected into the target
scale before the existing output projection. An optional small self-supervised
loss asks the transported source to align with the target scale, anchoring the
quaternion to actual multi-scale feature flow instead of a decorative aux head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.motion import MultiScaleFeatureProcessor
from models.motion_cleanest_quat_head import MotionCleanestLinXLQuatHead


def _as_quat(x):
    b, c, t, n = x.shape
    if c % 4 != 0:
        raise ValueError(f"channel count must be divisible by 4, got {c}")
    return x.reshape(b, c // 4, 4, t, n)


def _from_quat(x):
    b, g, _q, t, n = x.shape
    return x.reshape(b, g * 4, t, n)


def _hamilton(a, b):
    ar, ai, aj, ak = a.unbind(dim=2)
    br, bi, bj, bk = b.unbind(dim=2)
    return torch.stack(
        (
            ar * br - ai * bi - aj * bj - ak * bk,
            ar * bi + ai * br + aj * bk - ak * bj,
            ar * bj - ai * bk + aj * br + ak * bi,
            ar * bk + ai * bj - aj * bi + ak * br,
        ),
        dim=2,
    )


class QuaternionScaleTransport(nn.Module):
    """Adjacent-scale quaternion transport for 32-channel scale features."""

    def __init__(self, feature_dim=32, dropout=0.05):
        super().__init__()
        if feature_dim % 4 != 0:
            raise ValueError("feature_dim must be divisible by 4")
        self.feature_dim = int(feature_dim)
        self.q_pred = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        self.delta_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )
        # Starts close to baseline but not dead; sigmoid(-4.6) ~= 0.01.
        self.logit_gate = nn.Parameter(torch.tensor(-4.6))

    def forward(self, source, target):
        q = _as_quat(self.q_pred(torch.cat([source, target], dim=1)))
        q = F.normalize(q, dim=2, eps=1e-6)
        transported = _from_quat(_hamilton(q, _as_quat(source)))
        delta = self.delta_proj(transported - source)
        out = target + torch.sigmoid(self.logit_gate) * delta

        src_n = F.normalize(transported.flatten(2), dim=2, eps=1e-6)
        tgt_n = F.normalize(target.detach().flatten(2), dim=2, eps=1e-6)
        align_loss = 1.0 - (src_n * tgt_n).sum(dim=2).mean()
        return out, align_loss


class QuaternionMultiScaleFeatureProcessor(MultiScaleFeatureProcessor):
    """Drop-in replacement that keeps base multi_scale keys loadable."""

    def __init__(self, in_channels, num_scales=4, feature_dim=32,
                 qms_dropout=0.05, qms_aux_weight=0.0):
        super().__init__(in_channels, num_scales=num_scales, feature_dim=feature_dim)
        self.q_transport = nn.ModuleList([
            QuaternionScaleTransport(feature_dim=feature_dim, dropout=qms_dropout)
            for _ in range(num_scales - 1)
        ])
        self.qms_aux_weight = float(qms_aux_weight)
        self.aux_loss = None

    def forward(self, x):
        b, _c, t, n = x.shape
        scale_features = [scale_filter(x) for scale_filter in self.scale_filters]

        interacted_features = [scale_features[0]]
        losses = []
        for i in range(len(scale_features) - 1):
            source = F.interpolate(
                interacted_features[-1],
                size=(scale_features[i + 1].shape[2], n),
                mode="bilinear",
                align_corners=False,
            )
            target = scale_features[i + 1]
            combined = torch.cat([source, target], dim=1)
            interaction = self.scale_interaction[i](combined)
            target = target + interaction
            target, align_loss = self.q_transport[i](source, target)
            losses.append(align_loss)
            interacted_features.append(target)

        all_features = [
            F.interpolate(feat, size=(t, n), mode="bilinear", align_corners=False)
            for feat in interacted_features
        ]
        combined_features = torch.cat(all_features, dim=1)
        output = self.output_proj(torch.cat([x, combined_features], dim=1))

        if losses and self.qms_aux_weight > 0:
            self.aux_loss = self.qms_aux_weight * torch.stack(losses).mean()
        else:
            self.aux_loss = output.new_zeros(())
        return output + x


class MotionQuatMultiScaleHead(MotionCleanestLinXLQuatHead):
    def __init__(
        self,
        *args,
        qms_dropout=0.05,
        qms_aux_weight=0.0,
        qms_train_scope="all",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.multi_scale = QuaternionMultiScaleFeatureProcessor(
            in_channels=(self.coord_channels + 64) * 2 - self.coord_channels,
            num_scales=kwargs.get("multi_scale_num_scales", 5),
            feature_dim=32,
            qms_dropout=qms_dropout,
            qms_aux_weight=qms_aux_weight,
        )
        self.qms_aux_weight = float(qms_aux_weight)
        self.qms_train_scope = qms_train_scope
        self.aux_loss = None
        self._apply_qms_train_scope(qms_train_scope)

    def _apply_qms_train_scope(self, scope):
        if scope == "all":
            return
        if scope == "qms":
            prefixes = ("multi_scale.q_transport.",)
        elif scope == "qms_ms":
            prefixes = ("multi_scale.",)
        elif scope == "qms_head":
            prefixes = (
                "multi_scale.q_transport.",
                "stage3.",
                "pool3.",
                "mamba.",
                "stage5.",
                "stage6.",
                "global_bn.",
            )
        elif scope == "qms_ms_head":
            prefixes = (
                "multi_scale.",
                "stage3.",
                "pool3.",
                "mamba.",
                "stage5.",
                "stage6.",
                "global_bn.",
            )
        else:
            raise ValueError(f"unknown qms_train_scope: {scope}")

        for name, param in self.named_parameters():
            param.requires_grad_(name.startswith(prefixes))

    def no_decay_param_names(self):
        return {
            "quat_head_scale",
            *{
                f"multi_scale.q_transport.{i}.logit_gate"
                for i in range(len(self.multi_scale.q_transport))
            },
        }

    def forward(self, inputs):
        logits = super().forward(inputs)
        aux = getattr(self.multi_scale, "aux_loss", None)
        self.aux_loss = aux if aux is not None else logits.new_zeros(())
        return logits


class RealScaleTransport(nn.Module):
    """Param-matched non-quaternion control for adjacent scale correction."""

    def __init__(self, feature_dim=32, dropout=0.05):
        super().__init__()
        self.delta_pred = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        self.delta_proj = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )
        self.logit_gate = nn.Parameter(torch.tensor(-4.6))

    def forward(self, source, target):
        delta = self.delta_proj(self.delta_pred(torch.cat([source, target], dim=1)))
        out = target + torch.sigmoid(self.logit_gate) * delta
        return out, target.new_zeros(())


class RealMultiScaleFeatureProcessor(QuaternionMultiScaleFeatureProcessor):
    """Same integration point as QMS, but no quaternion grouping/product."""

    def __init__(self, in_channels, num_scales=4, feature_dim=32,
                 qms_dropout=0.05, qms_aux_weight=0.0):
        super().__init__(
            in_channels,
            num_scales=num_scales,
            feature_dim=feature_dim,
            qms_dropout=qms_dropout,
            qms_aux_weight=qms_aux_weight,
        )
        self.q_transport = nn.ModuleList([
            RealScaleTransport(feature_dim=feature_dim, dropout=qms_dropout)
            for _ in range(num_scales - 1)
        ])


class MotionRealMultiScaleHead(MotionQuatMultiScaleHead):
    """Param-matched real-valued control for MotionQuatMultiScaleHead."""

    def __init__(self, *args, qms_dropout=0.05, qms_aux_weight=0.0,
                 qms_train_scope="all", **kwargs):
        super().__init__(
            *args,
            qms_dropout=qms_dropout,
            qms_aux_weight=qms_aux_weight,
            qms_train_scope="all",
            **kwargs,
        )
        self.multi_scale = RealMultiScaleFeatureProcessor(
            in_channels=(self.coord_channels + 64) * 2 - self.coord_channels,
            num_scales=kwargs.get("multi_scale_num_scales", 5),
            feature_dim=32,
            qms_dropout=qms_dropout,
            qms_aux_weight=qms_aux_weight,
        )
        self.qms_train_scope = qms_train_scope
        self._apply_qms_train_scope(qms_train_scope)
