"""Stateful PatchGAN updates isolated from the final-image generator graph."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .adversarial import (MultiScaleDiscriminator, discriminator_hinge_loss,
                          feature_matching_loss, generator_hinge_loss, r1_penalty)


class AdversarialTrainer:
    """Accumulate one D update, returning a differentiable G loss per microbatch.

    The caller supplies actual final generated RGB, source RGB, the SAME base
    render condition and numerical state. D learns from detached copies in
    FP32. D weights are frozen and spectral iterations disabled for every G
    branch. The caller must backpropagate that branch before the next call;
    at the group's last microbatch D steps before constructing its G branch.
    """

    def __init__(self, device, weight=.02, feature_weight=.05, warmup_steps=64,
                 ramp_steps=256, r1_gamma=1., r1_interval=16, lr=1e-4,
                 base_channels=32):
        values = (weight, feature_weight, r1_gamma, lr)
        if not all(math.isfinite(value) for value in values) or min(weight, feature_weight, r1_gamma) < 0 or lr <= 0:
            raise ValueError("Adversarial weights/gamma must be finite and nonnegative; lr must be positive")
        if warmup_steps < 0 or min(ramp_steps, r1_interval, base_channels) < 1:
            raise ValueError("Warmup must be nonnegative; ramp/interval/channels must be positive")
        self.device = torch.device(device)
        self.config = {"weight": float(weight), "feature_weight": float(feature_weight),
                       "warmup_steps": int(warmup_steps), "ramp_steps": int(ramp_steps),
                       "r1_gamma": float(r1_gamma), "r1_interval": int(r1_interval),
                       "lr": float(lr), "base_channels": int(base_channels)}
        self.discriminator = MultiScaleDiscriminator(base_channels=base_channels).to(self.device).float()
        self.optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(0., .99))
        self.updates = 0
        self.counterfactual_updates = 0
        self._pending_microsteps = 0
        self._pending_accumulation_steps = None
        self._pending_counterfactual_valid = False

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.discriminator.parameters())

    def generator_loss(self, predicted, real, condition, state, *, accumulation_steps=1, microstep=0,
                       counterfactual=None, counterfactual_mask=None, counterfactual_weight=1.,
                       counterfactual_min_fraction=.005):
        """Train D on detached endpoints; return ramped hinge + feature matching.

        R1 is evaluated every r1_interval D updates, with gamma/2 times that
        interval compensating the lazy schedule. Source/condition/state never
        receive gradients. A warmup update still trains D but returns a zero
        generator term with a valid input graph. The ramp saturates at 1.

        microstep is zero-based and sequential within each accumulation group.
        D loss is divided by accumulation_steps here; the caller divides its
        complete G loss. D weights and updates change only at the last call.
        Lazy R1 eligibility and the G ramp stay fixed for the whole group.

        An optional offline Source-without-NPC image is an additional D-only
        negative under the SAME base/state condition. It is used only where an
        actual visible-NPC mask is nonempty and covers the minimum fraction.
        Per eligible image, generated/removed negative hinges are normalized
        by1+counterfactual_weight; valid natural-wall Source images are never
        relabeled negative. Counterfactual labels never receive G gradients.
        """
        if (not isinstance(accumulation_steps, int) or isinstance(accumulation_steps, bool)
                or not isinstance(microstep, int) or isinstance(microstep, bool)
                or accumulation_steps < 1 or not 0 <= microstep < accumulation_steps):
            raise ValueError("Adversarial accumulation requires positive steps and a valid zero-based microstep")
        if (not isinstance(counterfactual_weight, (int, float)) or not math.isfinite(counterfactual_weight)
                or counterfactual_weight < 0 or not isinstance(counterfactual_min_fraction, (int, float))
                or not math.isfinite(counterfactual_min_fraction) or not 0 <= counterfactual_min_fraction <= 1):
            raise ValueError('Counterfactual weight/fraction must be finite, with weight>=0 and fraction in[0,1]')
        cf_source = None
        cf_valid = torch.zeros(real.shape[0], device=self.device, dtype=torch.bool)
        if counterfactual is not None and counterfactual_mask is not None and counterfactual_weight > 0:
            if not isinstance(counterfactual, torch.Tensor) or counterfactual.shape != real.shape:
                raise ValueError('Counterfactual RGB must have the same shape as Source RGB')
            if not isinstance(counterfactual_mask, torch.Tensor):
                raise ValueError('Counterfactual NPC mask must be a tensor')
            mask = counterfactual_mask.detach().to(device=self.device, dtype=torch.float32)
            if mask.ndim == 2 and real.shape[0] == 1:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask[:, None]
            if mask.shape != (real.shape[0], 1, *real.shape[-2:]):
                raise ValueError('Counterfactual NPC mask must have shape[B,1,H,W]')
            if not bool(torch.isfinite(mask).all()) or bool(((mask < 0) | (mask > 1)).any()):
                raise ValueError('Counterfactual NPC mask must be finite and in[0,1]')
            fraction = mask.mean((1, 2, 3))
            cf_valid = (fraction > 0) & (fraction >= counterfactual_min_fraction)
            if bool(cf_valid.any()):
                cf_source = counterfactual.detach().to(device=self.device, dtype=torch.float32)[cf_valid]
        if microstep == 0:
            if self._pending_accumulation_steps is not None:
                raise ValueError("Cannot start another discriminator group while accumulation is pending")
            self._pending_accumulation_steps = accumulation_steps
            self._pending_counterfactual_valid = False
            self.optimizer.zero_grad(set_to_none=True)
        elif (self._pending_accumulation_steps != accumulation_steps
              or self._pending_microsteps != microstep):
            raise ValueError("Adversarial accumulation microsteps must be sequential with an unchanged group size")
        config = self.config
        step = self.updates + 1
        discriminator = self.discriminator
        discriminator.train().requires_grad_(True)
        source = real.detach().to(device=self.device, dtype=torch.float32)
        base = condition.detach().to(device=self.device, dtype=torch.float32)
        labels = None if state is None else state.detach().to(device=self.device, dtype=torch.float32)
        cf_labels = None if labels is None else labels[cf_valid]
        cf_base = base[cf_valid]
        lazy_r1 = config["r1_gamma"] > 0 and step % config["r1_interval"] == 0
        if lazy_r1:
            source = source.requires_grad_(True)
        # Disabling the outer generator autocast also keeps lazy R1's forward,
        # input gradients and squared norm in FP32.
        with torch.autocast(self.device.type, enabled=False):
            real_output = discriminator(source, base, labels)
            fake_output = discriminator(predicted.detach().to(device=self.device, dtype=torch.float32), base, labels)
            hinge = discriminator_hinge_loss(real_output, fake_output)
            cf_hinge = hinge.new_zeros(())
            if cf_source is not None:
                cf_output = discriminator(cf_source, cf_base, cf_labels)
                # Normalize negatives independently for every image. Otherwise
                # adding a hard-negative stream would silently double D loss or
                # downweight the natural-wall examples in a mixed batch.
                real_per_image = torch.stack([F.relu(1-value.float()).flatten(1).mean(1)
                                              for value in real_output['logits']]).mean(0)
                fake_per_image = torch.stack([F.relu(1+value.float()).flatten(1).mean(1)
                                              for value in fake_output['logits']]).mean(0)
                cf_per_image = torch.stack([F.relu(1+value.float()).flatten(1).mean(1)
                                            for value in cf_output['logits']]).mean(0)
                indices = cf_valid.nonzero(as_tuple=False).flatten()
                negatives = fake_per_image.index_copy(0, indices,
                    (fake_per_image[cf_valid]+counterfactual_weight*cf_per_image)/(1+counterfactual_weight))
                hinge = real_per_image.mean()+negatives.mean()
                cf_hinge = cf_per_image.mean()
            regularization = r1_penalty(real_output["logits"], source) if lazy_r1 else hinge.new_zeros(())
            discriminator_loss = hinge + .5 * config["r1_gamma"] * config["r1_interval"] * regularization
        if not bool(torch.isfinite(discriminator_loss)):
            self.optimizer.zero_grad(set_to_none=True)
            discriminator.eval().requires_grad_(False)
            if accumulation_steps == 1:
                # Preserve the original single-update retry behavior. A failed
                # multi-micro group instead requires restoring a safe boundary.
                self._pending_microsteps = 0
                self._pending_accumulation_steps = None
                self._pending_counterfactual_valid = False
            raise FloatingPointError("Non-finite discriminator loss; adversarial update skipped")
        (discriminator_loss / accumulation_steps).backward()
        self._pending_counterfactual_valid |= cf_source is not None
        self._pending_microsteps = microstep + 1
        optimizer_updated = self._pending_microsteps == accumulation_steps
        if optimizer_updated:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.updates = step
            if self._pending_counterfactual_valid:
                self.counterfactual_updates += 1
            self._pending_microsteps = 0
            self._pending_accumulation_steps = None
            self._pending_counterfactual_valid = False
        discriminator.eval().requires_grad_(False)
        with torch.autocast(self.device.type, enabled=False):
            fake_generator = discriminator(predicted.to(device=self.device, dtype=torch.float32), base, labels)
            with torch.no_grad():
                real_features = discriminator(source.detach(), base, labels)
                cf_score = source.new_zeros(())
                presence_gap = source.new_zeros(())
                if cf_source is not None:
                    # All presence metrics use one fixed eval-mode D snapshot,
                    # including after the group's optimizer step. Spectral
                    # power iterations cannot change between real/removed.
                    cf_evaluation = discriminator(cf_source, cf_base, cf_labels)
                    cf_score = torch.stack([value.float().mean() for value in cf_evaluation['logits']]).mean()
                    real_score_same = torch.stack([value[cf_valid].float().mean()
                                                   for value in real_features['logits']]).mean()
                    presence_gap = real_score_same-cf_score
            adversarial = generator_hinge_loss(fake_generator)
            matching = feature_matching_loss(real_features, fake_generator)
            ramp = min(1., max(0., (step - config["warmup_steps"]) / config["ramp_steps"]))
            weighted = ramp * (config["weight"] * adversarial + config["feature_weight"] * matching)
        metrics = {"discriminator_loss": float(discriminator_loss.detach()),
                   "generator_loss": float(adversarial.detach()),
                   "feature_matching": float(matching.detach()),
                   "r1": float(regularization.detach()),
                   "real_score": float(torch.stack([value.detach().float().mean() for value in real_output["logits"]]).mean()),
                   "fake_score": float(torch.stack([value.detach().float().mean() for value in fake_output["logits"]]).mean()),
                   "weight": config["weight"] * ramp,
                   "feature_weight": config["feature_weight"] * ramp, "ramp": ramp,
                   "step": step, "warmup_steps": config["warmup_steps"],
                   "ramp_steps": config["ramp_steps"],
                   "optimizer_updated": optimizer_updated, "completed_updates": self.updates,
                   'cf_valid_fraction': float(cf_valid.float().mean()),
                   'cf_fake_score': float(cf_score), 'cf_discriminator_loss': float(cf_hinge.detach()),
                   'cf_completed_updates': self.counterfactual_updates,
                   'presence_score_gap': float(presence_gap),
                   "accumulation_microstep": microstep, "accumulation_steps": accumulation_steps,
                   "enabled": bool(ramp > 0 and (config["weight"] > 0 or config["feature_weight"] > 0))}
        return weighted, metrics

    def state_dict(self):
        if (self._pending_accumulation_steps is not None or self._pending_microsteps or self._pending_counterfactual_valid
                or any(parameter.grad is not None for parameter in self.discriminator.parameters())):
            raise RuntimeError("Adversarial checkpoints require a completed discriminator accumulation group")
        return {"format_version": 1, "config": dict(self.config), "updates": self.updates,
                'counterfactual_updates': self.counterfactual_updates,
                "discriminator": self.discriminator.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state.get("config") != self.config:
            raise ValueError("Adversarial checkpoint configuration mismatch")
        updates = int(state["updates"])
        if updates < 0:
            raise ValueError("Adversarial update count must be nonnegative")
        cf_updates = state.get('counterfactual_updates', 0)
        if not isinstance(cf_updates, int) or not 0 <= cf_updates <= updates:
            raise ValueError('Counterfactual update count must be between0 and discriminator updates')
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)
        self.discriminator.eval().requires_grad_(False)
        self.updates = updates
        self.counterfactual_updates = cf_updates
        self._pending_microsteps = 0
        self._pending_accumulation_steps = None
        self._pending_counterfactual_valid = False
