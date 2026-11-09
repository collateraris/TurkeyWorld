"""Stateful PatchGAN updates isolated from the final-image generator graph."""
from __future__ import annotations

import math

import torch

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
        self._pending_microsteps = 0
        self._pending_accumulation_steps = None

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.discriminator.parameters())

    def generator_loss(self, predicted, real, condition, state, *, accumulation_steps=1, microstep=0):
        """Train D on detached endpoints; return ramped hinge + feature matching.

        R1 is evaluated every r1_interval D updates, with gamma/2 times that
        interval compensating the lazy schedule. Source/condition/state never
        receive gradients. A warmup update still trains D but returns a zero
        generator term with a valid input graph. The ramp saturates at 1.

        microstep is zero-based and sequential within each accumulation group.
        D loss is divided by accumulation_steps here; the caller divides its
        complete G loss. D weights and updates change only at the last call.
        Lazy R1 eligibility and the G ramp stay fixed for the whole group.
        """
        if (not isinstance(accumulation_steps, int) or isinstance(accumulation_steps, bool)
                or not isinstance(microstep, int) or isinstance(microstep, bool)
                or accumulation_steps < 1 or not 0 <= microstep < accumulation_steps):
            raise ValueError("Adversarial accumulation requires positive steps and a valid zero-based microstep")
        if microstep == 0:
            if self._pending_accumulation_steps is not None:
                raise ValueError("Cannot start another discriminator group while accumulation is pending")
            self._pending_accumulation_steps = accumulation_steps
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
        lazy_r1 = config["r1_gamma"] > 0 and step % config["r1_interval"] == 0
        if lazy_r1:
            source = source.requires_grad_(True)
        # Disabling the outer generator autocast also keeps lazy R1's forward,
        # input gradients and squared norm in FP32.
        with torch.autocast(self.device.type, enabled=False):
            real_output = discriminator(source, base, labels)
            fake_output = discriminator(predicted.detach().to(device=self.device, dtype=torch.float32), base, labels)
            hinge = discriminator_hinge_loss(real_output, fake_output)
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
            raise FloatingPointError("Non-finite discriminator loss; adversarial update skipped")
        (discriminator_loss / accumulation_steps).backward()
        self._pending_microsteps = microstep + 1
        optimizer_updated = self._pending_microsteps == accumulation_steps
        if optimizer_updated:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.updates = step
            self._pending_microsteps = 0
            self._pending_accumulation_steps = None
        discriminator.eval().requires_grad_(False)
        with torch.autocast(self.device.type, enabled=False):
            fake_generator = discriminator(predicted.to(device=self.device, dtype=torch.float32), base, labels)
            with torch.no_grad():
                real_features = discriminator(source.detach(), base, labels)
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
                   "accumulation_microstep": microstep, "accumulation_steps": accumulation_steps,
                   "enabled": bool(ramp > 0 and (config["weight"] > 0 or config["feature_weight"] > 0))}
        return weighted, metrics

    def state_dict(self):
        if (self._pending_accumulation_steps is not None or self._pending_microsteps
                or any(parameter.grad is not None for parameter in self.discriminator.parameters())):
            raise RuntimeError("Adversarial checkpoints require a completed discriminator accumulation group")
        return {"format_version": 1, "config": dict(self.config), "updates": self.updates,
                "discriminator": self.discriminator.state_dict(),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state):
        if state.get("format_version") != 1 or state.get("config") != self.config:
            raise ValueError("Adversarial checkpoint configuration mismatch")
        updates = int(state["updates"])
        if updates < 0:
            raise ValueError("Adversarial update count must be nonnegative")
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.optimizer.zero_grad(set_to_none=True)
        self.discriminator.eval().requires_grad_(False)
        self.updates = updates
        self._pending_microsteps = 0
        self._pending_accumulation_steps = None
