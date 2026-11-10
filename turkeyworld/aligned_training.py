"""Separate autoregressive predictions from pose-aligned supervised queries.

The main stream receives initial context, then only its own state/memory and
the recorded controls. Its trajectory and latent targets are supervised.
Combat, one-tick motion, spawn and image labels instead belong to a separate
GT-observation/state stream. They never label a classifier at a drifted pose.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .jepa_training_utils import latent_prediction_loss, variance_covariance_loss
from .train import image_loss, state_loss, supervised_event_losses


def spawn_supervision_statistics(result, previous, target):
    """Raw event sums for v5 global BxH hierarchical spawn supervision.

    Anchor CE is unmasked by own earlier simultaneous choices. Fallback CE is
    conditional on a non-anchor Source label and excludes the eight anchors;
    it is not another copy of the effective256-way hierarchical loss.
    """
    from .spawn import SPAWN_ANCHORS
    old = previous.detach().float()[:, 8:88].reshape(-1, 8, 10)
    expected = target.detach().float()[:, 8:88].reshape_as(old)
    revived = ~((old[..., 5] >= .5) & (old[..., 4] > 0)) & (expected[..., 5] >= .5) & (expected[..., 4] > 0)
    zero = result['state'].sum()*0
    if 'spawn_logits' not in result:
        return {'spawn_sum': zero, 'spawn_count': zero.detach(),
                'spawn_anchor_sum': zero, 'spawn_fallback_sum': zero, 'spawn_tile_sum': zero}
    tiles = (expected[..., :2]*16).long().clamp(0, 15)
    labels = tiles[..., 0]+16*tiles[..., 1]
    count = revived.to(torch.float32).sum()
    anchor_sum = fallback_sum = tile_sum = result['spawn_logits'].float().sum()*0
    if 'spawn_anchor_logits' in result:
        anchor_logits = result['spawn_anchor_logits'].float()
        if anchor_logits.shape != (*labels.shape, 9):
            raise ValueError('Expected spawn_anchor_logits[B,8,9]')
        anchor_tiles = labels.new_tensor([x+16*y for x, y in SPAWN_ANCHORS])
        matches = labels[..., None] == anchor_tiles
        anchor_labels = torch.where(matches.any(-1), matches.long().argmax(-1), torch.full_like(labels, 8))
        errors = F.cross_entropy(anchor_logits.reshape(-1, 9), anchor_labels.reshape(-1), reduction='none').reshape_as(labels)
        anchor_sum = (errors*revived).sum()
        fallback = revived & (anchor_labels == 8)
        if bool(fallback.any()):
            if 'spawn_fallback_logits' not in result:
                raise ValueError('Fallback Source spawns require raw spawn_fallback_logits')
            raw = result['spawn_fallback_logits'].float()
            if raw.shape != (*labels.shape, 256):
                raise ValueError('Expected spawn_fallback_logits[B,8,256]')
            forbidden = torch.zeros(256, dtype=torch.bool, device=raw.device)
            forbidden[anchor_tiles] = True
            fallback_sum = F.cross_entropy(raw[fallback].masked_fill(forbidden[None], -torch.inf), labels[fallback], reduction='sum')
        total = anchor_sum+fallback_sum
    else:
        errors = F.cross_entropy(result['spawn_logits'].float().reshape(-1, 256), labels.reshape(-1), reduction='none').reshape_as(labels)
        tile_sum = (errors*revived).sum()
        total = tile_sum
    return {'spawn_sum': total, 'spawn_count': count, 'spawn_anchor_sum': anchor_sum,
            'spawn_fallback_sum': fallback_sum, 'spawn_tile_sum': tile_sum}


def aggregate_spawn_supervision(statistics):
    sums = {name: torch.stack([value[name] for value in statistics]).sum()
            for name in ('spawn_sum', 'spawn_count', 'spawn_anchor_sum', 'spawn_fallback_sum', 'spawn_tile_sum')}
    denominator = sums['spawn_count'].clamp_min(1)
    return sums['spawn_sum']/denominator, {'spawn_event_count': sums['spawn_count'],
        'spawn_anchor_ce': sums['spawn_anchor_sum']/denominator,
        'spawn_fallback_ce': sums['spawn_fallback_sum']/denominator,
        'spawn_tile_ce': sums['spawn_tile_sum']/denominator}


@torch.no_grad()
def npc_motion_diagnostics(result, previous, target, *, simulation_hz=30.):
    """Measure masked proposal saturation without adding it to any loss."""
    proposal = result.get('proposed_npc_displacement')
    if proposal is None:
        return {}
    old = previous.detach().float()[:, 8:88].reshape(-1, 8, 10)
    expected = target.detach().float()[:, 8:88].reshape_as(old)
    continuing = (old[:, :, 5] > .5) & (expected[:, :, 5] > .5)
    saturated = proposal.detach().float().norm(dim=-1) > (1.2/simulation_hz/16)*(1+1e-5)
    stopped = (expected[:, :, :2]-old[:, :, :2]).norm(dim=-1) < 1e-7
    denominator = continuing.sum().clamp_min(1)
    return {'npc_saturated_fraction': (saturated & continuing).sum()/denominator,
        'npc_stopped_target_fraction': (stopped & continuing).sum()/denominator,
        'npc_stopped_saturated_fraction': (stopped & saturated & continuing).sum()/denominator}


def motion_error_parts(prediction, previous, target, *, simulation_hz=30., local_motion=False,
                       action=None):
    """Separate scaled player/NPC motion errors, excluding respawn teleports."""
    dt = 1 / simulation_hz
    old = previous[:, 8:88].reshape(-1, 8, 10)
    expected = target[:, 8:88].reshape_as(old)
    predicted = prediction[:, 8:88].reshape_as(old)
    continuing = (old[:, :, 5] > .5) & (expected[:, :, 5] > .5)
    first = torch.cat([(prediction[:, :2]-previous[:, :2])[:, None],
                       predicted[:, :, :2]-old[:, :, :2]], 1)
    second = torch.cat([(target[:, :2]-previous[:, :2])[:, None],
                        expected[:, :, :2]-old[:, :, :2]], 1)
    if local_motion:
        from .spatial_entities import entity_motion_basis, world_to_local
        basis = entity_motion_basis(previous, action=action, dt=dt)
        first, second = world_to_local(first, basis), world_to_local(second, basis)
    player = F.smooth_l1_loss(first[:, 0]*(16/(2.4*dt)), second[:, 0]*(16/(2.4*dt)))
    movement = F.smooth_l1_loss(first[:, 1:]*(16/(1.2*dt)), second[:, 1:]*(16/(1.2*dt)), reduction='none')
    npc = (movement*continuing[:, :, None]).sum()/(2*continuing.sum()).clamp_min(1)
    return player, npc


def aligned_auxiliary_step_loss(result, previous, target, hit_ids, attack_flags, events,
                                *, simulation_hz=30., none_class_weight=.2, attack_pos_weight=4.,
                                local_motion=False, action=None, spatial_combat=False, game_training_v5=False):
    """Label a transition computed at exactly GT previous state/observations.

    Callers must keep this query separate from their persistent AR stream.
    This helper is also suitable for an independent long-training auxiliary
    stream; its returned tensors do not replace any autoregressive state.
    """
    previous, target = previous.detach().float(), target.detach().float()
    if spatial_combat:
        from .combat_objectives import combat_transition_losses
        losses = combat_transition_losses(result, previous, target, hit_ids.detach(), attack_flags.detach(),
            events.detach(), action=action, dt=1/simulation_hz, none_class_weight=none_class_weight,
            attack_pos_weight=attack_pos_weight, game_training_v5=game_training_v5)
    else:
        losses = supervised_event_losses(result, hit_ids.detach(), attack_flags.detach(), events.detach(),
                                        none_class_weight=none_class_weight, attack_pos_weight=attack_pos_weight)
    losses["state"] = state_loss(result["state"], target)
    old = previous[:, 8:88].reshape(-1, 8, 10)
    expected = target[:, 8:88].reshape_as(old)
    player, npc = motion_error_parts(result['state'], previous, target, simulation_hz=simulation_hz,
                                    local_motion=local_motion, action=action)
    losses["motion"] = player+npc
    losses["spawn"] = result["state"].new_zeros(())
    if game_training_v5:
        statistics = spawn_supervision_statistics(result, previous, target)
        losses.update(statistics)
        losses['spawn'] = statistics['spawn_sum']/statistics['spawn_count'].clamp_min(1)
    elif "spawn_logits" in result:
        revived = (old[:, :, 5] < .5) & (expected[:, :, 5] > .5)
        tiles = (expected[:, :, :2] * 16).long().clamp(0, 15)
        labels = tiles[:, :, 0] + tiles[:, :, 1] * 16
        values = F.cross_entropy(result["spawn_logits"].float().reshape(-1, 256), labels.reshape(-1),
                                 reduction="none").reshape(-1, 8)
        losses["spawn"] = (values * revived).sum() / revived.sum().clamp_min(1)
    return losses


def _observed_memory(model, observed, states, map_features, end, context):
    keep = getattr(model.core, "context_frames", context)
    start = max(0, end - keep)
    return model.vjepa_backend.initialize_from_latents(observed[:, start:end],
                                                       states[:, start:end], map_features)


def aligned_short_rollout_loss(model, ema, batch, *, context=2, render=True, teacher=None,
                               distill_endpoint=False, render_last_only=True, object_region_boost=1.,
                               detach_flow_base=False):
    """Return loss/metrics and a GT-pose-aligned RGB/GAN endpoint.

    `endpoint['ar_state']`/`ar_features` are the independent main prediction.
    `state`/`base`/`target` are the same-GT-pose image query for FM/GAN losses.
    RGB/FM supervise the last target by default to bound250M training memory;
    render_last_only=False supervises every future target at its matching pose.
    All dynamics/event/spawn auxiliary steps are supervised in either mode.
    """
    actions = batch["actions"].detach()
    spatial_combat = bool(getattr(model.config, 'spatial_combat', False))
    game_training_v5 = bool(getattr(model.config, 'game_training_v5', False))
    horizon = actions.shape[1]
    if context < 1 or horizon < 1 or batch["frames"].shape[1] < context + horizon:
        raise ValueError("A complete positive context and future horizon are required")
    states = batch["states"].detach()
    previous = batch["previous_states"].detach()
    targets = batch["target_states"].detach()
    images = batch["target_frames"].detach()
    object_masks = batch.get('target_object_masks') if object_region_boost > 1 or game_training_v5 else None
    if (object_region_boost > 1 or game_training_v5) and object_masks is None:
        raise ValueError('Object-aware image supervision requires offline target NPC masks')
    pixel_weights = None
    hits = batch["hit_ids"] if "hit_ids" in batch else batch["hits"]
    attacks = batch["attack_flags"] if "attack_flags" in batch else batch["attacks"]
    events = batch["events"]
    map_features = model.encode_map(batch["map"])
    context_latents = model.core.encode_video(batch["context_frames"].detach())
    memory = model.vjepa_backend.initialize_from_latents(context_latents, states[:, :context], map_features)
    current = states[:, context - 1]
    main_latents, main_states = [], []
    trajectory_predictions = []
    # Complete the main forward trajectory BEFORE encoding any future source
    # observations for the auxiliary stream. No auxiliary tensor feeds it.
    for index in range(horizon):
        main = model.transition(current, actions[:, index], map_features, memory)
        main_latents.append(main["latent_prediction"])
        main_states.append(state_loss(main["state"], targets[:, index]))
        if game_training_v5:
            trajectory_predictions.append(main['state'])
        current, memory = main["state"], main["memory"]
    ar_state, ar_features, ar_memory = current, main["features"], memory
    with torch.no_grad():
        target_latents = ema(batch["frames"].detach())[:, context:context + horizon]
    regularizer, regularizer_metrics = variance_covariance_loss(context_latents)
    latent, latent_metrics = latent_prediction_loss(torch.stack(main_latents, 1), target_latents)
    parts = {"regularizer": .1 * regularizer, "state": 10 * torch.stack(main_states).mean(), "jepa": latent}
    trajectory_metrics = {}
    if game_training_v5:
        from .trajectory_objectives import trajectory_state_losses
        trajectory_parts, trajectory_metrics = trajectory_state_losses(torch.stack(trajectory_predictions, 1),
            states[:, context-1], targets, batch['map'])
        parts.update(trajectory_parts)

    teacher_map = teacher_observed = teacher_current = None
    if teacher is not None:
        with torch.no_grad():
            teacher_map = teacher.encode_map(batch["map"])
            teacher_memory = teacher.initialize_from_image(batch["context_frames"].detach(), states[:, :context], teacher_map)
            teacher_current = states[:, context - 1]
            for index in range(horizon):
                teach_ar = teacher.transition(teacher_current, actions[:, index], teacher_map, teacher_memory)
                teacher_current, teacher_memory = teach_ar["state"], teach_ar["memory"]
        parts["teacher_state"] = .5 * state_loss(ar_state, teacher_current)

    # Causal encoder: one shared GT-past pass, excluding the final target RGB.
    observed = model.core.encode_video(batch["frames"][:, :context + horizon - 1].detach())
    if teacher is not None:
        with torch.no_grad():
            teacher_observed = teacher.core.encode_video(batch["frames"][:, :context + horizon - 1].detach())
    auxiliary, auxiliary_latents, base_values, flow_values, distill_values, reflow_values = [], [], [], [], [], []
    flow_metrics = {}
    motion_diagnostics = []
    combat_contexts, combat_features, foreground_values = [], [], []
    for index in range(horizon):
        end = context + index
        auxiliary_memory = _observed_memory(model, observed, states, map_features, end, context)
        result = model.transition(previous[:, index], actions[:, index], map_features, auxiliary_memory)
        auxiliary_latents.append(result["latent_prediction"])
        auxiliary.append(aligned_auxiliary_step_loss(result, previous[:, index], targets[:, index],
            hits[:, index], attacks[:, index], events[:, index], simulation_hz=model.config.simulation_hz,
            local_motion=getattr(model.config, 'spatial_entities', False), action=actions[:, index],
            spatial_combat=spatial_combat, game_training_v5=game_training_v5))
        if spatial_combat:
            combat_contexts.append({name: result['combat_context'][name].detach()
                                    for name in ('state', 'pooled', 'spatial')})
            combat_features.append(result['features'].detach())
        diagnostics = npc_motion_diagnostics(result, previous[:, index], targets[:, index],
                                           simulation_hz=model.config.simulation_hz)
        with torch.no_grad():
            player_motion, npc_motion = motion_error_parts(result['state'], previous[:, index], targets[:, index],
                simulation_hz=model.config.simulation_hz, local_motion=getattr(model.config, 'spatial_entities', False),
                action=actions[:, index])
        diagnostics.update(motion_player=5*player_motion, motion_npc=5*npc_motion)
        motion_diagnostics.append(diagnostics)
        teach = None
        if teacher is not None:
            with torch.no_grad():
                teacher_memory = _observed_memory(teacher, teacher_observed, states, teacher_map, end, context)
                teach = teacher.transition(previous[:, index], actions[:, index], teacher_map, teacher_memory)
            distill_values.append(F.kl_div(F.log_softmax(result["target_logits"], -1),
                F.softmax(teach["target_logits"], -1), reduction="batchmean") +
                F.binary_cross_entropy_with_logits(result["attack_logits"], teach["attack_logits"].sigmoid()))
        if render_last_only and index != horizon - 1:
            continue
        render_state = targets[:, index]
        base = model.render_base(render_state, result["features"], map_features)
        target_rgb = images[:, index]
        if game_training_v5:
            from .object_losses import balanced_image_l1, balanced_edge_l1
            base_values.append(balanced_image_l1(base, target_rgb, object_masks[:, index])
                               +.2*balanced_edge_l1(base, target_rgb, object_masks[:, index]))
            if object_masks is not None:
                from .object_losses import object_region_weights
                pixel_weights = object_region_weights(object_masks[:, index], object_region_boost)
        elif object_masks is not None:
            from .object_losses import object_region_weights, weighted_image_loss
            pixel_weights = object_region_weights(object_masks[:, index], object_region_boost)
            base_values.append(weighted_image_loss(base, target_rgb, pixel_weights))
        else:
            base_values.append(image_loss(base, target_rgb))
        if spatial_combat:
            from .object_losses import foreground_contrastive_loss
            if object_masks is None or 'target_background_frames' not in batch:
                raise ValueError('Combat-aware image training requires exact NPC masks and backgrounds')
            foreground_values.append(.25*foreground_contrastive_loss(base, target_rgb,
                batch['target_background_frames'][:, index], object_masks[:, index]))
        if render:
            options = ({'object_mask': object_masks[:, index]} if game_training_v5
                       else {'pixel_weights': pixel_weights} if pixel_weights is not None else {})
            if detach_flow_base:
                options['detach_base'] = True
            if spatial_combat:
                options['map_condition'] = batch['map'].detach()
            flow, flow_metrics = model.flow_renderer.loss(target_rgb, base, render_state, **options)
            flow_values.append(flow)
        if teacher is not None and distill_endpoint:
            count = 1
            with torch.no_grad():
                noise = torch.randn_like(target_rgb[:count])
                teacher_base = teacher.render_base(render_state[:count], teach["features"][:count], teacher_map[:count])
                teacher_options = {'map_condition': batch['map'][:count].detach()} if getattr(teacher.config, 'spatial_combat', False) else {}
                endpoint = teacher.flow_renderer.sample(teacher_base, render_state[:count], noise=noise, **teacher_options)
            reflow_options = {'detach_base': True} if detach_flow_base else {}
            if game_training_v5:
                reflow_options['object_mask'] = object_masks[:, index][:count]
            if spatial_combat:
                reflow_options['map_condition'] = batch['map'][:count].detach()
            reflow, _ = model.flow_renderer.loss(target_rgb[:count], base[:count], render_state[:count],
                                                noise=noise, target_endpoint=endpoint,
                                                 **reflow_options)
            reflow_values.append(reflow)
    combat_names = ('events', 'target', 'attack', 'damage_npc', 'damage_player', 'death_npc') if spatial_combat else ('events', 'target', 'attack')
    balanced_attack_metrics = {}
    if game_training_v5 and spatial_combat:
        from .combat_objectives import aggregate_balanced_attack
        attack_loss, balanced_attack_metrics = aggregate_balanced_attack(auxiliary)
        combat_parts = {name: torch.stack([value[name] for value in auxiliary]).mean() for name in combat_names}
        combat_parts['attack'] = attack_loss
        parts['combat'] = sum(combat_parts.values())
    else:
        parts["combat"] = torch.stack([sum(value[name] for name in combat_names) for value in auxiliary]).mean()
    parts["motion"] = 5 * torch.stack([value["motion"] for value in auxiliary]).mean()
    spawn_metrics = {}
    if game_training_v5:
        spawn_loss, spawn_metrics = aggregate_spawn_supervision(auxiliary)
        parts['spawn'] = .5*spawn_loss
    else:
        parts["spawn"] = .5 * torch.stack([value["spawn"] for value in auxiliary]).mean()
    parts["aux_state"] = 2 * torch.stack([value["state"] for value in auxiliary]).mean()
    auxiliary_latent, _ = latent_prediction_loss(torch.stack(auxiliary_latents, 1), target_latents)
    parts["aux_jepa"] = .5 * auxiliary_latent
    parts["base_image"] = torch.stack(base_values).mean()
    if foreground_values:
        parts['foreground_base'] = torch.stack(foreground_values).mean()
    if flow_values:
        parts["flow"] = torch.stack(flow_values).mean()
    if distill_values:
        parts["teacher_events"] = .2 * torch.stack(distill_values).mean()
    if reflow_values:
        parts["reflow"] = torch.stack(reflow_values).mean()
    metrics = {name: float(value.detach()) for name, value in parts.items()}
    for name in combat_names:
        metrics['combat_'+name] = (float(combat_parts[name].detach()) if game_training_v5 and spatial_combat
            else float(torch.stack([value[name].detach() for value in auxiliary]).mean()))
    if game_training_v5:
        metrics.update(trajectory_metrics)
        metrics.update({'combat_'+name: float(value.detach()) for name, value in balanced_attack_metrics.items()})
        metrics.update({'global_'+name: float(value.detach()) for name, value in spawn_metrics.items()})
        metrics.update(game_training_v5=True, spawn_normalization='real_respawn_slots_BxH',
                       attack_normalization='present_class_means_positive4_negative1_BxH')
    for name in ('npc_saturated_fraction', 'npc_stopped_target_fraction', 'npc_stopped_saturated_fraction',
                 'motion_player', 'motion_npc'):
        if all(name in value for value in motion_diagnostics):
            metrics[name] = float(torch.stack([value[name] for value in motion_diagnostics]).mean())
    metrics.update({name: float(value) for name, value in {**regularizer_metrics, **latent_metrics, **flow_metrics}.items()})
    metrics.update(supervised_pose_aligned=True, auxiliary_steps=horizon,
                   rgb_supervised_steps=len(base_values), ar_future_gt_feedback=False)
    if detach_flow_base:
        metrics['flow_base_detached'] = True
    if object_masks is not None:
        from .object_losses import region_error_metrics
        metrics.update({name: float(value) for name, value in region_error_metrics(
            base, target_rgb, object_masks[:, -1]).items()})
    endpoint = {"base": base, "state": render_state, "target": target_rgb,
        "features": result["features"], "ar_state": ar_state, "ar_features": ar_features,
        "ar_memory": ar_memory, "pose_aligned": True, 'pixel_weights': pixel_weights,
        'object_mask': object_masks[:, -1] if object_masks is not None else None,
        'background': batch['target_background_frames'][:, -1] if 'target_background_frames' in batch else None,
        'map_condition': batch['map'].detach()}
    if spatial_combat:
        endpoint['combat_payload'] = {
            'contexts': {name: torch.stack([context[name] for context in combat_contexts], 1)
                         for name in combat_contexts[0]},
            'features': torch.stack(combat_features, 1), 'map_features': map_features.detach(),
            'target_states': targets.detach(), 'actions': actions.detach(), 'events': events.detach(),
            'target_frames': images.detach(), 'object_masks': object_masks.detach(),
            'backgrounds': batch['target_background_frames'].detach(), 'walkable': (batch['map'] == 0).detach(),
            'map_grid': batch['map'].detach()}
    return sum(parts.values()), metrics, endpoint
