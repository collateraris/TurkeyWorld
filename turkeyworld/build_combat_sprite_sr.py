"""Package trained scene and gun weights with the embedded seven-pose enemy atlas."""
import argparse
from pathlib import Path
import torch

from .distill_renderer import _save
from .models import build_upscaler, load_sr_checkpoint
from .small_enemy_data import digest
from .training_pipeline import atomic_json


def build(parent_path, out):
    if out.exists():
        raise FileExistsError(out)
    _, parent = load_sr_checkpoint(parent_path, "cpu")
    if parent["config"].get("combat_appearance") is not True:
        raise ValueError("Expected a trained combat SR with a muzzle phase classifier")
    sprite = parent["config"]["sprite_config"]
    config = {**parent["config"], "source_sprite_compositor": True}
    del config["combat_appearance"]
    config["sprite_config"] = {
        k: sprite[k]
        for k in (
            "atlas_frame_sizes",
            "atlas_buffer_shape",
            "atlas_pixels_sha256",
            "atlas_asset_sha256",
        )
    }
    weights = {}
    for name, value in parent["model"].items():
        if name.startswith("sprite_renderer.") and name not in (
            "sprite_renderer.atlas_frames_rgba",
            "sprite_renderer.atlas_native_sizes",
        ):
            continue
        if name.startswith("combat_appearance.decoder."):
            continue
        weights[name.replace("combat_appearance.weapon_phase.", "combat_weapon_phase.")] = value
    model = build_upscaler(config)
    model.load_state_dict(weights, strict=True)
    provenance = dict(
        parent_sha256=digest(parent_path),
        training_steps=parent["training_steps"],
        atlas_pose_count=7,
        new_gradient_updates=0,
        raster="own_state_native_texel_map_dda_weapon_last",
        atlas_pixels_sha256=config["sprite_config"]["atlas_pixels_sha256"],
    )
    payload = dict(
        stage="sr",
        config=config,
        model=weights,
        training_steps=parent["training_steps"],
        source_sprite_migration=provenance,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    _save(out, payload)
    loaded, _ = load_sr_checkpoint(out, "cpu")
    if not loaded.config["source_sprite_compositor"]:
        raise RuntimeError("SR did not reload its seven-pose compositor")
    atomic_json(out.parent / "migration.json", {**provenance, "sr_sha256": digest(out)})
    return provenance


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    print(build(args.parent, args.out))
