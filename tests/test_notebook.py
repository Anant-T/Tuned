import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "kaggle_smoke.ipynb"


def test_notebook_is_valid_and_complete():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    sources = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(sources)
    # the operator's switches default to the 8B DDP lane's next gate: PROBE and
    # SAVETEST both ran green 2026-08-08 (peaks ~13.0 GiB @ seq 8192; grad_norm
    # finite at step 3; last-checkpoint/ verified in the hub repo with both
    # ranks' rng states), so the ladder advances to SMOKE (60 steps, ~1.4 h)
    assert 'MODE = "SMOKE"' in joined
    assert "DDP = False" in joined
    assert "MP = False" in joined
    assert "DDP_8B = True" in joined
    # the two multi-GPU lanes are mutually exclusive - torchrun replicating a
    # device_map-split model would be neither DDP nor MP
    assert "DDP and MP" in joined
    # GPU mask and config follow the lane together: DDP needs every rank to see
    # both GPUs (a leaked single-GPU mask killed rank 1 with "invalid device
    # ordinal" on 2026-08-06), MP splits layers across both, and each lane has
    # its own config (grad accumulation, seq length, device_map, ckpt repo)
    assert '"0,1" if (DDP or MP) else "0"' in joined
    assert '"configs/law_v1_mp.yaml" if MP' in joined
    assert '"configs/law_v1_ddp.yaml" if DDP' in joined
    assert '"configs/law_v1.yaml"' in joined
    # MP must NOT go through torchrun - only the DDP-style lanes select it
    assert '["torchrun", "--nproc_per_node=2"] if (DDP or DDP_8B) else [sys.executable, "-u"]' in joined
    # 8B lane wiring: its own config, and the chunked-CE env var must be set
    # lane-scoped BEFORE `import unsloth` in the torchrun children (default
    # n_chunks=4 leaves a ~2.4-3.0 GiB loss-step transient at seq 8192)
    assert '"configs/law_v1_8b_ddp.yaml" if DDP_8B' in joined
    assert 'UNSLOTH_CE_LOSS_N_CHUNKS"] = "16"' in joined
    # the MP lane's gate: PROBE runs no-hub on the long probe dataset (short
    # unpacked examples would make the long-seq VRAM probe a false green);
    # PROBE_SEQ probes above-config lengths without a config edit
    assert '"data/probe_long.jsonl"' in joined
    assert "--no-hub" in joined
    assert "PROBE_SEQ" in joined
    assert "--max-seq-length" in joined
    assert "tuned.data.probe" in joined
    # allocator headroom - DDP peaks ~1 GiB from the 14.56 GiB cap
    assert 'PYTORCH_ALLOC_CONF"] = "expandable_segments:True"' in joined
    # scratch cache - never /kaggle/working
    assert "/tmp/hf_cache" in joined
    assert "/kaggle/working/hf_cache" not in joined
    # secrets come from Kaggle, never hardcoded
    assert "UserSecretsClient" in joined
    # token hunt must be depth-bounded: rglob walks EVERY attached dataset, and
    # the not-found diagnostic must stream lazily (islice), never materialize
    # the whole /kaggle/input tree - a mounted multi-GB dataset turns either
    # into a minutes-long network-FS crawl
    assert 'rglob("token.txt")' not in joined
    assert '"*/token.txt"' in joined
    assert "islice" in joined
    # xet turbo is scoped to the pre-download child only (hub 1.x ignores
    # HF_HUB_ENABLE_HF_TRANSFER; this is the xet-native equivalent), and the
    # 25-min timeout still bounds the phase
    assert 'HF_XET_HIGH_PERFORMANCE": "1"' in joined
    # staged-snapshot fast path: pre-download prefers a mounted snapshot whose
    # REVISION.txt matches the config pin (staleness guard), else falls back
    # to the bounded hub download
    assert "TUNED_MODEL_PATH" in joined
    assert "qwen3-8b-staged/REVISION.txt" in joined
    # progress pushes must never block the watchdog loop (a hung upload would
    # freeze heartbeats AND the timeout check)
    assert "_push_inflight" in joined
    assert "hf_" not in joined.replace("hf_cache", "").replace("hf_transfer", "").replace("HF_", "")
    # hf_transfer must be explicitly disabled: unsloth_zoo force-enables it when
    # the var is absent, and its no-retry fast path can stall downloads silently
    assert 'HF_HUB_ENABLE_HF_TRANSFER"] = "0"' in joined
    # xet must stay ENABLED: disabling it forces the legacy bridge path, which
    # stalls from Kaggle (v6-v8 all hung on downloads with xet off; v1-v5 were fine)
    assert "HF_HUB_DISABLE_XET" not in joined.replace("HF_HUB_DISABLE_XET: v", "")
    assert "UNSLOTH_STABLE_DOWNLOADS" not in joined.replace("UNSLOTH_STABLE_DOWNLOADS / ", "")
    # dataset build follows CONFIG (think tags must match the model)
    assert '"--config", CONFIG' in joined
    # dataset-build phases must be BOUNDED: an unbounded subprocess.run wedged
    # interactive sessions when the smoke-build child hung at interpreter
    # shutdown after finishing its work (2026-08-08; likely v9's real stall)
    assert 'CONFIG], timeout=20 * 60' in joined
    assert joined.count("_probe_cmd, timeout=5 * 60") == 2
    # notebook cells read configs with plain yaml and never import the package:
    # a kernel started before the editable install misses the .pth, so only
    # subprocesses can import tuned
    assert "from tuned" not in joined
    # every run is re-homed to the session account's own HF namespace
    assert "whoami" in joined


def test_stage_model_notebook_matches_the_8b_pin():
    import yaml

    stage = json.loads((NB.parent / "stage_model.ipynb").read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in stage["cells"])
    cfg = yaml.safe_load(
        (Path(__file__).parent.parent / "configs" / "law_v1_8b_ddp.yaml").read_text(encoding="utf-8")
    )
    # the staging notebook must stage EXACTLY the lane's pinned repo+revision -
    # a drifted pin would make the fast path silently fall back (or worse,
    # stage a snapshot no lane can use)
    assert cfg["model"]["repo"] in src
    assert cfg["model"]["revision"] in src
    # staged layout contract shared with kaggle_smoke's pre-download cell
    assert "qwen3-8b-staged" in src
    assert "REVISION.txt" in src
