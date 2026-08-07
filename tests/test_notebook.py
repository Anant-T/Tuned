import json
from pathlib import Path

NB = Path(__file__).parent.parent / "notebooks" / "kaggle_smoke.ipynb"


def test_notebook_is_valid_and_complete():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    sources = ["".join(c["source"]) for c in nb["cells"]]
    joined = "\n".join(sources)
    # the operator's switches exist and default to the cheap single-GPU gate
    assert 'MODE = "SAVETEST"' in joined
    assert "DDP = False" in joined
    assert "MP = False" in joined
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
    # MP must NOT go through torchrun - only DDP selects it
    assert '["torchrun", "--nproc_per_node=2"] if DDP else' in joined
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
    # notebook cells read configs with plain yaml and never import the package:
    # a kernel started before the editable install misses the .pth, so only
    # subprocesses can import tuned
    assert "from tuned" not in joined
    # every run is re-homed to the session account's own HF namespace
    assert "whoami" in joined
