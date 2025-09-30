# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sgad",
    local_dir=Path(__file__).parent / "models",
    allow_patterns=[
        "am/",
        "bmam/",
        "pretrain_for_bmam/",
        "torsion/",
    ],
)
