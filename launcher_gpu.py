import os
import re
import subprocess
from functools import lru_cache


_FALSEY = {"0", "false", "no", "off"}
_BLACKWELL_NAME_RE = re.compile(r"\b(?:RTX\s*50\d{2}|Blackwell)\b", re.IGNORECASE)


def _env_flag(name: str):
    value = os.environ.get(name)
    if value is None:
        return None
    return value.strip().lower() not in _FALSEY


@lru_cache(maxsize=1)
def nvidia_gpu_names():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            errors="ignore",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def has_blackwell_gpu():
    forced = _env_flag("ANIGEN_FORCE_BLACKWELL")
    if forced is not None:
        return forced
    return any(_BLACKWELL_NAME_RE.search(name) for name in nvidia_gpu_names())


def apply_attention_profile():
    blackwell = has_blackwell_gpu()
    if blackwell:
        # Dense attention on RTX 50-series is more reliable through torch SDPA.
        os.environ["XFORMERS_DISABLED"] = "1"
        os.environ["ATTN_BACKEND"] = "sdpa"
        # Sparse attention still depends on xformers in upstream AniGen.
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
        os.environ["ANIGEN_GPU_FAMILY"] = "blackwell"
    else:
        os.environ.setdefault("ATTN_BACKEND", "xformers")
        os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
        os.environ.setdefault("ANIGEN_GPU_FAMILY", "legacy")
    return {
        "blackwell": blackwell,
        "gpu_names": nvidia_gpu_names(),
        "dense_backend": os.environ.get("ATTN_BACKEND"),
        "sparse_backend": os.environ.get("SPARSE_ATTN_BACKEND"),
    }
