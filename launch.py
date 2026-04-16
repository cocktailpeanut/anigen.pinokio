import os
import shutil
import sys
from pathlib import Path


# Keep the upstream repo untouched while forcing the demo to bind to localhost.
repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import app as anigen_app


REQUIRED_CKPT_FILES = (
    Path("ckpts/anigen/ss_flow_duet/ckpts/denoiser.pt"),
    Path("ckpts/anigen/slat_flow_auto/ckpts/denoiser.pt"),
)


def reset_partial_checkpoints():
    if all(path.exists() for path in REQUIRED_CKPT_FILES):
        return

    ckpts_dir = Path("ckpts")
    hf_download_dir = Path(".cache/huggingface/download/ckpts")
    if ckpts_dir.exists():
        shutil.rmtree(ckpts_dir)
    if hf_download_dir.exists():
        shutil.rmtree(hf_download_dir)


reset_partial_checkpoints()
anigen_app.ensure_ckpts()
anigen_app.pipeline = anigen_app.AnigenImageTo3DPipeline.from_pretrained(
    ss_flow_path=f"ckpts/anigen/{anigen_app.DEFAULT_SS_MODEL}",
    slat_flow_path=f"ckpts/anigen/{anigen_app.DEFAULT_SLAT_MODEL}",
    device="cuda",
    use_ema=False,
)
anigen_app.pipeline.cuda()

launch_kwargs = {
    "server_name": os.environ.get("ANIGEN_GRADIO_SERVER_NAME", "127.0.0.1"),
    "share": os.environ.get("ANIGEN_GRADIO_SHARE", "0") == "1",
}
server_port = os.environ.get("ANIGEN_GRADIO_SERVER_PORT")
if server_port:
    launch_kwargs["server_port"] = int(server_port)

anigen_app.demo.launch(**launch_kwargs)
