import gc
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from gradio_litmodel3d import LitModel3D


repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from anigen.pipelines.anigen_image_to_3d import AnigenImageTo3DPipeline
from anigen.utils.ckpt_utils import ensure_ckpts
from anigen.utils.random_utils import set_random_seed


MAX_SEED = 100
TMP_DIR = os.path.join(repo_root, "tmp_lowvram")
os.makedirs(TMP_DIR, exist_ok=True)

SS_MODEL_CHOICES = ["ss_flow_duet", "ss_flow_solo", "ss_flow_epic"]
SLAT_MODEL_CHOICES = ["slat_flow_auto", "slat_flow_control"]
DEFAULT_SS_MODEL = "ss_flow_duet"
DEFAULT_SLAT_MODEL = "slat_flow_auto"

LOW_VRAM_DEFAULT_SS_STEPS = 16
LOW_VRAM_DEFAULT_SLAT_STEPS = 12
LOW_VRAM_DEFAULT_TEXTURE_SIZE = 0
LOW_VRAM_MAX_TEXTURE_SIZE = 512


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LowVramAnigenPipeline(AnigenImageTo3DPipeline):
    def __init__(self, models=None, ss_config=None, slat_config=None):
        super().__init__(models=models, ss_config=ss_config, slat_config=slat_config)
        self._device = torch.device("cpu")
        self._active_names = set()

    @property
    def device(self) -> torch.device:
        return self._device

    @classmethod
    def from_pretrained(
        cls,
        ss_flow_path: str = f"ckpts/anigen/{DEFAULT_SS_MODEL}",
        slat_flow_path: str = f"ckpts/anigen/{DEFAULT_SLAT_MODEL}",
        use_ema: bool = False,
    ) -> "LowVramAnigenPipeline":
        base = AnigenImageTo3DPipeline.from_pretrained(
            ss_flow_path=ss_flow_path,
            slat_flow_path=slat_flow_path,
            device="cpu",
            use_ema=use_ema,
        )
        pipeline = cls(base.models, base.ss_config, base.slat_config)
        pipeline.release_vram()
        return pipeline

    def _move_model(self, name: str, device: str):
        model = self.models[name]
        target = torch.device(device)
        if hasattr(model, "to"):
            model.to(target)
        elif device == "cuda" and hasattr(model, "cuda"):
            model.cuda()
        elif device == "cpu" and hasattr(model, "cpu"):
            model.cpu()
        if hasattr(model, "eval"):
            model.eval()

    def _set_active(self, *names: str):
        requested = set(names)
        if requested == self._active_names:
            return

        for name in sorted(self._active_names - requested):
            self._move_model(name, "cpu")
        for name in sorted(requested - self._active_names):
            self._move_model(name, "cuda")

        self._active_names = requested
        self._device = torch.device("cuda" if requested else "cpu")
        cuda_cleanup()

    def release_vram(self):
        self._set_active()

    def load_ss_flow_model(self, ss_flow_path: str, device: str = "cpu", use_ema: bool = False):
        super().load_ss_flow_model(ss_flow_path, device=device, use_ema=use_ema)
        self.release_vram()

    def load_slat_flow_model(self, slat_flow_path: str, device: str = "cpu", use_ema: bool = False):
        super().load_slat_flow_model(slat_flow_path, device=device, use_ema=use_ema)
        self.release_vram()

    def preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, Image.Image]:
        self._set_active("dsine")
        try:
            return super().preprocess_image(image)
        finally:
            self.release_vram()

    def get_cond(self, image: Image.Image, normal_image: Image.Image):
        self._set_active("image_cond_model")
        try:
            return super().get_cond(image, normal_image)
        finally:
            self.release_vram()

    def sample_sparse_structure(self, *args, **kwargs):
        self._set_active("ss_flow_model", "ss_decoder")
        try:
            return super().sample_sparse_structure(*args, **kwargs)
        finally:
            self.release_vram()

    def sample_slat(self, *args, **kwargs):
        self._set_active("slat_flow_model")
        try:
            return super().sample_slat(*args, **kwargs)
        finally:
            self.release_vram()

    def decode_slat(self, *args, **kwargs):
        self._set_active("slat_decoder")
        try:
            return super().decode_slat(*args, **kwargs)
        finally:
            self.release_vram()


current_ss_model_name = DEFAULT_SS_MODEL
current_slat_model_name = DEFAULT_SLAT_MODEL
pipeline = None


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image_preview(image: Image.Image) -> Image.Image:
    processed_image, _ = pipeline.preprocess_image(image)
    pipeline.release_vram()
    return processed_image


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def on_slat_model_change(slat_model_name: str):
    is_control = slat_model_name == "slat_flow_control"
    return (
        gr.update(interactive=is_control),
        gr.update(visible=not is_control),
    )


def image_to_3d(
    image: Image.Image,
    seed: int,
    ss_model_name: str,
    slat_model_name: str,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    joints_density: int,
    texture_size: int,
    req: gr.Request = None,
    progress=gr.Progress(track_tqdm=False),
):
    global current_ss_model_name, current_slat_model_name

    texture_size = max(0, min(int(texture_size), LOW_VRAM_MAX_TEXTURE_SIZE))
    ss_sampling_steps = max(1, int(ss_sampling_steps))
    slat_sampling_steps = max(1, int(slat_sampling_steps))
    joints_density = max(0, min(int(joints_density), 4))

    session_id = req.session_hash if req else uuid.uuid4().hex
    user_dir = os.path.join(TMP_DIR, session_id)
    os.makedirs(user_dir, exist_ok=True)

    run_id = uuid.uuid4().hex
    run_dir = os.path.join(user_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    output_glb_path = os.path.join(run_dir, "mesh.glb")
    skeleton_glb_path = os.path.join(run_dir, "skeleton.glb")

    try:
        if ss_model_name != current_ss_model_name:
            progress(0, desc=f"Loading SS model: {ss_model_name}...")
            pipeline.load_ss_flow_model(f"ckpts/anigen/{ss_model_name}", use_ema=False)
            current_ss_model_name = ss_model_name

        if slat_model_name != current_slat_model_name:
            progress(0, desc=f"Loading SLAT model: {slat_model_name}...")
            pipeline.load_slat_flow_model(f"ckpts/anigen/{slat_model_name}", use_ema=False)
            current_slat_model_name = slat_model_name

        def ss_progress_callback(step, total):
            frac = (step + 1) / total
            progress(frac * 0.45, desc=f"SS Sampling: {step + 1}/{total}")

        def slat_progress_callback(step, total):
            frac = (step + 1) / total
            progress(0.45 + frac * 0.45, desc=f"SLAT Sampling: {step + 1}/{total}")

        def postprocess_progress_callback(frac, desc):
            progress(0.90 + frac * 0.10, desc=desc)

        outputs = pipeline.run(
            image,
            seed=seed,
            cfg_scale_ss=ss_guidance_strength,
            cfg_scale_slat=slat_guidance_strength,
            ss_steps=ss_sampling_steps,
            slat_steps=slat_sampling_steps,
            joints_density=joints_density,
            fill_holes=False,
            no_smooth_skin_weights=False,
            no_filter_skin_weights=False,
            smooth_skin_weights_iters=100,
            smooth_skin_weights_alpha=1.0,
            texture_size=texture_size,
            output_glb=output_glb_path,
            ss_progress_callback=ss_progress_callback,
            slat_progress_callback=slat_progress_callback,
            postprocess_progress_callback=postprocess_progress_callback,
        )

        processed_image = outputs["processed_image"]
        del outputs
        gc.collect()
        cuda_cleanup()

        if not os.path.exists(skeleton_glb_path):
            skeleton_glb_path = None

        return output_glb_path, skeleton_glb_path, processed_image
    finally:
        pipeline.release_vram()
        cuda_cleanup()


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown(
        """
        ## AniGen Low VRAM
        This mode keeps models on CPU until each stage needs them, then offloads them again. It is slower than the regular launcher but reduces idle and peak VRAM pressure.
        """
    )

    gr.Markdown(
        """
        Low-VRAM defaults:
        - lower sampling steps
        - texture baking disabled by default
        - same AniGen checkpoint family, but staged onto CUDA one piece at a time
        """
    )

    with gr.Row():
        with gr.Column():
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=300)

            with gr.Accordion(label="Generation Settings", open=True):
                seed = gr.Slider(0, MAX_SEED, label="Seed", value=42, step=1)
                randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)

                gr.Markdown("**Model Selection**")
                with gr.Row():
                    ss_model_dropdown = gr.Dropdown(
                        choices=SS_MODEL_CHOICES,
                        value=DEFAULT_SS_MODEL,
                        label="SS Model (Sparse Structure)",
                    )
                    slat_model_dropdown = gr.Dropdown(
                        choices=SLAT_MODEL_CHOICES,
                        value=DEFAULT_SLAT_MODEL,
                        label="SLAT Model (Structured Latent)",
                    )

                gr.Markdown("Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 15.0, label="Guidance Strength", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 30, label="Sampling Steps", value=LOW_VRAM_DEFAULT_SS_STEPS, step=1)

                gr.Markdown("Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 30, label="Sampling Steps", value=LOW_VRAM_DEFAULT_SLAT_STEPS, step=1)

                gr.Markdown("Skeleton & Skinning Settings")
                joints_density = gr.Slider(0, 4, label="Joints Density", value=1, step=1, interactive=False)
                density_hint = gr.Markdown(
                    "*Switch `SLAT Model` to `slat_flow_control` to enable joint density control.*",
                    visible=True,
                )

                gr.Markdown("Texture Settings")
                texture_size = gr.Slider(
                    0,
                    LOW_VRAM_MAX_TEXTURE_SIZE,
                    label="Texture Resolution",
                    value=LOW_VRAM_DEFAULT_TEXTURE_SIZE,
                    step=256,
                    info="0 disables texture baking and saves the most VRAM.",
                )

            generate_btn = gr.Button("Generate")

        with gr.Column():
            mesh_output = gr.Model3D(label="Generated Mesh", height=300, interactive=False)
            download_mesh = gr.DownloadButton(label="Download Mesh GLB", interactive=False)
            skeleton_output = LitModel3D(label="Generated Skeleton", exposure=5.0, height=300, interactive=False)
            download_skeleton = gr.DownloadButton(label="Download Skeleton GLB", interactive=False)
            processed_image_output = gr.Image(label="Processed Image", type="pil", height=300)

    demo.load(start_session)
    demo.unload(end_session)

    image_prompt.upload(
        preprocess_image_preview,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    slat_model_dropdown.change(
        on_slat_model_change,
        inputs=[slat_model_dropdown],
        outputs=[joints_density, density_hint],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[
            image_prompt,
            seed,
            ss_model_dropdown,
            slat_model_dropdown,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
            joints_density,
            texture_size,
        ],
        outputs=[mesh_output, skeleton_output, processed_image_output],
    ).then(
        lambda mesh_path, skel_path: tuple(
            [
                gr.DownloadButton(value=mesh_path, interactive=True) if mesh_path else gr.DownloadButton(interactive=False),
                gr.DownloadButton(value=skel_path, interactive=True) if skel_path else gr.DownloadButton(interactive=False),
            ]
        ),
        inputs=[mesh_output, skeleton_output],
        outputs=[download_mesh, download_skeleton],
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("AniGen low-VRAM mode still requires CUDA.")

    ensure_ckpts()
    pipeline = LowVramAnigenPipeline.from_pretrained(
        ss_flow_path=f"ckpts/anigen/{DEFAULT_SS_MODEL}",
        slat_flow_path=f"ckpts/anigen/{DEFAULT_SLAT_MODEL}",
        use_ema=False,
    )

    launch_kwargs = {
        "server_name": os.environ.get("ANIGEN_GRADIO_SERVER_NAME", "127.0.0.1"),
        "share": os.environ.get("ANIGEN_GRADIO_SHARE", "0") == "1",
    }
    server_port = os.environ.get("ANIGEN_GRADIO_SERVER_PORT")
    if server_port:
        launch_kwargs["server_port"] = int(server_port)

    demo.launch(**launch_kwargs)
