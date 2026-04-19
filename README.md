# AniGen

AniGen is a Pinokio launcher for [VAST-AI-Research/AniGen](https://github.com/VAST-AI-Research/AniGen), a Gradio demo for generating animatable 3D characters from a single image.

## Requirements

- NVIDIA GPU
- Windows x64 or Linux x64
- CUDA toolkit on `PATH` (`nvcc --version` must work)
- Enough VRAM for the upstream demo workflow

For Windows, the launcher also expects Visual Studio Build Tools 2019+ with the C++ toolchain installed so PyTorch3D and nvdiffrast can compile.

## What This Launcher Does

- Clones the upstream AniGen repository into `app/`
- Creates a Python virtual environment at `app/env`
- On Linux, runs the upstream `setup.sh --demo` install flow inside that environment
- On Linux, adds the matching `xformers` wheel required by AniGen's sparse attention modules
- On Windows, installs the AniGen dependency stack natively with `uv`, PyTorch CUDA wheels, `xformers`, `spconv`, PyTorch3D, and `nvdiffrast`
- Starts the Gradio demo through a small wrapper so the UI binds to `127.0.0.1` for Pinokio

## How To Use

1. Click `Install`.
2. Wait for the dependency install to finish. On Windows, compiling PyTorch3D and nvdiffrast can take a while.
3. Click `Start`.
4. Open the `Open Web UI` tab once the launcher captures the local Gradio URL.

The first launch may take longer because AniGen downloads model checkpoints automatically through Hugging Face.

If your GPU is tight on memory, use `Start (Low VRAM)` instead. That mode:

- keeps AniGen models on CPU until each stage needs them
- offloads them again after each stage finishes
- starts with lower sampling defaults
- disables texture baking by default to cut the largest extra memory spike
- skips hole filling in mesh postprocessing to avoid the extra Windows CUDA rasterization build path

The tradeoff is speed. Low-VRAM mode is slower than the normal launcher path.

## Launcher Scripts

- `install.js`: clones the upstream repo and runs the Linux or Windows install flow
- `install_windows.py`: native Windows dependency installer used by `install.js`
- `start.js`: launches the Gradio demo on localhost
- `start_lowvram.js`: launches a slower low-VRAM Gradio wrapper with CPU/GPU stage offloading
- `start_rig_viewer.js`: launches a separate local rig viewer for AniGen `mesh.glb` files
- `update.js`: pulls the launcher repo and the upstream AniGen repo
- `reset.js`: removes the cloned upstream app folder so you can reinstall cleanly

## Rig Viewer

`Start Rig Viewer` opens a separate local viewer that does not modify the upstream AniGen Gradio page.

Use it when you want to validate the exported rig itself:

1. Generate a character in AniGen.
2. In the output folder, use the main `mesh.glb`, not `skeleton.glb`.
3. Open `Start Rig Viewer`.
4. Drop `mesh.glb` into the viewer.
5. Click joint markers or use the joint list to pose the rig.
6. Export a posed GLB if you want to test the result elsewhere.

The viewer currently supports:

- textured skinned `mesh.glb` preview
- full joint list with search
- interactive joint rotation via gizmos
- root joint translation when needed
- skeleton overlay and clickable joint markers
- posed GLB export directly from the browser

## Windows Notes

- Upstream AniGen says the project is tested on Linux. This launcher adds a native Windows install path instead of routing through WSL.
- The Windows path intentionally keeps the upstream `app/` repo untouched. The platform-specific logic lives in the launcher root.
- PyTorch is pinned to the upstream-supported 2.4.x stack for compatibility with PyTorch3D.
- The launcher uses the matching `xformers` wheel for the Torch 2.4 stack and skips PyTorch3D's unused `pulsar` build on Windows because that component fails on the bundled CUDA 12.8 toolchain.
- `spconv` is installed from the Windows wheel matching the detected CUDA family.
- The Windows installer reuses an already-activated Pinokio MSVC environment when available, then falls back to detected Visual Studio toolchains instead of assuming the newest Visual Studio install is the right one.
- On Windows RTX 50-series / Blackwell GPUs, the installer switches to a dedicated `cu128` Torch stack and the launcher forces dense attention to `sdpa` while leaving sparse attention on `xformers`, which matches AniGen's current sparse-module requirements.
- If auto-detection misses a Blackwell GPU, set `ANIGEN_FORCE_BLACKWELL=1` before running `install.js` or `start.js` to force the Blackwell path.

## Programmatic Access

AniGen runs as a Gradio app. The main generation endpoint is `/image_to_3d`.

Inputs, in order:

1. Input image
2. Seed
3. S^3 model name
4. SLAT model name
5. S^3 guidance strength
6. S^3 sampling steps
7. SLAT guidance strength
8. SLAT sampling steps
9. Joints density
10. Texture size

Returns, in order:

1. Rigged mesh `.glb`
2. Skeleton `.glb`
3. Preview image

### Python

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:7860")
result = client.predict(
    handle_file("input.png"),
    0,
    "ss_flow_duet",
    "slat_flow_auto",
    7.5,
    25,
    3.0,
    25,
    1,
    1024,
    api_name="/image_to_3d",
)
print(result)
```

### JavaScript

```javascript
import { Client, handle_file } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const result = await client.predict("/image_to_3d", [
  handle_file("input.png"),
  0,
  "ss_flow_duet",
  "slat_flow_auto",
  7.5,
  25,
  3.0,
  25,
  1,
  1024,
]);

console.log(result.data);
```

### Curl

```bash
ANIGEN_URL="http://127.0.0.1:7860"

EVENT_ID=$(curl -s -X POST "$ANIGEN_URL/call/image_to_3d" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"path": "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"},
      0,
      "ss_flow_duet",
      "slat_flow_auto",
      7.5,
      25,
      3.0,
      25,
      1,
      1024
    ]
  }' | python -c "import json,sys; print(json.load(sys.stdin)['event_id'])")

curl -N "$ANIGEN_URL/call/image_to_3d/$EVENT_ID"
```

If you want to inspect the full Gradio schema from the running app, open the Gradio `View API` page from the footer once the launcher is running.
