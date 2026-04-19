import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from launcher_gpu import has_blackwell_gpu, nvidia_gpu_names

CUB_SOURCE = "https://github.com/NVIDIA/cub/archive/refs/tags/1.10.0.tar.gz"
PYTORCH3D_SOURCE = "https://github.com/facebookresearch/pytorch3d/archive/refs/tags/V0.7.8.tar.gz"
PYTORCH3D_WHEEL_INDEX = "https://miropsota.github.io/torch_packages_builder/pytorch3d/"
NVDIFFRAST_SOURCE = "https://github.com/NVlabs/nvdiffrast/archive/refs/tags/v0.3.3.tar.gz"
INSTALL_MARKER = ".pinokio-installed"
DEMO_PACKAGES = [
    "gradio==4.44.1",
    "gradio_litmodel3d==0.0.1",
    "fastapi==0.112.2",
    "starlette==0.38.6",
    "jinja2==3.1.5",
    "pydantic==2.10.6",
    "huggingface_hub<0.25",
    "rtree",
]
SPCONV_PACKAGES = [
    "spconv",
    "spconv-cu118",
    "spconv-cu120",
    "spconv-cu121",
    "spconv-cu124",
    "spconv-cu126",
    "spconv-cu128",
    "cumm",
    "cumm-cu118",
    "cumm-cu120",
    "cumm-cu121",
    "cumm-cu124",
    "cumm-cu126",
    "cumm-cu128",
]

LEGACY_TORCH_STACK = {
    "torch_version": "2.4.0",
    "torchvision_version": "0.19.0",
    "torchaudio_version": "2.4.0",
    "xformers_version": "0.0.27.post2",
    "pytorch3d_wheels": ("0.7.8+pt2.4.0", "0.7.7+pt2.4.0"),
}

BLACKWELL_TORCH_STACK = {
    # Match the current Pinokio cu128 convention for NVIDIA Windows installs.
    "torch_version": "2.7.0",
    "torchvision_version": "0.22.0",
    "torchaudio_version": "2.7.0",
    "xformers_version": "0.0.30",
    "pytorch3d_wheels": (
        "0.7.9+pt2.7.0",
        "0.7.8+pt2.7.0",
        "0.7.8+pt2.4.0",
        "0.7.7+pt2.4.0",
    ),
}


def launcher_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "requirements.txt").exists():
        return cwd
    candidate = launcher_root() / "app"
    if (candidate / "requirements.txt").exists():
        return candidate
    raise SystemExit("Could not resolve the AniGen app directory.")


def build_base_env():
    root = launcher_root()
    uv_cache = root / "cache" / "uv"
    pip_cache = root / "cache" / "pip"
    uv_cache.mkdir(parents=True, exist_ok=True)
    pip_cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(uv_cache)
    env["PIP_CACHE_DIR"] = str(pip_cache)
    return env


def source_cache_root() -> Path:
    root = launcher_root() / "cache" / "src"
    root.mkdir(parents=True, exist_ok=True)
    return root


def install_marker_path(repo_root: Path) -> Path:
    return repo_root / INSTALL_MARKER


def clear_install_marker(repo_root: Path):
    install_marker_path(repo_root).unlink(missing_ok=True)


def write_install_marker(repo_root: Path, dry_run=False):
    if dry_run:
        return
    install_marker_path(repo_root).write_text("ok\n", encoding="utf-8")


def download(url: str, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(destination)


def prepare_source(name: str, url: str) -> Path:
    root = source_cache_root()
    archive = root / f"{name}.tar.gz"
    extract_root = root / f"{name}-extract"
    final_dir = root / name
    if final_dir.exists():
        return final_dir
    if archive.exists():
        archive.unlink()
    if extract_root.exists():
        shutil.rmtree(extract_root)
    download(url, archive)
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_root)
    extracted = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(extracted) != 1:
        raise SystemExit(f"Unexpected archive layout for {name}.")
    extracted[0].replace(final_dir)
    shutil.rmtree(extract_root)
    return final_dir


def replace_once(path: Path, old: str, new: str):
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"Could not patch expected content in {path}.")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_pytorch3d_for_windows(source_root: Path):
    setup_path = source_root / "setup.py"
    replace_once(
        setup_path,
        "    define_macros = []\n    include_dirs = [extensions_dir]\n",
        "    define_macros = []\n"
        "    include_dirs = [extensions_dir]\n"
        '    disable_pulsar = os.getenv("PYTORCH3D_DISABLE_PULSAR", "0") == "1"\n',
    )
    replace_once(
        setup_path,
        "        extra_compile_args[\"nvcc\"] = nvcc_args\n\n    sources = [os.path.join(extensions_dir, s) for s in sources]\n",
        "        extra_compile_args[\"nvcc\"] = nvcc_args\n\n"
        "    if disable_pulsar:\n"
        "        define_macros += [(\"PYTORCH3D_DISABLE_PULSAR\", None)]\n"
        "        sources = [s for s in sources if f\"{os.path.sep}pulsar{os.path.sep}\" not in s]\n\n"
        "    sources = [os.path.join(extensions_dir, s) for s in sources]\n",
    )

    ext_cpp = source_root / "pytorch3d" / "csrc" / "ext.cpp"
    replace_once(
        ext_cpp,
        "#if !defined(USE_ROCM)\n#include \"./pulsar/global.h\" // Include before <torch/extension.h>.\n#endif\n",
        "#if !defined(USE_ROCM) && !defined(PYTORCH3D_DISABLE_PULSAR)\n"
        "#include \"./pulsar/global.h\" // Include before <torch/extension.h>.\n"
        "#endif\n",
    )
    replace_once(
        ext_cpp,
        "#if !defined(USE_ROCM)\n#include \"./pulsar/pytorch/renderer.h\"\n#include \"./pulsar/pytorch/tensor_util.h\"\n#endif\n",
        "#if !defined(USE_ROCM) && !defined(PYTORCH3D_DISABLE_PULSAR)\n"
        "#include \"./pulsar/pytorch/renderer.h\"\n"
        "#include \"./pulsar/pytorch/tensor_util.h\"\n"
        "#endif\n",
    )
    replace_once(
        ext_cpp,
        "#if !defined(USE_ROCM)\n#ifdef PULSAR_LOGGING_ENABLED\n",
        "#if !defined(USE_ROCM) && !defined(PYTORCH3D_DISABLE_PULSAR)\n#ifdef PULSAR_LOGGING_ENABLED\n",
    )

    renderer_init = source_root / "pytorch3d" / "renderer" / "__init__.py"
    replace_once(
        renderer_init,
        "# Pulsar is not enabled on amd.\nif not torch.version.hip:\n    from .points import PulsarPointsRenderer\n",
        "# Pulsar is optional in the launcher's Windows build.\n"
        "if not torch.version.hip:\n"
        "    try:\n"
        "        from .points import PulsarPointsRenderer\n"
        "    except (AttributeError, ImportError):\n"
        "        pass\n",
    )

    points_init = source_root / "pytorch3d" / "renderer" / "points" / "__init__.py"
    replace_once(
        points_init,
        "# Pulsar not enabled on amd.\nif not torch.version.hip:\n    from .pulsar.unified import PulsarPointsRenderer\n",
        "# Pulsar is optional in the launcher's Windows build.\n"
        "if not torch.version.hip:\n"
        "    try:\n"
        "        from .pulsar.unified import PulsarPointsRenderer\n"
        "    except (AttributeError, ImportError):\n"
        "        pass\n",
    )

    build_dir = source_root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)


def resolve_cub_home(dry_run=False) -> Path:
    if dry_run:
        return source_cache_root() / "cub"

    nvcc = shutil.which("nvcc")
    if nvcc:
        candidate = Path(nvcc).resolve().parent.parent / "include"
        if (candidate / "cub").exists():
            return candidate
    return prepare_source("cub", CUB_SOURCE)


def run(cmd, cwd=None, env=None, dry_run=False):
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def uv_pip(args, cwd=None, env=None, dry_run=False):
    uv = shutil.which("uv")
    if uv:
        command = [uv, "pip", "install", "--python", sys.executable, *args]
    else:
        command = [sys.executable, "-m", "pip", "install", *args]
    run(command, cwd=cwd, env=env, dry_run=dry_run)


def uv_pip_uninstall(packages, cwd=None, env=None, dry_run=False):
    uv = shutil.which("uv")
    if uv:
        command = [uv, "pip", "uninstall", "--python", sys.executable, *packages]
    else:
        command = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    print("+", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=False)


def install_public_pytorch3d_wheel(repo_root: Path, cuda, env=None, dry_run=False) -> bool:
    uv_pip_uninstall(["pytorch3d"], cwd=repo_root, env=env, dry_run=dry_run)
    for version in cuda["pytorch3d_wheels"]:
        package = f"pytorch3d=={version}{cuda['pytorch3d_compute']}"
        print(f"Trying public PyTorch3D wheel {package}")
        try:
            uv_pip(
                ["--no-index", "--no-deps", "--find-links", PYTORCH3D_WHEEL_INDEX, package],
                cwd=repo_root,
                env=env,
                dry_run=dry_run,
            )
        except subprocess.CalledProcessError:
            print(f"Public PyTorch3D wheel unavailable for {package}.")
        else:
            return True
    return False


def detect_cuda():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise SystemExit("nvcc was not found on PATH. Install the CUDA toolkit before installing AniGen.")
    output = subprocess.check_output([nvcc, "--version"], text=True, errors="ignore")
    match = re.search(r"release (\d+)\.(\d+)", output)
    if not match:
        raise SystemExit("Could not detect the CUDA version from `nvcc --version`.")
    major = int(match.group(1))
    minor = int(match.group(2))
    blackwell = has_blackwell_gpu()
    if major >= 12:
        if blackwell:
            return {
                "version": f"{major}.{minor}",
                "profile": "blackwell",
                "blackwell": True,
                "torch_index": "https://download.pytorch.org/whl/cu128",
                "pytorch3d_compute": "cu128",
                # spconv's cu121 wheel still works with the cu128 runtime via CUDA minor compat.
                "spconv_package": "spconv-cu121",
                **BLACKWELL_TORCH_STACK,
            }
        return {
            "version": f"{major}.{minor}",
            "profile": "legacy",
            "blackwell": False,
            "torch_index": "https://download.pytorch.org/whl/cu121",
            "pytorch3d_compute": "cu121",
            "spconv_package": "spconv-cu121",
            **LEGACY_TORCH_STACK,
        }
    if major == 11:
        return {
            "version": f"{major}.{minor}",
            "profile": "legacy",
            "blackwell": False,
            "torch_index": "https://download.pytorch.org/whl/cu118",
            "pytorch3d_compute": "cu118",
            "spconv_package": "spconv-cu118",
            **LEGACY_TORCH_STACK,
        }
    raise SystemExit(f"CUDA {major}.{minor} is too old for AniGen's dependency stack.")


def finalize_msvc_env(base_env):
    env = base_env.copy()
    env["DISTUTILS_USE_SDK"] = "1"
    env["MSSdk"] = "1"
    env["USE_NINJA"] = "0"
    env["FORCE_CUDA"] = "1"
    env["MAX_JOBS"] = "1"
    cuda_home = env.get("CUDA_HOME") or env.get("CUDA_PATH")
    if not cuda_home:
        nvcc = shutil.which("nvcc", path=env.get("PATH"))
        if nvcc:
            cuda_home = str(Path(nvcc).resolve().parent.parent)
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
        cuda_lib = Path(cuda_home) / "lib"
        if cuda_lib.exists():
            existing = env.get("LIB", "")
            parts = [str(cuda_lib)]
            if existing:
                parts.append(existing)
            env["LIB"] = ";".join(parts)
    if shutil.which("cl.exe", path=env.get("PATH")):
        env.setdefault("CC", "cl.exe")
        env.setdefault("CXX", "cl.exe")
    return env


def has_msvc_toolchain(env):
    return bool(
        shutil.which("cl.exe", path=env.get("PATH"))
        and (
            env.get("VCToolsInstallDir")
            or env.get("VCINSTALLDIR")
            or env.get("VSINSTALLDIR")
        )
    )


def resolve_vswhere():
    vswhere = shutil.which("vswhere")
    if not vswhere:
        fallback = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
        if fallback.exists():
            vswhere = str(fallback)
    return vswhere


def append_msvc_candidate(candidates, seen, script_path: Path, args=""):
    candidate = Path(script_path)
    if not candidate.exists():
        return
    key = (str(candidate).lower(), args)
    if key in seen:
        return
    seen.add(key)
    candidates.append((candidate, args))


def iter_msvc_activation_candidates(base_env):
    candidates = []
    seen = set()

    vsinstall = base_env.get("VSINSTALLDIR")
    if vsinstall:
        root = Path(vsinstall.rstrip("\\/"))
        append_msvc_candidate(candidates, seen, root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        append_msvc_candidate(
            candidates,
            seen,
            root / "Common7" / "Tools" / "VsDevCmd.bat",
            "-arch=x64 -host_arch=x64",
        )

    vswhere = resolve_vswhere()
    if not vswhere:
        return candidates

    queries = [
        (r"VC\Auxiliary\Build\vcvars64.bat", ""),
        (r"Common7\Tools\VsDevCmd.bat", "-arch=x64 -host_arch=x64"),
    ]
    for pattern, args in queries:
        output = subprocess.check_output(
            [
                vswhere,
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-find",
                pattern,
            ],
            text=True,
            errors="ignore",
        )
        for line in output.splitlines():
            line = line.strip()
            if line:
                append_msvc_candidate(candidates, seen, Path(line), args)
    return candidates


def capture_batch_env(batch_file: Path, args: str, base_env):
    command = f'call "{batch_file}"'
    if args:
        command = f"{command} {args}"
    command = f"{command} >nul && set"
    completed = subprocess.run(
        command,
        text=True,
        errors="ignore",
        capture_output=True,
        env=base_env,
        check=False,
        shell=True,
        executable=os.environ.get("COMSPEC", "cmd.exe"),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if detail:
            detail = "\n".join(detail.splitlines()[-8:])
        raise RuntimeError(detail or f"exit status {completed.returncode}")

    env = base_env.copy()
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return finalize_msvc_env(env)


def capture_msvc_env(base_env):
    current_env = finalize_msvc_env(base_env)
    if has_msvc_toolchain(current_env):
        print("Using existing MSVC environment from the Pinokio shell.")
        return current_env

    candidates = iter_msvc_activation_candidates(base_env)
    if not candidates:
        raise SystemExit(
            "Could not find a usable Visual Studio C++ environment. "
            "Install Visual Studio Build Tools 2019 or newer with the Desktop development with C++ workload."
        )

    errors = []
    for batch_file, args in candidates:
        try:
            env = capture_batch_env(batch_file, args, base_env)
        except RuntimeError as exc:
            errors.append(f"{batch_file}: {exc}")
            continue
        if has_msvc_toolchain(env):
            print(f"Activated MSVC toolchain via {batch_file.name}.")
            return env
        errors.append(f"{batch_file}: activated, but cl.exe was still unavailable.")

    detail = "\n".join(f"- {line}" for line in errors[-4:])
    message = (
        "Could not initialize the Visual Studio C++ build environment. "
        "Install Visual Studio Build Tools 2019 or newer with the Desktop development with C++ workload "
        "and a Windows SDK."
    )
    if detail:
        message = f"{message}\n{detail}"
    raise SystemExit(message)


def verify_install(repo_root: Path, dry_run=False):
    code = "\n".join(
        [
            "import gradio",
            "import nvdiffrast.torch",
            "import pytorch3d.ops",
            "import rtree",
            "import spconv.pytorch",
            "import torch",
            "import xformers.ops",
            "print('torch', torch.__version__)",
            "print('cuda', torch.version.cuda)",
            "print('gradio', gradio.__version__)",
        ]
    )
    command = [sys.executable, "-c", code]
    print("+", " ".join(command))
    if dry_run:
        return

    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        errors="ignore",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise subprocess.CalledProcessError(
            completed.returncode,
            command,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")


def summarize_probe_failure(stdout: str, stderr: str) -> str:
    lines = []
    if stdout:
        lines.extend(stdout.splitlines())
    if stderr:
        lines.extend(stderr.splitlines())

    for line in reversed(lines):
        text = line.strip()
        if not text:
            continue
        match = re.search(r"No module named '([^']+)'", text)
        if match:
            return f"missing module: {match.group(1)}"
        return text
    return ""


def probe_install(repo_root: Path):
    code = "\n".join(
        [
            "import gradio",
            "import nvdiffrast.torch",
            "import pytorch3d.ops",
            "import rtree",
            "import spconv.pytorch",
            "import torch",
            "import xformers.ops",
            "print('torch', torch.__version__)",
            "print('cuda', torch.version.cuda)",
            "print('gradio', gradio.__version__)",
        ]
    )
    command = [sys.executable, "-c", code]
    print("+", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        errors="ignore",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return False, summarize_probe_failure(completed.stdout, completed.stderr)

    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    return True, ""


def main():
    dry_run = "--dry-run" in sys.argv
    repo_root = resolve_repo_root()
    base_env = build_base_env()
    print(f"Using Python {sys.version.split()[0]}")

    if not dry_run:
        if install_marker_path(repo_root).exists():
            verified, detail = probe_install(repo_root)
            if verified:
                print("AniGen dependencies are already verified. Skipping reinstall.")
                write_install_marker(repo_root)
                return
            if detail:
                print(f"Existing AniGen install is incomplete or broken ({detail}). Reinstalling dependencies.")
            else:
                print("Existing AniGen install is incomplete or broken. Reinstalling dependencies.")
        else:
            print("No verified AniGen install marker found. Installing dependencies.")

    cuda = detect_cuda()
    gpu_names = nvidia_gpu_names()
    if gpu_names:
        print("Detected NVIDIA GPU(s):", ", ".join(gpu_names))
    print(f"Using install profile: {cuda['profile']}")
    print(f"Detected CUDA toolkit {cuda['version']}")
    print(f"Using PyTorch index {cuda['torch_index']}")
    print(f"Using {cuda['spconv_package']} for spconv")

    clear_install_marker(repo_root)

    uv_pip(
        ["--upgrade", "pip", "setuptools", "wheel", "packaging", "cmake", "ninja", "iopath", "fvcore"],
        cwd=repo_root,
        env=base_env,
        dry_run=dry_run,
    )
    uv_pip(
        [
            f"torch=={cuda['torch_version']}",
            f"torchvision=={cuda['torchvision_version']}",
            f"torchaudio=={cuda['torchaudio_version']}",
            "--index-url",
            cuda["torch_index"],
        ],
        cwd=repo_root,
        env=base_env,
        dry_run=dry_run,
    )
    uv_pip([f"xformers=={cuda['xformers_version']}"], cwd=repo_root, env=base_env, dry_run=dry_run)
    uv_pip(["-r", "requirements.txt"], cwd=repo_root, env=base_env, dry_run=dry_run)
    uv_pip(DEMO_PACKAGES, cwd=repo_root, env=base_env, dry_run=dry_run)

    uv_pip_uninstall(SPCONV_PACKAGES, cwd=repo_root, env=base_env, dry_run=dry_run)
    uv_pip([cuda["spconv_package"]], cwd=repo_root, env=base_env, dry_run=dry_run)

    pytorch3d_from_wheel = install_public_pytorch3d_wheel(repo_root, cuda, env=base_env, dry_run=dry_run)
    msvc_env = capture_msvc_env(base_env)
    if not pytorch3d_from_wheel:
        cub_source = resolve_cub_home(dry_run=dry_run)
        if dry_run:
            pytorch3d_source = source_cache_root() / "p3d"
        else:
            pytorch3d_source = prepare_source("p3d", PYTORCH3D_SOURCE)
            patch_pytorch3d_for_windows(pytorch3d_source)
        msvc_env["CUB_HOME"] = str(cub_source)
        msvc_env["PYTORCH3D_DISABLE_PULSAR"] = "1"
        uv_pip(
            ["--no-build-isolation", str(pytorch3d_source)],
            cwd=repo_root,
            env=msvc_env,
            dry_run=dry_run,
        )
    if dry_run:
        nvdiffrast_source = source_cache_root() / "nvd"
    else:
        nvdiffrast_source = prepare_source("nvd", NVDIFFRAST_SOURCE)
    uv_pip(
        ["--no-build-isolation", str(nvdiffrast_source)],
        cwd=repo_root,
        env=msvc_env,
        dry_run=dry_run,
    )

    verify_install(repo_root, dry_run=dry_run)
    write_install_marker(repo_root, dry_run=dry_run)


if __name__ == "__main__":
    main()
