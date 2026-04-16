module.exports = {
  run: [
    {
      when: "{{gpu !== 'nvidia'}}",
      method: "notify",
      params: {
        html: "AniGen requires an NVIDIA GPU."
      }
    },
    {
      when: "{{gpu === 'nvidia' && (platform === 'win32' || platform === 'linux') && !exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone --recurse-submodules https://github.com/VAST-AI-Research/AniGen app"
        ]
      }
    },
    {
      when: "{{gpu === 'nvidia' && platform === 'linux' && exists('app')}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        build: true,
        message: [
          "python -c \"from pathlib import Path; Path('.pinokio-installed').unlink(missing_ok=True)\"",
          "source ./setup.sh --demo",
          "uv pip install xformers==0.0.27.post2 rtree",
          "python -c \"import gradio, nvdiffrast.torch, pytorch3d.ops, rtree, spconv.pytorch, xformers.ops; from pathlib import Path; Path('.pinokio-installed').write_text('ok\\n')\""
        ]
      }
    },
    {
      when: "{{gpu === 'nvidia' && platform === 'win32' && exists('app')}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        build: true,
        message: [
          "python ../install_windows.py"
        ]
      }
    },
    {
      when: "{{gpu === 'nvidia' && (platform === 'win32' || platform === 'linux')}}",
      method: "notify",
      params: {
        html: "AniGen is installed. Click the 'Start' tab to launch the UI."
      }
    },
    {
      when: "{{gpu === 'nvidia' && platform !== 'win32' && platform !== 'linux'}}",
      method: "notify",
      params: {
        html: "AniGen currently has launcher install flows for Windows and Linux only."
      }
    }
  ]
}
