module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        build: true,
        venv: "env",
        env: {
          ANIGEN_GRADIO_SERVER_NAME: "127.0.0.1",
          ANIGEN_GRADIO_SHARE: "0",
          ATTN_BACKEND: "xformers",
          SPARSE_ATTN_BACKEND: "xformers",
          PYTHONUNBUFFERED: "1"
        },
        path: "app",
        message: [
          "python ../launch.py"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
