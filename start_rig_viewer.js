module.exports = {
  daemon: true,
  run: [
    {
      when: "{{exists('viewer/package-lock.json') && !exists('viewer/node_modules/three/package.json')}}",
      method: "shell.run",
      params: {
        path: "viewer",
        message: [
          "npm ci --no-fund --no-audit"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        path: "viewer",
        message: [
          "python -m http.server {{port}} --bind 127.0.0.1"
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
