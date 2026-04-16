module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    when: "{{exists('app')}}",
    method: "shell.run",
    params: {
      path: "app",
      message: [
        "git pull",
        "git submodule update --init --recursive"
      ]
    }
  }, {
    when: "{{exists('viewer/package-lock.json')}}",
    method: "shell.run",
    params: {
      path: "viewer",
      message: [
        "npm ci --no-fund --no-audit"
      ]
    }
  }]
}
