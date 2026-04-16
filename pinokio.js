module.exports = {
  version: "7.0",
  title: "AniGen",
  description: "[NVIDIA GPU required] Unified S^3 fields for animatable 3D asset generation from a single image. Upstream is tested on Linux; this launcher also includes a native Windows install path. https://github.com/VAST-AI-Research/AniGen",
  menu: async (kernel, info) => {
    let installed = info.exists("app/.pinokio-installed")
    let appExists = info.exists("app")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      start_lowvram: info.running("start_lowvram.js"),
      start_rig_viewer: info.running("start_rig_viewer.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js")
    }

    let appendRunningLink = (items, script, options) => {
      if (!running[script]) {
        return
      }

      let local = info.local(`${script}.js`)
      if (local && local.url) {
        items.push({
          default: items.length === 0,
          icon: options.openIcon || "fa-solid fa-rocket",
          text: options.openText,
          href: local.url
        })
      }

      items.push({
        default: items.length === 0,
        icon: "fa-solid fa-terminal",
        text: options.terminalText,
        href: `${script}.js`
      })
    }

    let appendViewerAction = (items) => {
      if (running.start_rig_viewer) {
        return
      }

      items.push({
        icon: "fa-solid fa-bone",
        text: "Start Viewer",
        href: "start_rig_viewer.js"
      })
    }

    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js"
      }]
    } else if (running.update) {
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Updating",
        href: "update.js"
      }]
    } else if (running.reset) {
      return [{
        default: true,
        icon: "fa-solid fa-terminal",
        text: "Resetting",
        href: "reset.js"
      }]
    }

    let runningItems = []
    appendRunningLink(runningItems, "start", {
      openText: "Open AniGen UI",
      terminalText: "AniGen Terminal"
    })
    appendRunningLink(runningItems, "start_lowvram", {
      openText: "Open Low VRAM UI",
      terminalText: "Low VRAM Terminal"
    })
    appendRunningLink(runningItems, "start_rig_viewer", {
      openText: "Open Viewer",
      openIcon: "fa-solid fa-bone",
      terminalText: "Viewer Terminal"
    })
    if (runningItems.length) {
      appendViewerAction(runningItems)
      return runningItems
    } else if (installed) {
      return [{
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js"
      }, {
        icon: "fa-solid fa-memory",
        text: "Start (Low VRAM)",
        href: "start_lowvram.js"
      }, {
        icon: "fa-solid fa-bone",
        text: "Start Viewer",
        href: "start_rig_viewer.js"
      }, {
        icon: "fa-solid fa-plug",
        text: "Update",
        href: "update.js"
      }, {
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js"
      }, {
        icon: "fa-regular fa-circle-xmark",
        text: "Reset",
        href: "reset.js"
      }]
    }

    let menu = [{
      default: true,
      icon: "fa-solid fa-plug",
      text: "Install",
      href: "install.js"
    }]
    menu.push({
      icon: "fa-solid fa-bone",
      text: "Start Viewer",
      href: "start_rig_viewer.js"
    })
    if (appExists) {
      menu.push({
        icon: "fa-regular fa-circle-xmark",
        text: "Reset",
        href: "reset.js"
      })
    }
    return menu
  }
}
