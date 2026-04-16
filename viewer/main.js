import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const elements = {
  viewport: document.getElementById("viewport"),
  fileInput: document.getElementById("file-input"),
  dropzone: document.getElementById("dropzone"),
  status: document.getElementById("status"),
  fitCamera: document.getElementById("fit-camera"),
  resetPose: document.getElementById("reset-pose"),
  resetJoint: document.getElementById("reset-joint"),
  exportGlb: document.getElementById("export-glb"),
  modeRotate: document.getElementById("mode-rotate"),
  modeTranslate: document.getElementById("mode-translate"),
  toggleGrid: document.getElementById("toggle-grid"),
  toggleSkeleton: document.getElementById("toggle-skeleton"),
  toggleMarkers: document.getElementById("toggle-markers"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x13201f);
scene.fog = new THREE.Fog(0x13201f, 18, 44);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 500);
camera.position.set(2.6, 1.7, 2.9);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
elements.viewport.appendChild(renderer.domElement);

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.target.set(0, 0.85, 0);

const transformControls = new TransformControls(camera, renderer.domElement);
transformControls.setSpace("local");
transformControls.setSize(1.08);
scene.add(transformControls.getHelper());

const ambientLight = new THREE.HemisphereLight(0xf8efe1, 0x243635, 1.35);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xffe1c5, 2.1);
keyLight.position.set(4, 7, 5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
scene.add(keyLight);

const rimLight = new THREE.DirectionalLight(0x80b8ff, 0.7);
rimLight.position.set(-5, 3, -4);
scene.add(rimLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(8, 80),
  new THREE.MeshStandardMaterial({
    color: 0x283a39,
    roughness: 0.95,
    metalness: 0.02,
    transparent: true,
    opacity: 0.9,
  }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.001;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(16, 48, 0xe0b373, 0x567371);
grid.material.opacity = 0.24;
grid.material.transparent = true;
scene.add(grid);

const markerGroup = new THREE.Group();
scene.add(markerGroup);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const loader = new GLTFLoader();
const exporter = new GLTFExporter();

const scratchBox = new THREE.Box3();
const scratchCenter = new THREE.Vector3();
const scratchSize = new THREE.Vector3();

const state = {
  assetRoot: null,
  sceneRoot: null,
  skeletonHelper: null,
  selectedBone: null,
  bones: [],
  boneSet: new Set(),
  initialPose: new Map(),
  markerRadius: 0.04,
  showGrid: true,
  showSkeleton: true,
  showMarkers: true,
  transformMode: "rotate",
  modelName: null,
};

function resizeRenderer() {
  const { clientWidth, clientHeight } = elements.viewport;
  if (!clientWidth || !clientHeight) {
    return;
  }
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight, false);
}

function animate() {
  requestAnimationFrame(animate);
  orbitControls.update();
  syncMarkers();
  renderer.render(scene, camera);
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.style.color = isError ? "#ffb39d" : "";
}

function clearObject(object) {
  if (!object) {
    return;
  }
  object.traverse((child) => {
    if (child.geometry) {
      child.geometry.dispose?.();
    }
    if (child.material) {
      if (Array.isArray(child.material)) {
        child.material.forEach((material) => material.dispose?.());
      } else {
        child.material.dispose?.();
      }
    }
  });
  object.removeFromParent();
}

function clearMarkers() {
  markerGroup.children.forEach((marker) => {
    marker.geometry?.dispose?.();
    if (Array.isArray(marker.material)) {
      marker.material.forEach((material) => material.dispose?.());
    } else {
      marker.material?.dispose?.();
    }
  });
  markerGroup.clear();
}

function normalizeBoneName(bone, index = 0) {
  const name = (bone.name || "").trim();
  return name || `joint_${index}`;
}

function isRootBone(bone) {
  return !bone.parent || !bone.parent.isBone || !state.boneSet.has(bone.parent);
}

function resetViewerState() {
  transformControls.detach();
  state.selectedBone = null;
  state.bones = [];
  state.boneSet = new Set();
  state.initialPose = new Map();

  clearObject(state.skeletonHelper);
  state.skeletonHelper = null;

  clearMarkers();
  clearObject(state.sceneRoot);
  state.sceneRoot = null;
  state.assetRoot = null;
  state.modelName = null;

  updateToolbar();
}

function createMarkerMaterial(isSelected = false) {
  return new THREE.MeshStandardMaterial({
    color: isSelected ? 0xffd49a : 0xb54f2f,
    emissive: isSelected ? 0x6a2d18 : 0x1e0b05,
    roughness: 0.36,
    metalness: 0.14,
  });
}

function rebuildMarkers() {
  clearMarkers();
  const geometry = new THREE.SphereGeometry(state.markerRadius, 18, 18);
  state.bones.forEach((bone, index) => {
    const marker = new THREE.Mesh(geometry, createMarkerMaterial(bone === state.selectedBone));
    marker.castShadow = true;
    marker.userData.bone = bone;
    marker.userData.label = normalizeBoneName(bone, index);
    marker.scale.setScalar(bone === state.selectedBone ? 1.45 : 1);
    markerGroup.add(marker);
  });
  markerGroup.visible = state.showMarkers;
}

function syncMarkers() {
  markerGroup.children.forEach((marker) => {
    marker.userData.bone.getWorldPosition(marker.position);
  });
}

function setInteractiveEnabled(enabled) {
  [
    elements.resetPose,
    elements.exportGlb,
    elements.modeRotate,
    elements.modeTranslate,
    elements.toggleSkeleton,
    elements.toggleMarkers,
  ].forEach((element) => {
    element.disabled = !enabled;
  });
  elements.resetJoint.disabled = !enabled || !state.selectedBone;
}

function syncTransformMode() {
  const translateAllowed = state.selectedBone && isRootBone(state.selectedBone);
  const mode = state.transformMode === "translate" && translateAllowed ? "translate" : "rotate";
  transformControls.setMode(mode);
  elements.modeRotate.classList.toggle("is-active", mode === "rotate");
  elements.modeTranslate.classList.toggle("is-active", mode === "translate");
  elements.modeTranslate.disabled = !state.assetRoot || !translateAllowed;
}

function updateToolbar() {
  const hasSelection = Boolean(state.selectedBone);
  setInteractiveEnabled(Boolean(state.assetRoot));
  syncTransformMode();

  grid.visible = state.showGrid;
  elements.toggleGrid.textContent = state.showGrid ? "Hide Grid" : "Show Grid";
  elements.toggleSkeleton.textContent = state.showSkeleton ? "Hide Bones" : "Show Bones";
  elements.toggleMarkers.textContent = state.showMarkers ? "Hide Markers" : "Show Markers";
  elements.resetJoint.disabled = !hasSelection;
}

function selectBone(bone, announce = true) {
  state.selectedBone = bone;
  if (bone) {
    transformControls.attach(bone);
    if (announce) {
      const rootNote = isRootBone(bone) ? " Root can also be moved." : "";
      setStatus(`Selected ${normalizeBoneName(bone)}. Drag the gizmo to pose.${rootNote}`);
    }
  } else {
    transformControls.detach();
  }
  rebuildMarkers();
  updateToolbar();
}

function fitCameraToAsset() {
  if (!state.sceneRoot) {
    orbitControls.target.set(0, 0.85, 0);
    camera.position.set(2.6, 1.7, 2.9);
    camera.updateProjectionMatrix();
    orbitControls.update();
    return;
  }

  scratchBox.setFromObject(state.sceneRoot);
  scratchBox.getCenter(scratchCenter);
  scratchBox.getSize(scratchSize);
  const maxDim = Math.max(scratchSize.x, scratchSize.y, scratchSize.z, 0.5);
  const fov = THREE.MathUtils.degToRad(camera.fov);
  let distance = (maxDim * 0.5) / Math.tan(fov * 0.5);
  distance *= 1.8;
  camera.position.copy(scratchCenter).add(new THREE.Vector3(1, 0.65, 1).normalize().multiplyScalar(distance));
  camera.near = Math.max(distance / 150, 0.01);
  camera.far = distance * 25;
  camera.updateProjectionMatrix();
  orbitControls.target.copy(scratchCenter);
  orbitControls.update();
}

function snapshotPose() {
  state.initialPose = new Map();
  state.bones.forEach((bone) => {
    state.initialPose.set(bone.uuid, {
      position: bone.position.clone(),
      quaternion: bone.quaternion.clone(),
      scale: bone.scale.clone(),
    });
  });
}

function resetBoneToInitialPose(bone) {
  const initial = state.initialPose.get(bone.uuid);
  if (!initial) {
    return;
  }
  bone.position.copy(initial.position);
  bone.quaternion.copy(initial.quaternion);
  bone.scale.copy(initial.scale);
}

function resetPose(selectedOnly = false) {
  if (!state.assetRoot) {
    return;
  }
  if (selectedOnly && state.selectedBone) {
    resetBoneToInitialPose(state.selectedBone);
    setStatus(`Reset ${normalizeBoneName(state.selectedBone)}.`);
  } else {
    state.bones.forEach((bone) => resetBoneToInitialPose(bone));
    setStatus("Reset the current pose.");
  }
  updateToolbar();
}

function collectBones(root) {
  const bones = [];
  const boneSet = new Set();
  const skinnedMeshes = [];

  root.traverse((object) => {
    if (object.isMesh) {
      object.castShadow = true;
      object.receiveShadow = true;
    }
    if (object.isSkinnedMesh) {
      skinnedMeshes.push(object);
      object.frustumCulled = false;
      object.skeleton.bones.forEach((bone) => {
        if (!boneSet.has(bone)) {
          boneSet.add(bone);
          bones.push(bone);
        }
      });
    }
  });

  return { bones, boneSet, skinnedMeshes };
}

function setupAsset(root, fileName) {
  resetViewerState();

  state.modelName = fileName.replace(/\.(glb|gltf)$/i, "");
  state.sceneRoot = new THREE.Group();
  state.sceneRoot.name = "RigViewerSceneRoot";
  state.sceneRoot.add(root);
  state.assetRoot = root;
  scene.add(state.sceneRoot);

  const { bones, boneSet, skinnedMeshes } = collectBones(root);
  state.bones = bones;
  state.boneSet = boneSet;
  snapshotPose();

  if (skinnedMeshes.length) {
    state.skeletonHelper = new THREE.SkeletonHelper(root);
    state.skeletonHelper.material.linewidth = 2;
    state.skeletonHelper.material.color.setHex(0xe0b373);
    state.skeletonHelper.visible = state.showSkeleton;
    scene.add(state.skeletonHelper);
  }

  scratchBox.setFromObject(root);
  scratchBox.getSize(scratchSize);
  const maxDim = Math.max(scratchSize.x, scratchSize.y, scratchSize.z, 0.4);
  state.markerRadius = THREE.MathUtils.clamp(maxDim * 0.02, 0.016, 0.11);

  rebuildMarkers();
  fitCameraToAsset();
  updateToolbar();

  if (skinnedMeshes.length && state.bones.length) {
    selectBone(state.bones[0], false);
    setStatus(`Loaded ${fileName}. Click a joint marker and drag the gizmo to pose.`);
  } else {
    selectBone(null, false);
    setStatus(`Loaded ${fileName}, but it does not contain a skinned mesh. Use AniGen's mesh.glb, not skeleton.glb.`, true);
  }
}

function loadFile(file) {
  if (!file) {
    return;
  }
  const lowerName = file.name.toLowerCase();
  if (!lowerName.endsWith(".glb") && !lowerName.endsWith(".gltf")) {
    setStatus("Unsupported file type. Load a .glb or .gltf file.", true);
    return;
  }

  setStatus(`Loading ${file.name}...`);
  file.arrayBuffer().then((buffer) => {
    loader.parse(
      buffer,
      "",
      (gltf) => {
        const root = gltf.scene || gltf.scenes?.[0];
        if (!root) {
          setStatus("The file loaded, but it did not contain a scene.", true);
          return;
        }
        setupAsset(root, file.name);
      },
      (error) => {
        console.error(error);
        setStatus(`Failed to load ${file.name}. Check the browser console for details.`, true);
      },
    );
  }).catch((error) => {
    console.error(error);
    setStatus(`Could not read ${file.name}.`, true);
  });
}

function handlePointerSelect(event) {
  if (!state.assetRoot || !markerGroup.children.length || transformControls.dragging) {
    return;
  }
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(markerGroup.children, false);
  if (hits.length) {
    selectBone(hits[0].object.userData.bone);
  }
}

function exportCurrentPose() {
  if (!state.assetRoot) {
    return;
  }

  setStatus("Exporting posed GLB...");
  exporter.parse(
    state.assetRoot,
    (result) => {
      const blob = result instanceof ArrayBuffer
        ? new Blob([result], { type: "model/gltf-binary" })
        : new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${state.modelName || "posed-mesh"}-posed.${result instanceof ArrayBuffer ? "glb" : "gltf"}`;
      anchor.click();
      URL.revokeObjectURL(url);
      setStatus("Exported posed GLB.");
    },
    (error) => {
      console.error(error);
      setStatus("Export failed. Check the browser console for details.", true);
    },
    { binary: true, onlyVisible: true },
  );
}

function bindInputs() {
  elements.fileInput.addEventListener("change", (event) => {
    loadFile(event.target.files?.[0]);
    event.target.value = "";
  });

  ["dragenter", "dragover"].forEach((type) => {
    elements.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      elements.dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((type) => {
    elements.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      elements.dropzone.classList.remove("is-dragover");
    });
  });

  elements.dropzone.addEventListener("drop", (event) => {
    loadFile(event.dataTransfer?.files?.[0]);
  });

  elements.fitCamera.addEventListener("click", fitCameraToAsset);
  elements.resetPose.addEventListener("click", () => resetPose(false));
  elements.resetJoint.addEventListener("click", () => resetPose(true));
  elements.exportGlb.addEventListener("click", exportCurrentPose);

  elements.modeRotate.addEventListener("click", () => {
    state.transformMode = "rotate";
    syncTransformMode();
    setStatus("Rotation mode enabled.");
  });

  elements.modeTranslate.addEventListener("click", () => {
    if (!state.selectedBone || !isRootBone(state.selectedBone)) {
      setStatus("Only a root joint can be moved.", true);
      return;
    }
    state.transformMode = "translate";
    syncTransformMode();
    setStatus(`Move-root mode enabled for ${normalizeBoneName(state.selectedBone)}.`);
  });

  elements.toggleGrid.addEventListener("click", () => {
    state.showGrid = !state.showGrid;
    updateToolbar();
  });

  elements.toggleSkeleton.addEventListener("click", () => {
    state.showSkeleton = !state.showSkeleton;
    if (state.skeletonHelper) {
      state.skeletonHelper.visible = state.showSkeleton;
    }
    updateToolbar();
  });

  elements.toggleMarkers.addEventListener("click", () => {
    state.showMarkers = !state.showMarkers;
    markerGroup.visible = state.showMarkers;
    updateToolbar();
  });

  transformControls.addEventListener("dragging-changed", (event) => {
    orbitControls.enabled = !event.value;
  });

  transformControls.addEventListener("mouseDown", () => {
    if (state.selectedBone) {
      setStatus(`Dragging ${normalizeBoneName(state.selectedBone)}...`);
    }
  });

  transformControls.addEventListener("mouseUp", () => {
    if (state.selectedBone) {
      setStatus(`Updated ${normalizeBoneName(state.selectedBone)}.`);
    }
  });

  renderer.domElement.addEventListener("pointerdown", handlePointerSelect);
  window.addEventListener("resize", resizeRenderer);
}

resizeRenderer();
bindInputs();
updateToolbar();
fitCameraToAsset();
animate();
