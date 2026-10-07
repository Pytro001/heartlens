// The 3D heart: BodyParts3D anatomy (CC BY-SA 2.1 JP), coronary vessels painted by predicted probability.
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { TargetKey } from './types'

export type VesselKey = 'LAD' | 'LCX' | 'RCA'
const VESSEL_KEYS: VesselKey[] = ['LAD', 'LCX', 'RCA']

/** green (low) to amber to red (high) */
export function riskColor(p: number, out = new THREE.Color()): THREE.Color {
  const stops: [number, number, number, number][] = [
    [0.0, 0.16, 0.85, 0.52], [0.35, 0.62, 0.86, 0.25], [0.55, 1.0, 0.72, 0.12], [0.75, 1.0, 0.38, 0.1], [1.0, 0.95, 0.12, 0.16],
  ]
  const q = Math.min(1, Math.max(0, p))
  for (let i = 1; i < stops.length; i++) {
    if (q <= stops[i][0]) {
      const a = stops[i - 1], b = stops[i], f = (q - a[0]) / (b[0] - a[0])
      return out.setRGB(a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f, THREE.SRGBColorSpace)
    }
  }
  return out.setRGB(0.95, 0.12, 0.16, THREE.SRGBColorSpace)
}
export function riskCss(p: number): string { return '#' + riskColor(p).getHexString(THREE.SRGBColorSpace) }

export interface Pose { az: number; el: number; dist: number; tx: number; ty: number; tz: number }
export const POSES: Record<string, Pose> = {
  front: { az: 0.15, el: 0.12, dist: 5.0, tx: 0, ty: 0.2, tz: 0 },
  hero: { az: -0.35, el: 0.16, dist: 5.2, tx: 0, ty: 0.2, tz: 0 },
  LAD: { az: 0.5, el: 0.1, dist: 4.0, tx: 0.1, ty: 0.12, tz: 0.08 },
  LCX: { az: 1.75, el: 0.3, dist: 4.1, tx: 0.15, ty: 0.1, tz: -0.05 },
  RCA: { az: -0.8, el: 0.05, dist: 4.1, tx: -0.1, ty: 0.12, tz: 0.08 },
  wide: { az: 0.3, el: 0.2, dist: 6.2, tx: 0, ty: 0.22, tz: 0 },
}

interface Vessel {
  key: VesselKey; mesh: THREE.Mesh; glow: THREE.Mesh; colors: THREE.BufferAttribute
  anchor: THREE.Vector3; anchorNormal: THREE.Vector3
  p: number; shown: number; selected: number; hover: number
}

const flowUniforms = { uTime: { value: 0 } }

export class HeartView {
  renderer: THREE.WebGLRenderer
  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50)
  controls: OrbitControls
  root = new THREE.Group()
  vessels = new Map<VesselKey, Vessel>()
  wallMat!: THREE.MeshPhysicalMaterial
  leftMainMat!: THREE.MeshStandardMaterial
  cad = -1; cadShown = 0
  active = 0; activeShown = 0 // 0 = no patient (grey), 1 = painted
  time = 0
  autoRotate = true
  lastInteract = -999
  scripted = false
  pose: Pose = { ...POSES.hero }
  private ray = new THREE.Raycaster()
  private hoverKey: VesselKey | null = null
  onPick: (k: VesselKey) => void = () => {}

  constructor(private host: HTMLElement) {
    const r = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' })
    r.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
    r.toneMapping = THREE.ACESFilmicToneMapping
    r.toneMappingExposure = 1.05
    r.outputColorSpace = THREE.SRGBColorSpace
    r.shadowMap.enabled = true
    r.shadowMap.type = THREE.PCFShadowMap
    host.appendChild(r.domElement)
    r.domElement.setAttribute('aria-label', '3D heart. Use the vessel list to select arteries with the keyboard.')
    r.domElement.setAttribute('role', 'img')
    this.renderer = r
    const pm = new THREE.PMREMGenerator(r)
    this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture
    this.scene.environmentIntensity = 0.45
    this.scene.add(this.root)

    const key = new THREE.DirectionalLight(0xfff1e6, 2.4)
    key.position.set(2.2, 3.2, 3.0)
    key.castShadow = true
    key.shadow.mapSize.set(1024, 1024)
    key.shadow.camera.left = -1.6; key.shadow.camera.right = 1.6
    key.shadow.camera.top = 1.6; key.shadow.camera.bottom = -1.6
    key.shadow.radius = 6; key.shadow.bias = -0.0008
    this.scene.add(key)
    const rim = new THREE.DirectionalLight(0x8fb6ff, 2.2)
    rim.position.set(-2.5, 1.2, -3.0)
    this.scene.add(rim)
    const fill = new THREE.DirectionalLight(0xffd2c8, 0.6)
    fill.position.set(-3, -0.5, 2)
    this.scene.add(fill)
    this.scene.add(new THREE.HemisphereLight(0x9fb4d8, 0x1a0f12, 0.5))

    const ground = new THREE.Mesh(new THREE.CircleGeometry(3, 64), new THREE.ShadowMaterial({ opacity: 0.22 }))
    ground.rotation.x = -Math.PI / 2
    ground.position.y = -0.98
    ground.receiveShadow = true
    this.scene.add(ground)

    this.controls = new OrbitControls(this.camera, r.domElement)
    this.controls.enableDamping = true
    this.controls.minDistance = 1.6
    this.controls.maxDistance = 7
    this.controls.enablePan = false
    this.controls.addEventListener('start', () => { this.lastInteract = this.time })
    this.applyPose(this.pose)

    r.domElement.addEventListener('pointermove', e => this.onMove(e))
    r.domElement.addEventListener('click', e => { const k = this.pick(e); if (k) this.onPick(k) })
    new ResizeObserver(() => this.resize()).observe(host)
    this.resize()
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight
    if (!w || !h) return
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  async load(url: string) {
    const gltf = await new GLTFLoader().loadAsync(url)
    const byName = new Map<string, THREE.Mesh>()
    gltf.scene.traverse(o => { if ((o as THREE.Mesh).isMesh) byName.set(o.name, o as THREE.Mesh) })
    const prep = (name: string) => {
      const m = byName.get(name)
      if (!m) throw new Error('missing mesh ' + name)
      let g = m.geometry.clone()
      g.deleteAttribute('normal')
      g = mergeVertices(g, 1e-5)
      g.computeVertexNormals()
      return g
    }
    this.wallMat = new THREE.MeshPhysicalMaterial({
      color: 0x9a6a66, roughness: 0.55, metalness: 0, clearcoat: 0.55, clearcoatRoughness: 0.32,
      sheen: 0.7, sheenColor: new THREE.Color(0xff9a8a), sheenRoughness: 0.5,
    })
    const wall = new THREE.Mesh(prep('heart_wall'), this.wallMat)
    wall.castShadow = true; wall.receiveShadow = true
    this.root.add(wall)
    const vessel = (name: string, color: number, rough = 0.42) => {
      const m = new THREE.Mesh(prep(name), new THREE.MeshPhysicalMaterial({ color, roughness: rough, clearcoat: 0.5, clearcoatRoughness: 0.3 }))
      m.castShadow = true; m.receiveShadow = true
      this.root.add(m)
      return m
    }
    vessel('aorta', 0xa8625c)
    vessel('svc', 0x5d6496)
    vessel('coronary_sinus', 0x6a6aa8)
    const lm = vessel('left_main', 0x8a8f99)
    this.leftMainMat = lm.material as THREE.MeshStandardMaterial
    for (const k of VESSEL_KEYS) this.vessels.set(k, this.makeVessel(k, prep(k)))
  }

  private makeVessel(key: VesselKey, g: THREE.BufferGeometry): Vessel {
    const pos = g.getAttribute('position') as THREE.BufferAttribute
    const nrm = g.getAttribute('normal') as THREE.BufferAttribute
    const n = pos.count
    // inflate a little so the vessels read at laptop distance
    for (let i = 0; i < n; i++) {
      pos.setXYZ(i, pos.getX(i) + nrm.getX(i) * 0.005, pos.getY(i) + nrm.getY(i) * 0.005, pos.getZ(i) + nrm.getZ(i) * 0.005)
    }
    // distance from the ostium (most superior vertex) drives the flow pulse
    let top = 0
    for (let i = 1; i < n; i++) if (pos.getY(i) > pos.getY(top)) top = i
    const o = new THREE.Vector3().fromBufferAttribute(pos, top)
    const d = new Float32Array(n)
    let dmax = 0
    const v = new THREE.Vector3()
    for (let i = 0; i < n; i++) { d[i] = v.fromBufferAttribute(pos, i).distanceTo(o); dmax = Math.max(dmax, d[i]) }
    for (let i = 0; i < n; i++) d[i] /= dmax
    g.setAttribute('aDist', new THREE.BufferAttribute(d, 1))
    const colors = new THREE.BufferAttribute(new Float32Array(n * 3), 3)
    g.setAttribute('color', colors)
    // label anchor: a vertex on the visible course of each artery (anterior groove for LAD,
    // left lateral AV groove for LCX, right anterior AV groove for RCA)
    const dir = { LAD: [0.35, 0, 1], LCX: [1, 0.15, -0.2], RCA: [-1, 0, 0.6] }[key]
    const yAim = { LAD: -0.05, LCX: 0.05, RCA: 0.0 }[key]
    let best = 0, bestScore = -1e9
    for (let i = 0; i < n; i++) {
      v.fromBufferAttribute(pos, i)
      const score = v.x * dir[0] + v.y * dir[1] + v.z * dir[2] - Math.abs(v.y - yAim) * 1.2
      if (score > bestScore) { bestScore = score; best = i }
    }
    const mat = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.32, clearcoat: 0.8, clearcoatRoughness: 0.2, emissive: 0xffffff, emissiveIntensity: 0 })
    mat.onBeforeCompile = sh => {
      sh.uniforms.uTime = flowUniforms.uTime
      sh.uniforms.uGlow = { value: 0 }
      mat.userData.shader = sh
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nattribute float aDist;\nvarying float vDist;')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvDist = aDist;')
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\nuniform float uTime;\nuniform float uGlow;\nvarying float vDist;')
        .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
          float band = smoothstep(0.0, 0.18, fract(vDist * 2.2 - uTime * 0.45)) * (1.0 - smoothstep(0.18, 0.32, fract(vDist * 2.2 - uTime * 0.45)));
          totalEmissiveRadiance = vColor.rgb * uGlow * (0.55 + 0.9 * band);`)
    }
    const mesh = new THREE.Mesh(g, mat)
    mesh.castShadow = true
    mesh.userData.vessel = key
    this.root.add(mesh)
    const glowMat = new THREE.ShaderMaterial({
      uniforms: { uColor: { value: new THREE.Color() }, uOpacity: { value: 0 } },
      vertexShader: `varying vec3 vN; varying vec3 vV;
        void main(){ vec4 mv = modelViewMatrix * vec4(position + normal * 0.012, 1.0); vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix * mv; }`,
      fragmentShader: `uniform vec3 uColor; uniform float uOpacity; varying vec3 vN; varying vec3 vV;
        void main(){ float f = pow(1.0 - abs(dot(vN, vV)), 2.0); gl_FragColor = vec4(uColor * f * uOpacity, 1.0); }`,
      transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
    })
    const glow = new THREE.Mesh(g, glowMat)
    glow.userData.vessel = key
    glow.renderOrder = 2
    this.root.add(glow)
    return { key, mesh, glow, colors, anchor: new THREE.Vector3().fromBufferAttribute(pos, best), anchorNormal: new THREE.Vector3().fromBufferAttribute(nrm, best), p: 0, shown: 0, selected: 0, hover: 0 }
  }

  setRisk(p: Partial<Record<TargetKey, number>> | null) {
    if (!p) { this.active = 0; return }
    this.active = 1
    for (const k of VESSEL_KEYS) if (p[k] != null) this.vessels.get(k)!.p = p[k]!
    if (p.Cath != null) this.cad = p.Cath
  }

  selected: VesselKey | null = null
  select(k: VesselKey | null) { this.selected = k }

  private pick(e: PointerEvent | MouseEvent): VesselKey | null {
    const rect = this.renderer.domElement.getBoundingClientRect()
    const ndc = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1)
    this.ray.setFromCamera(ndc, this.camera)
    const targets = [...this.vessels.values()].flatMap(v => [v.mesh, v.glow])
    const hits = this.ray.intersectObjects(targets, false)
    return hits.length ? (hits[0].object.userData.vessel as VesselKey) : null
  }
  private onMove(e: PointerEvent) {
    this.hoverKey = this.pick(e)
    this.renderer.domElement.style.cursor = this.hoverKey ? 'pointer' : 'grab'
  }

  applyPose(p: Pose) {
    this.pose = { ...p }
    const t = new THREE.Vector3(p.tx, p.ty, p.tz)
    this.camera.position.set(
      t.x + p.dist * Math.cos(p.el) * Math.sin(p.az), t.y + p.dist * Math.sin(p.el), t.z + p.dist * Math.cos(p.el) * Math.cos(p.az))
    this.controls.target.copy(t)
    this.camera.lookAt(t)
  }

  /** advance animation state by dt seconds (sim time, deterministic in film mode) */
  update(dt: number) {
    this.time += dt
    flowUniforms.uTime.value = this.time
    const k = 1 - Math.exp(-dt * 5)
    this.activeShown += (this.active - this.activeShown) * k
    if (this.cad >= 0) this.cadShown += (this.cad - this.cadShown) * k
    const grey = new THREE.Color(0x7d828c), col = new THREE.Color(), tmp = new THREE.Color()
    for (const v of this.vessels.values()) {
      v.shown += (v.p - v.shown) * k
      v.selected += ((this.selected === v.key ? 1 : 0) - v.selected) * k
      v.hover += ((this.hoverKey === v.key ? 1 : 0) - v.hover) * (1 - Math.exp(-dt * 10))
      riskColor(v.shown, tmp)
      col.copy(grey).lerp(tmp, this.activeShown)
      const arr = v.colors.array as Float32Array
      for (let i = 0; i < arr.length; i += 3) { arr[i] = col.r; arr[i + 1] = col.g; arr[i + 2] = col.b }
      v.colors.needsUpdate = true
      const sh = (v.mesh.material as THREE.Material).userData.shader
      const dim = this.selected && this.selected !== v.key ? 0.45 : 1
      if (sh) sh.uniforms.uGlow.value = this.activeShown * (0.25 + 0.75 * v.shown) * dim + v.hover * 0.3
      const gm = v.glow.material as THREE.ShaderMaterial
      gm.uniforms.uColor.value.copy(col)
      gm.uniforms.uOpacity.value = this.activeShown * (0.35 + 1.4 * v.shown * v.shown) * dim + v.selected * 0.9 + v.hover * 0.6
    }
    // heart muscle tinted subtly by overall CAD probability
    const calm = new THREE.Color(0x9a6a66), hot = new THREE.Color(0x96433d)
    this.wallMat.color.copy(calm).lerp(hot, this.activeShown * Math.max(0, this.cadShown))
    const lm = (this.vessels.get('LAD')!.shown + this.vessels.get('LCX')!.shown) / 2
    this.leftMainMat.color.copy(grey).lerp(riskColor(lm, tmp), this.activeShown * 0.85)
    // gentle heartbeat, about 64 bpm
    const ph = (this.time * 64 / 60) % 1
    const beat = 1 + 0.012 * Math.exp(-Math.pow((ph - 0.08) / 0.05, 2)) + 0.006 * Math.exp(-Math.pow((ph - 0.3) / 0.06, 2))
    this.root.scale.setScalar(beat)
    if (!this.scripted) {
      this.controls.autoRotate = this.autoRotate && this.time - this.lastInteract > 6
      this.controls.autoRotateSpeed = 0.55
      this.controls.update(dt)
    }
  }

  render() { this.renderer.render(this.scene, this.camera) }

  /** screen space anchors for the HTML labels */
  anchors(): { key: VesselKey; x: number; y: number; facing: number }[] {
    const w = this.host.clientWidth, h = this.host.clientHeight
    const out: { key: VesselKey; x: number; y: number; facing: number }[] = []
    const camDir = new THREE.Vector3()
    for (const v of this.vessels.values()) {
      const p = v.anchor.clone().multiply(this.root.scale)
      camDir.copy(this.camera.position).sub(p).normalize()
      const facing = v.anchorNormal.dot(camDir)
      p.project(this.camera)
      out.push({ key: v.key, x: (p.x * 0.5 + 0.5) * w, y: (-p.y * 0.5 + 0.5) * h, facing })
    }
    return out
  }
  centerScreen() {
    const p = new THREE.Vector3(0, 0, 0).project(this.camera)
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0).multiplyScalar(0.62)
    const q = right.project(this.camera)
    const w = this.host.clientWidth
    return { x: (p.x * 0.5 + 0.5) * w, y: (-p.y * 0.5 + 0.5) * this.host.clientHeight, r: Math.abs(q.x - p.x) * 0.5 * w }
  }
}
