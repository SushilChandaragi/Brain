import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import '../ThreeNeuralVisualizer.css';

export interface LayerConfig {
  id: string;
  name: string;
  neuronCount: number;
  activation: 'SiLU' | 'GELU' | 'ReLU' | 'RMSNorm' | 'Linear' | 'Softmax';
}

const ACTIVATION_CYCLE: LayerConfig['activation'][] = [
  'SiLU',
  'GELU',
  'ReLU',
  'RMSNorm',
  'Linear',
  'Softmax',
];

interface NodeData {
  id: string;
  layerIdx: number;
  nodeIdx: number;
  x: number;
  y: number;
  z: number;
  baseActivation: number;
  currentActivation: number;
}

interface SynapseData {
  id: string;
  sourceId: string;
  targetId: string;
  sourceLayer: number;
  weight: number;
  sourcePos: THREE.Vector3;
  targetPos: THREE.Vector3;
}

export const ThreeNeuralVisualizer: React.FC = () => {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Modular Layer Configuration
  const [layers, setLayers] = useState<LayerConfig[]>([
    { id: 'l0', name: 'Layer 1 (Input)', neuronCount: 6, activation: 'Linear' },
    { id: 'l1', name: 'Layer 2 (Syntax)', neuronCount: 8, activation: 'SiLU' },
    { id: 'l2', name: 'Layer 3 (Hidden)', neuronCount: 10, activation: 'GELU' },
    { id: 'l3', name: 'Layer 4 (Deep)', neuronCount: 8, activation: 'RMSNorm' },
    { id: 'l4', name: 'Layer 5 (Output)', neuronCount: 6, activation: 'Softmax' },
  ]);

  // Simulation & View State
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [autoRotate, setAutoRotate] = useState<boolean>(false);
  const [propagationSpeed, setPropagationSpeed] = useState<number>(1.2);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    node: NodeData | null;
  }>({
    visible: false,
    x: 0,
    y: 0,
    node: null,
  });

  // Dynamic Metrics
  const [fps, setFps] = useState<number>(60);
  const [waveProgress, setWaveProgress] = useState<number>(0);

  // Mesh & Object Registries
  const nodeMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const haloMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const synapseLinesRef = useRef<Map<string, THREE.Line>>(new Map());
  const pulseParticlesRef = useRef<THREE.Points | null>(null);
  const layerPlanesGroupRef = useRef<THREE.Group>(new THREE.Group());
  const networkGroupRef = useRef<THREE.Group>(new THREE.Group());

  // Derive Nodes & Synapses Topology
  const { nodes, synapses, nodeMap } = useMemo(() => {
    const nList: NodeData[] = [];
    const nMap = new Map<string, NodeData>();
    const layerSpacing = 5.2;
    const totalLayers = layers.length;

    layers.forEach((layer, lIdx) => {
      const x = (lIdx - (totalLayers - 1) / 2) * layerSpacing;
      const count = layer.neuronCount;
      const verticalSpan = Math.min(10, Math.max(5, count * 0.95));
      const step = count > 1 ? verticalSpan / (count - 1) : 0;

      for (let i = 0; i < count; i++) {
        const y = count > 1 ? verticalSpan / 2 - i * step : 0;
        const z = (Math.sin((i / count) * Math.PI) * 0.4) - 0.2;
        const id = `l${lIdx}_n${i}`;
        // Pseudo-random deterministic activation
        const pseudoAct = 0.35 + 0.6 * Math.sin(lIdx * 2.3 + i * 1.7);
        const node: NodeData = {
          id,
          layerIdx: lIdx,
          nodeIdx: i,
          x,
          y,
          z,
          baseActivation: Number(pseudoAct.toFixed(3)),
          currentActivation: Number(pseudoAct.toFixed(3)),
        };
        nList.push(node);
        nMap.set(id, node);
      }
    });

    const sList: SynapseData[] = [];
    for (let l = 0; l < totalLayers - 1; l++) {
      const srcNodes = nList.filter((n) => n.layerIdx === l);
      const tgtNodes = nList.filter((n) => n.layerIdx === l + 1);

      srcNodes.forEach((src) => {
        tgtNodes.forEach((tgt) => {
          const w = 0.2 + 0.8 * Math.abs(Math.sin(src.nodeIdx * 3.1 + tgt.nodeIdx * 4.7));
          sList.push({
            id: `${src.id}->${tgt.id}`,
            sourceId: src.id,
            targetId: tgt.id,
            sourceLayer: l,
            weight: Number(w.toFixed(3)),
            sourcePos: new THREE.Vector3(src.x, src.y, src.z),
            targetPos: new THREE.Vector3(tgt.x, tgt.y, tgt.z),
          });
        });
      });
    }

    return { nodes: nList, synapses: sList, nodeMap: nMap };
  }, [layers]);

  // Selected Node Telemetry Info
  const selectedNode = useMemo(() => {
    if (!selectedNodeId) return null;
    return nodeMap.get(selectedNodeId) || null;
  }, [selectedNodeId, nodeMap]);

  const selectedSynapses = useMemo(() => {
    if (!selectedNodeId) return [];
    return synapses.filter(
      (s) => s.sourceId === selectedNodeId || s.targetId === selectedNodeId
    );
  }, [selectedNodeId, synapses]);

  // Initialize Three.js Scene
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b132b);
    scene.fog = new THREE.FogExp2(0x0b132b, 0.028);
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 6, 22);
    cameraRef.current = camera;

    // 3. Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = false;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 38;
    controls.minDistance = 6;
    controls.maxPolarAngle = Math.PI / 2 - 0.02; // Keep above floor
    controlsRef.current = controls;

    // 5. Lights
    const ambientLight = new THREE.AmbientLight(0x334e68, 1.4);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(10, 20, 15);
    scene.add(dirLight);

    const amberLight = new THREE.PointLight(0xffd166, 2.0, 30);
    amberLight.position.set(0, 4, 8);
    scene.add(amberLight);

    // 6. 3D Sci-Fi Floor Perspective Grid
    const floorY = -6.2;
    const gridHelper = new THREE.GridHelper(60, 40, 0x2a4365, 0x102a43);
    gridHelper.position.y = floorY;
    scene.add(gridHelper);

    // Ground Plane with subtle reflection
    const groundGeo = new THREE.PlaneGeometry(80, 80);
    const groundMat = new THREE.MeshBasicMaterial({
      color: 0x070c1a,
      transparent: true,
      opacity: 0.85,
      side: THREE.DoubleSide,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = floorY - 0.05;
    scene.add(ground);

    // Add groups
    scene.add(layerPlanesGroupRef.current);
    scene.add(networkGroupRef.current);

    // Resize Handler
    const handleResize = () => {
      if (!container || !rendererRef.current || !cameraRef.current) return;
      const newW = container.clientWidth;
      const newH = container.clientHeight;
      cameraRef.current.aspect = newW / newH;
      cameraRef.current.updateProjectionMatrix();
      rendererRef.current.setSize(newW, newH);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      renderer.dispose();
      container.innerHTML = '';
    };
  }, []);

  // Build & Rebuild 3D Layers, Glass Panes, Nodes, and Synapses
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    // Clear previous objects
    const planesGroup = layerPlanesGroupRef.current;
    const netGroup = networkGroupRef.current;
    while (planesGroup.children.length > 0) planesGroup.remove(planesGroup.children[0]);
    while (netGroup.children.length > 0) netGroup.remove(netGroup.children[0]);

    nodeMeshesRef.current.clear();
    haloMeshesRef.current.clear();
    synapseLinesRef.current.clear();

    const layerSpacing = 5.2;
    const totalLayers = layers.length;

    // 1. Create Glassmorphic Layer Panes
    layers.forEach((layer, lIdx) => {
      const x = (lIdx - (totalLayers - 1) / 2) * layerSpacing;
      const count = layer.neuronCount;
      const height = Math.max(8, count * 1.1 + 2);
      const width = 2.6;

      // Translucent pane
      const planeGeo = new THREE.PlaneGeometry(width, height);
      const planeMat = new THREE.MeshBasicMaterial({
        color: 0x1c2541,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const planeMesh = new THREE.Mesh(planeGeo, planeMat);
      planeMesh.position.set(x, 0, 0);

      // Glowing border frame
      const edges = new THREE.EdgesGeometry(planeGeo);
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x3b82f6,
        transparent: true,
        opacity: 0.45,
      });
      const wireframe = new THREE.LineSegments(edges, lineMat);
      planeMesh.add(wireframe);

      planesGroup.add(planeMesh);
    });

    // 2. Create Synapse Web Lines
    synapses.forEach((syn) => {
      const points = [syn.sourcePos, syn.targetPos];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
      const baseOpacity = 0.06 + syn.weight * 0.24;
      const lineMat = new THREE.LineBasicMaterial({
        color: 0xe2e8f0,
        transparent: true,
        opacity: baseOpacity,
        depthWrite: false,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      line.userData = { synapseId: syn.id, ...syn, baseOpacity };
      netGroup.add(line);
      synapseLinesRef.current.set(syn.id, line);
    });

    // 3. Create Glowing Nodes
    const sphereGeo = new THREE.SphereGeometry(0.22, 24, 24);
    const ringGeo = new THREE.RingGeometry(0.28, 0.4, 24);

    nodes.forEach((node) => {
      // Node core
      const nodeMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        emissive: 0x94a3b8,
        emissiveIntensity: 0.6,
        roughness: 0.2,
        metalness: 0.8,
      });
      const nodeMesh = new THREE.Mesh(sphereGeo, nodeMat);
      nodeMesh.position.set(node.x, node.y, node.z);
      nodeMesh.userData = { nodeId: node.id, ...node };

      // Outer glowing halo ring
      const haloMat = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const halo = new THREE.Mesh(ringGeo, haloMat);
      halo.position.copy(nodeMesh.position);
      halo.position.z += 0.02;

      netGroup.add(nodeMesh);
      netGroup.add(halo);

      nodeMeshesRef.current.set(node.id, nodeMesh);
      haloMeshesRef.current.set(node.id, halo);
    });

    // 4. Create Pulse Wave Particles
    const particleCount = 200;
    const particleGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const alphas = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = 0;
      positions[i * 3 + 1] = 0;
      positions[i * 3 + 2] = 0;
      alphas[i] = 0;
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particleMat = new THREE.PointsMaterial({
      color: 0xffd166,
      size: 0.32,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    const particles = new THREE.Points(particleGeo, particleMat);
    netGroup.add(particles);
    pulseParticlesRef.current = particles;
  }, [layers, nodes, synapses]);

  // Highlight Active Nodes & Synapses when Hovered or Selected
  useEffect(() => {
    const activeId = hoveredNodeId || selectedNodeId;

    nodeMeshesRef.current.forEach((mesh, id) => {
      const halo = haloMeshesRef.current.get(id);
      const isTarget = id === activeId;
      const isConnected =
        activeId &&
        synapses.some(
          (s) =>
            (s.sourceId === activeId && s.targetId === id) ||
            (s.targetId === activeId && s.sourceId === id)
        );

      const mat = mesh.material as THREE.MeshStandardMaterial;
      const hMat = halo ? (halo.material as THREE.MeshBasicMaterial) : null;

      if (isTarget) {
        mat.color.setHex(0xffd166);
        mat.emissive.setHex(0xffd166);
        mat.emissiveIntensity = 1.6;
        mesh.scale.set(1.4, 1.4, 1.4);
        if (hMat && halo) {
          hMat.color.setHex(0xffd166);
          hMat.opacity = 0.9;
          halo.scale.set(1.5, 1.5, 1.5);
        }
      } else if (isConnected) {
        mat.color.setHex(0xffe082);
        mat.emissive.setHex(0xffd166);
        mat.emissiveIntensity = 1.0;
        mesh.scale.set(1.2, 1.2, 1.2);
        if (hMat && halo) {
          hMat.color.setHex(0xffe082);
          hMat.opacity = 0.7;
          halo.scale.set(1.25, 1.25, 1.25);
        }
      } else {
        mat.color.setHex(0xffffff);
        mat.emissive.setHex(activeId ? 0x223344 : 0x94a3b8);
        mat.emissiveIntensity = activeId ? 0.2 : 0.6;
        mesh.scale.set(1.0, 1.0, 1.0);
        if (hMat && halo) {
          hMat.color.setHex(0xffffff);
          hMat.opacity = activeId ? 0.08 : 0.4;
          halo.scale.set(1.0, 1.0, 1.0);
        }
      }
    });

    // Update Synapse Lines
    synapseLinesRef.current.forEach((line) => {
      const syn = line.userData as SynapseData & { baseOpacity: number };
      const lineMat = line.material as THREE.LineBasicMaterial;

      if (!activeId) {
        lineMat.color.setHex(0xe2e8f0);
        lineMat.opacity = syn.baseOpacity;
      } else if (syn.sourceId === activeId || syn.targetId === activeId) {
        lineMat.color.setHex(0xffd166);
        lineMat.opacity = 0.95;
      } else {
        lineMat.color.setHex(0x1e293b);
        lineMat.opacity = 0.03;
      }
    });
  }, [hoveredNodeId, selectedNodeId, synapses]);

  // Raycasting for Mouse Interaction
  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      const container = mountRef.current;
      const camera = cameraRef.current;
      if (!container || !camera) return;

      const rect = container.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(x, y), camera);

      const meshes = Array.from(nodeMeshesRef.current.values());
      const intersects = raycaster.intersectObjects(meshes, false);

      if (intersects.length > 0) {
        const hitMesh = intersects[0].object as THREE.Mesh;
        const nodeId = hitMesh.userData.nodeId;
        setHoveredNodeId(nodeId);
        const node = nodeMap.get(nodeId) || null;
        setTooltip({
          visible: true,
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          node,
        });
      } else {
        setHoveredNodeId(null);
        setTooltip((prev) => ({ ...prev, visible: false }));
      }
    },
    [nodeMap]
  );

  const handleClick = useCallback(() => {
    if (hoveredNodeId) {
      setSelectedNodeId((prev) => (prev === hoveredNodeId ? null : hoveredNodeId));
    }
  }, [hoveredNodeId]);

  // Main 3D Animation Loop (Pulse Waves, OrbitControls, Camera)
  useEffect(() => {
    let lastTime = performance.now();
    let frameCount = 0;
    let fpsTimer = performance.now();
    let progress = 0;

    const animate = (currentTime: number) => {
      animFrameRef.current = requestAnimationFrame(animate);

      const delta = (currentTime - lastTime) / 1000;
      lastTime = currentTime;

      // FPS Counter
      frameCount++;
      if (currentTime - fpsTimer >= 500) {
        setFps(Math.round((frameCount * 1000) / (currentTime - fpsTimer)));
        frameCount = 0;
        fpsTimer = currentTime;
      }

      // Update Controls
      if (controlsRef.current) {
        controlsRef.current.autoRotate = autoRotate;
        controlsRef.current.autoRotateSpeed = 0.8;
        controlsRef.current.update();
      }

      // Real-time Pulse Wave Propagation
      if (isPlaying && synapses.length > 0) {
        progress = (progress + delta * propagationSpeed * 0.4) % 1.0;
        setWaveProgress(progress);

        // Animate pulse particles along synapse trajectories
        const particles = pulseParticlesRef.current;
        if (particles) {
          const positions = particles.geometry.attributes.position.array as Float32Array;
          const count = positions.length / 3;

          for (let i = 0; i < count; i++) {
            const syn = synapses[i % synapses.length];
            const pPhase = (progress + (i / count) * 0.8) % 1.0;
            const pos = new THREE.Vector3().lerpVectors(syn.sourcePos, syn.targetPos, pPhase);

            // Add subtle sine wobble
            const wobble = Math.sin(pPhase * Math.PI * 4) * 0.08;
            positions[i * 3] = pos.x;
            positions[i * 3 + 1] = pos.y + wobble;
            positions[i * 3 + 2] = pos.z;
          }
          particles.geometry.attributes.position.needsUpdate = true;
        }

        // Pulse node scale as the wave passes through each layer
        const totalLayers = layers.length;
        nodeMeshesRef.current.forEach((mesh, id) => {
          if (id === selectedNodeId || id === hoveredNodeId) return;
          const node = nodeMap.get(id);
          if (!node) return;

          const layerNormPos = node.layerIdx / Math.max(1, totalLayers - 1);
          const distToWave = Math.abs(progress - layerNormPos);

          if (distToWave < 0.12) {
            const pulseFactor = 1.0 + (1 - distToWave / 0.12) * 0.45;
            mesh.scale.set(pulseFactor, pulseFactor, pulseFactor);
            (mesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 1.0 + (1 - distToWave / 0.12) * 1.2;
          } else {
            mesh.scale.set(1.0, 1.0, 1.0);
            (mesh.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.5;
          }
        });
      }

      // Render
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, autoRotate, propagationSpeed, synapses, layers, selectedNodeId, hoveredNodeId, nodeMap]);

  // Camera Presets
  const setCameraView = (view: 'perspective' | 'front' | 'top') => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    if (view === 'perspective') {
      camera.position.set(0, 6, 22);
      controls.target.set(0, 0, 0);
    } else if (view === 'front') {
      camera.position.set(0, 0, 24);
      controls.target.set(0, 0, 0);
    } else if (view === 'top') {
      camera.position.set(0, 24, 0.1);
      controls.target.set(0, 0, 0);
    }
    controls.update();
  };

  // Modular Layer Mutators
  const cycleActivation = (lIdx: number) => {
    setLayers((prev) =>
      prev.map((l, idx) => {
        if (idx !== lIdx) return l;
        const curIdx = ACTIVATION_CYCLE.indexOf(l.activation);
        const nextAct = ACTIVATION_CYCLE[(curIdx + 1) % ACTIVATION_CYCLE.length];
        return { ...l, activation: nextAct };
      })
    );
  };

  const addLayer = () => {
    if (layers.length >= 8) return;
    const newIdx = layers.length;
    setLayers((prev) => [
      ...prev.slice(0, prev.length - 1),
      {
        id: `l${newIdx}`,
        name: `Layer ${newIdx} (Hidden)`,
        neuronCount: 8,
        activation: 'SiLU',
      },
      prev[prev.length - 1],
    ]);
  };

  const removeLayer = () => {
    if (layers.length <= 3) return;
    setLayers((prev) => [
      ...prev.slice(0, prev.length - 2),
      prev[prev.length - 1],
    ]);
  };

  const adjustNeurons = (lIdx: number, delta: number) => {
    setLayers((prev) =>
      prev.map((l, idx) => {
        if (idx !== lIdx) return l;
        const newCount = Math.max(3, Math.min(14, l.neuronCount + delta));
        return { ...l, neuronCount: newCount };
      })
    );
  };

  return (
    <div className="three-visualizer-container">
      {/* 3D WebGL Canvas Viewport */}
      <div
        className="three-canvas-wrap"
        ref={mountRef}
        onPointerMove={handlePointerMove}
        onClick={handleClick}
      />

      {/* 3D Sci-Fi Floating Tooltip */}
      {tooltip.visible && tooltip.node && (
        <div
          className="three-tooltip"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          <strong>{tooltip.node.id.toUpperCase()}</strong> &bull; Layer {tooltip.node.layerIdx + 1}
          <span className="sub">
            Neuron #{tooltip.node.nodeIdx} | Base Act: {tooltip.node.baseActivation.toFixed(3)}
          </span>
        </div>
      )}

      {/* Top Header HUD */}
      <div className="three-hud-topbar">
        <div className="hud-title-lockup">
          <div className="hud-sys-icon">3D</div>
          <div className="hud-heading-text">
            <h2>
              3D Neural Network Spatial Visualizer
              <span style={{ color: '#FFD166', fontSize: 10 }}>[PROTOTYPE HUD]</span>
            </h2>
            <span>Modular Layer Propagation &bull; WebGL Synaptic Engine</span>
          </div>
        </div>

        <div className="hud-status-pills">
          <div className="hud-pill active">
            <span className="hud-pill-dot" />
            <span>PULSE: {(waveProgress * 100).toFixed(0)}%</span>
          </div>
          <div className="hud-pill">
            <span>FPS: {fps}</span>
          </div>
          <div className="hud-pill">
            <span>LAYERS: {layers.length}</span>
          </div>
          <div className="hud-pill">
            <span>NODES: {nodes.length}</span>
          </div>
          <div className="hud-pill">
            <span>SYNAPSES: {synapses.length}</span>
          </div>
          <button
            type="button"
            className="hud-pill"
            style={{ cursor: 'pointer', background: 'rgba(255, 209, 102, 0.15)', color: '#FFD166', borderColor: '#FFD166' }}
            onClick={() => {
              const el = mountRef.current?.parentElement;
              if (!document.fullscreenElement && el) {
                void el.requestFullscreen();
              } else if (document.fullscreenElement) {
                void document.exitFullscreen();
              }
            }}
            title="Toggle Fullscreen 3D Spatial HUD"
          >
            <span>⛶ FULLSCREEN</span>
          </button>
        </div>
      </div>

      {/* Layer Ribbons (Vertical Layer Panes Metadata) */}
      <div className="three-layer-ribbon">
        {layers.map((layer, idx) => (
          <div
            key={layer.id}
            className={`layer-glass-card ${
              selectedNode?.layerIdx === idx ? 'highlighted' : ''
            }`}
          >
            <div className="layer-card-title">
              <span>{layer.name}</span>
              <span className="badge">L{idx + 1}</span>
            </div>
            <div className="layer-card-neurons">
              Neurons: <strong>{layer.neuronCount}</strong>
              <span style={{ float: 'right' }}>
                <button
                  type="button"
                  style={{ background: 'none', border: 0, color: '#FFD166', cursor: 'pointer', padding: '0 2px' }}
                  onClick={() => adjustNeurons(idx, 1)}
                >
                  +
                </button>
                <button
                  type="button"
                  style={{ background: 'none', border: 0, color: '#94A3B8', cursor: 'pointer', padding: '0 2px' }}
                  onClick={() => adjustNeurons(idx, -1)}
                >
                  -
                </button>
              </span>
            </div>
            <button
              type="button"
              className="layer-card-activation-btn"
              onClick={() => cycleActivation(idx)}
              title="Click to cycle activation function"
            >
              <span>{layer.activation}</span>
              <span style={{ fontSize: 8, opacity: 0.7 }}>CYCLE ↻</span>
            </button>
          </div>
        ))}
      </div>

      {/* Bottom Floating Controls HUD */}
      <div className="three-hud-controls">
        <button
          type="button"
          className={`hud-btn ${isPlaying ? 'primary' : ''}`}
          onClick={() => setIsPlaying((p) => !p)}
        >
          {isPlaying ? '⏸ PAUSE WAVE' : '▶ RESUME WAVE'}
        </button>

        <button
          type="button"
          className={`hud-btn ${autoRotate ? 'active' : ''}`}
          onClick={() => setAutoRotate((r) => !r)}
        >
          🔄 {autoRotate ? 'ROTATING' : 'AUTO-ROTATE'}
        </button>

        <div className="hud-slider-group">
          <span>SPEED</span>
          <input
            type="range"
            min="0.4"
            max="3.0"
            step="0.2"
            value={propagationSpeed}
            onChange={(e) => setPropagationSpeed(parseFloat(e.target.value))}
          />
          <span>{propagationSpeed.toFixed(1)}x</span>
        </div>

        <button
          type="button"
          className="hud-btn"
          onClick={() => setCameraView('perspective')}
        >
          📐 3D ANGLE
        </button>
        <button
          type="button"
          className="hud-btn"
          onClick={() => setCameraView('front')}
        >
          🔲 FRONT
        </button>
        <button
          type="button"
          className="hud-btn"
          onClick={() => setCameraView('top')}
        >
          🔝 TOP
        </button>

        <button type="button" className="hud-btn" onClick={addLayer}>
          + ADD LAYER
        </button>
        <button type="button" className="hud-btn" onClick={removeLayer}>
          - REMOVE
        </button>
      </div>

      {/* Right Side Telemetry HUD */}
      <div className="three-hud-sidebar">
        <div className="sidebar-header">
          <h3>
            <span>⚡</span> SPATIAL NEURON TELEMETRY
          </h3>
          <span>Interactive 3D Pathway Inspection</span>
        </div>

        {selectedNode ? (
          <div>
            <div className="hud-stat-box" style={{ borderColor: '#FFD166' }}>
              <div className="hud-stat-label">
                <span>SELECTED NODE</span>
                <span style={{ color: '#FFD166' }}>ACTIVE</span>
              </div>
              <div className="hud-stat-val accent">{selectedNode.id.toUpperCase()}</div>
              <span style={{ fontSize: 10, color: '#94A3B8' }}>
                Layer #{selectedNode.layerIdx + 1} ({layers[selectedNode.layerIdx]?.name})
              </span>
            </div>

            <div className="hud-stat-box">
              <div className="hud-stat-label">
                <span>3D WORLD COORDINATES</span>
              </div>
              <div className="hud-stat-val" style={{ fontSize: 12 }}>
                X: {selectedNode.x.toFixed(2)} &bull; Y: {selectedNode.y.toFixed(2)} &bull; Z: {selectedNode.z.toFixed(2)}
              </div>
            </div>

            <div className="hud-stat-box">
              <div className="hud-stat-label">
                <span>BASE SCALAR ACTIVATION</span>
                <span>{layers[selectedNode.layerIdx]?.activation}</span>
              </div>
              <div className="hud-stat-val accent">
                +{selectedNode.baseActivation.toFixed(4)}
              </div>
            </div>

            <div className="hud-stat-box">
              <div className="hud-stat-label">
                <span>CONNECTED SYNAPSES</span>
                <span style={{ color: '#FFD166' }}>{selectedSynapses.length} LINKS</span>
              </div>
              <div className="hud-synapse-list">
                {selectedSynapses.map((syn) => (
                  <div key={syn.id} className="hud-synapse-row">
                    <span>{syn.id}</span>
                    <span className="weight">w={syn.weight.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            </div>

            <button
              type="button"
              className="hud-btn"
              style={{ width: '100%', marginTop: 8 }}
              onClick={() => setSelectedNodeId(null)}
            >
              CLEAR SELECTION
            </button>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '60px 10px', color: '#64748B' }}>
            <span style={{ fontSize: 32, display: 'block', marginBottom: 12 }}>🌐</span>
            <strong style={{ color: '#FFD166', display: 'block', marginBottom: 6 }}>
              No 3D Node Selected
            </strong>
            <p style={{ fontSize: 11, lineHeight: 1.6, color: '#94A3B8' }}>
              Hover and tap any glowing node or connection line in the 3D spatial interface to trace its forward and backward pathways in electric yellow (<code>#FFD166</code>).
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
