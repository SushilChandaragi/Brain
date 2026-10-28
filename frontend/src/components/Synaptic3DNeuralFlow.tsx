import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { apiClient, NeuralGraphResponse, NeuronNode, SynapseLink } from '../api/client';
import '../Synaptic3DNeuralFlow.css';

interface Synaptic3DNeuralFlowProps {
  initialPrompt?: string;
  onProceedToStep3?: () => void;
}

interface Node3DPoint {
  id: string;
  layerIdx: number;
  neuronIdx: number;
  node: NeuronNode;
  pos: THREE.Vector3;
}

export const Synaptic3DNeuralFlow: React.FC<Synaptic3DNeuralFlowProps> = ({
  initialPrompt = 'The capital of Bihar is',
  onProceedToStep3,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // User Query & Graph State
  const [prompt, setPrompt] = useState<string>(initialPrompt);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<NeuralGraphResponse | null>(null);

  // Interaction State
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [silencedNodes, setSilencedNodes] = useState<Set<string>>(new Set());
  const [autoRotate, setAutoRotate] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [pulseSpeed, setPulseSpeed] = useState<number>(1.2);

  // 3D Object Registries
  const nodeMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const haloMeshesRef = useRef<Map<string, THREE.Mesh>>(new Map());
  const synapseLinesRef = useRef<Map<string, THREE.Line>>(new Map());
  const pulseParticlesRef = useRef<THREE.Points | null>(null);
  const textSpritesRef = useRef<THREE.Sprite[]>([]);
  const layerPlanesGroupRef = useRef<THREE.Group>(new THREE.Group());
  const networkGroupRef = useRef<THREE.Group>(new THREE.Group());

  // Tooltip
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    node: NeuronNode | null;
  }>({
    visible: false,
    x: 0,
    y: 0,
    node: null,
  });

  // Fetch Live Graph Data from Backend
  const fetchGraph = useCallback(async (queryPrompt: string) => {
    if (!queryPrompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getNeuralGraph(queryPrompt.trim(), 10);
      setGraphData(data);
      if (data.winner_node_id) {
        setSelectedNodeId(data.winner_node_id);
      }
    } catch (err: unknown) {
      setError(`Failed to compute neural flow: ${err}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    setPrompt(initialPrompt);
    void fetchGraph(initialPrompt);
  }, [fetchGraph, initialPrompt]);

  // Compute 3D Positions for Nodes & Curved Synapses
  const { nodePoints, nodePointMap, curvedSynapses } = useMemo(() => {
    if (!graphData) return { nodePoints: [], nodePointMap: new Map(), curvedSynapses: [] };

    const totalCols = graphData.columns.length || 5;
    const colSpacing = 4.2;
    const nPoints: Node3DPoint[] = [];
    const nMap = new Map<string, Node3DPoint>();

    // Group nodes by column
    const colGroups: NeuronNode[][] = Array.from({ length: totalCols }, () => []);
    graphData.nodes.forEach((n) => {
      if (n.layer_idx >= 0 && n.layer_idx < totalCols) {
        colGroups[n.layer_idx].push(n);
      }
    });

    colGroups.forEach((cNodes, colIdx) => {
      const x = (colIdx - (totalCols - 1) / 2) * colSpacing;
      const count = cNodes.length;
      const verticalSpan = Math.min(10.5, Math.max(5.5, count * 0.95));
      const step = count > 1 ? verticalSpan / (count - 1) : 0;

      cNodes.forEach((node, rowIdx) => {
        const y = count > 1 ? verticalSpan / 2 - rowIdx * step : 0;
        // Subtle depth curve
        const z = Math.sin((rowIdx / Math.max(1, count - 1)) * Math.PI) * 0.45;
        const pos = new THREE.Vector3(x, y, z);
        const pt: Node3DPoint = {
          id: node.id,
          layerIdx: colIdx,
          neuronIdx: rowIdx,
          node,
          pos,
        };
        nPoints.push(pt);
        nMap.set(node.id, pt);
      });
    });

    // Generate Smooth Cubic Bézier Curves for Synapses (like the reference images!)
    const curves: {
      id: string;
      link: SynapseLink;
      curve: THREE.CubicBezierCurve3;
      srcPt: Node3DPoint;
      tgtPt: Node3DPoint;
    }[] = [];

    graphData.links.forEach((link) => {
      const src = nMap.get(link.source_id);
      const tgt = nMap.get(link.target_id);
      if (!src || !tgt) return;

      const deltaX = tgt.pos.x - src.pos.x;
      // Soft arching wobble
      const arch = ((src.neuronIdx - tgt.neuronIdx) * 0.04);
      const cp1 = new THREE.Vector3(
        src.pos.x + deltaX * 0.45,
        src.pos.y + arch,
        src.pos.z + 0.1
      );
      const cp2 = new THREE.Vector3(
        src.pos.x + deltaX * 0.55,
        tgt.pos.y - arch,
        tgt.pos.z - 0.1
      );

      const bezier = new THREE.CubicBezierCurve3(src.pos, cp1, cp2, tgt.pos);
      curves.push({
        id: `${link.source_id}->${link.target_id}`,
        link,
        curve: bezier,
        srcPt: src,
        tgtPt: tgt,
      });
    });

    return { nodePoints: nPoints, nodePointMap: nMap, curvedSynapses: curves };
  }, [graphData]);

  // Selected Node Data for HUD
  const selectedNodeObj = useMemo(() => {
    if (!selectedNodeId || !graphData) return null;
    return graphData.nodes.find((n) => n.id === selectedNodeId) || null;
  }, [selectedNodeId, graphData]);

  const selectedStats = useMemo(() => {
    if (!selectedNodeId || !graphData) return null;
    const incoming = graphData.links.filter((l) => l.target_id === selectedNodeId);
    const outgoing = graphData.links.filter((l) => l.source_id === selectedNodeId);
    const maxIn = incoming.length ? Math.max(...incoming.map((l) => l.weight)) : 0;
    const maxOut = outgoing.length ? Math.max(...outgoing.map((l) => l.weight)) : 0;
    return {
      incomingCount: incoming.length,
      outgoingCount: outgoing.length,
      maxInWeight: maxIn,
      maxOutWeight: maxOut,
    };
  }, [selectedNodeId, graphData]);

  // Initialize Three.js Scene with Clean Light Mode Theme
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // 1. Clean White Scene matching website
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    scene.fog = new THREE.FogExp2(0xffffff, 0.024);
    sceneRef.current = scene;

    // 2. Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 0.4, 26);
    cameraRef.current = camera;

    // 3. High Performance Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.innerHTML = '';
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 38;
    controls.minDistance = 6;
    controls.maxPolarAngle = Math.PI / 2 - 0.02;
    controlsRef.current = controls;

    // 5. Lighting for Light Mode
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.3);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(12, 22, 16);
    scene.add(dirLight);

    const softFillLight = new THREE.DirectionalLight(0xe2e8f0, 0.6);
    softFillLight.position.set(-12, -10, -10);
    scene.add(softFillLight);

    // 6. Floor Perspective Grid in subtle light grey
    const floorY = -6.4;
    const gridHelper = new THREE.GridHelper(60, 40, 0xd7d5cc, 0xf0eee5);
    gridHelper.position.y = floorY;
    scene.add(gridHelper);

    // Ground Plane underneath
    const groundGeo = new THREE.PlaneGeometry(80, 80);
    const groundMat = new THREE.MeshBasicMaterial({
      color: 0xfcfbf9,
      side: THREE.DoubleSide,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = floorY - 0.05;
    scene.add(ground);

    // Add Groups
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

  // Helper to generate 2D canvas text sprites for 3D tags (like reference image!)
  const createTextSprite = (
    text: string,
    isWinner: boolean = false,
    isInput: boolean = false
  ): THREE.Sprite => {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 72;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (isWinner) {
        // Vibrant electric yellow box exactly like reference image!
        ctx.fillStyle = '#FFD166';
        ctx.strokeStyle = '#eab308';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.roundRect(10, 8, 300, 56, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#000000';
        ctx.font = 'bold 26px "DM Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 160, 36);
      } else if (isInput) {
        // Clean white input pill with subtle border
        ctx.fillStyle = '#ffffff';
        ctx.strokeStyle = '#cbd5e1';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(10, 10, 300, 52, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#20221f';
        ctx.font = '500 24px "DM Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 160, 36);
      } else {
        // Output candidate or node index tag
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.strokeStyle = '#d7d5cc';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.roundRect(10, 10, 300, 52, 4);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#374151';
        ctx.font = '600 22px "DM Mono", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(text, 160, 36);
      }
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const spriteMat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthWrite: false,
    });
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(2.4, 0.54, 1.0);
    return sprite;
  };

  // Build & Rebuild 3D Layers, Curved Light-Grey Synapses, and Nodes
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !graphData) return;

    const planesGroup = layerPlanesGroupRef.current;
    const netGroup = networkGroupRef.current;
    while (planesGroup.children.length > 0) planesGroup.remove(planesGroup.children[0]);
    while (netGroup.children.length > 0) netGroup.remove(netGroup.children[0]);

    nodeMeshesRef.current.clear();
    haloMeshesRef.current.clear();
    synapseLinesRef.current.clear();
    textSpritesRef.current = [];

    const totalCols = graphData.columns.length || 5;
    const colSpacing = 4.2;

    // 1. Frosted Translucent Glass Layer Panes in 3D
    graphData.columns.forEach((_, cIdx) => {
      const x = (cIdx - (totalCols - 1) / 2) * colSpacing;
      const height = 10.4;
      const width = 2.2;

      const planeGeo = new THREE.PlaneGeometry(width, height);
      // Clean white translucent frosted pane matching reference
      const planeMat = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.65,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const plane = new THREE.Mesh(planeGeo, planeMat);
      plane.position.set(x, 0, -0.05);

      // Clean subtle light grey edge
      const edges = new THREE.EdgesGeometry(planeGeo);
      const edgeMat = new THREE.LineBasicMaterial({
        color: 0xd7d5cc,
        transparent: true,
        opacity: 0.85,
      });
      const border = new THREE.LineSegments(edges, edgeMat);
      plane.add(border);

      planesGroup.add(plane);
    });

    // 2. Curved Light-Grey Synapse Lines (Bézier curves!)
    curvedSynapses.forEach((syn) => {
      const pts = syn.curve.getPoints(24);
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      // Base light-grey color (#cbd5e1 / #94a3b8)
      const baseOpacity = 0.16 + syn.link.weight * 0.32;
      const lineMat = new THREE.LineBasicMaterial({
        color: 0x94a3b8,
        transparent: true,
        opacity: baseOpacity,
        depthWrite: false,
      });
      const line = new THREE.Line(lineGeo, lineMat);
      line.userData = { synapseId: syn.id, ...syn, baseOpacity };
      netGroup.add(line);
      synapseLinesRef.current.set(syn.id, line);
    });

    // 3. Circular Glowing Nodes
    const sphereGeo = new THREE.SphereGeometry(0.24, 24, 24);
    const ringGeo = new THREE.RingGeometry(0.28, 0.42, 28);

    nodePoints.forEach((pt) => {
      const isWinner = pt.id === graphData.winner_node_id;

      // Node sphere: clean white with subtle border
      const nodeMat = new THREE.MeshStandardMaterial({
        color: isWinner ? 0xffd166 : 0xffffff,
        emissive: isWinner ? 0xeab308 : 0xcbd5e1,
        emissiveIntensity: isWinner ? 1.0 : 0.4,
        roughness: 0.15,
        metalness: 0.6,
      });
      const mesh = new THREE.Mesh(sphereGeo, nodeMat);
      mesh.position.copy(pt.pos);
      mesh.userData = { nodeId: pt.id, ...pt };

      // Halo ring
      const haloMat = new THREE.MeshBasicMaterial({
        color: isWinner ? 0xffd166 : 0x94a3b8,
        transparent: true,
        opacity: isWinner ? 0.8 : 0.25,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const halo = new THREE.Mesh(ringGeo, haloMat);
      halo.position.copy(pt.pos);
      halo.position.z += 0.02;

      netGroup.add(mesh);
      netGroup.add(halo);

      nodeMeshesRef.current.set(pt.id, mesh);
      haloMeshesRef.current.set(pt.id, halo);

      // Attach 3D Label Tag Sprite
      if (pt.node.role === 'input') {
        const sprite = createTextSprite(pt.node.details?.token_string || pt.node.label, false, true);
        sprite.position.set(pt.pos.x - 1.35, pt.pos.y, pt.pos.z);
        netGroup.add(sprite);
        textSpritesRef.current.push(sprite);
      } else if (pt.node.role === 'output') {
        const textStr = `${pt.node.details?.token_string} (${pt.node.details?.percentage})`;
        const sprite = createTextSprite(textStr, isWinner, false);
        sprite.position.set(pt.pos.x + 1.35, pt.pos.y, pt.pos.z);
        netGroup.add(sprite);
        textSpritesRef.current.push(sprite);
      } else {
        // Hidden node tag e.g. "N1", "N2" exactly like reference image!
        const sprite = createTextSprite(`N${pt.neuronIdx + 1}`, false, false);
        sprite.scale.set(0.9, 0.26, 1.0);
        sprite.position.set(pt.pos.x, pt.pos.y - 0.42, pt.pos.z);
        netGroup.add(sprite);
        textSpritesRef.current.push(sprite);
      }
    });

    // 4. Pulse Particles along Bézier curves
    const particleCount = 180;
    const particleGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i++) positions[i] = 0;
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particleMat = new THREE.PointsMaterial({
      color: 0xeab308,
      size: 0.28,
      transparent: true,
      opacity: 0.9,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    netGroup.add(particles);
    pulseParticlesRef.current = particles;
  }, [graphData, nodePoints, curvedSynapses]);

  // Handle Node Hover & Selection Highlight (Electric Yellow Highlight!)
  useEffect(() => {
    const activeId = hoveredNodeId || selectedNodeId;

    nodeMeshesRef.current.forEach((mesh, id) => {
      const halo = haloMeshesRef.current.get(id);
      const isTarget = id === activeId;
      const isWinner = id === graphData?.winner_node_id;
      const isConnected =
        activeId &&
        graphData?.links.some(
          (l) =>
            (l.source_id === activeId && l.target_id === id) ||
            (l.target_id === activeId && l.source_id === id)
        );

      const mat = mesh.material as THREE.MeshStandardMaterial;
      const hMat = halo ? (halo.material as THREE.MeshBasicMaterial) : null;

      if (isTarget) {
        // Electric Yellow / Amber
        mat.color.setHex(0xffd166);
        mat.emissive.setHex(0xeab308);
        mat.emissiveIntensity = 1.2;
        mesh.scale.set(1.4, 1.4, 1.4);
        if (hMat && halo) {
          hMat.color.setHex(0xffd166);
          hMat.opacity = 0.95;
          halo.scale.set(1.5, 1.5, 1.5);
        }
      } else if (isConnected) {
        // Soft yellow connected node
        mat.color.setHex(0xfef08a);
        mat.emissive.setHex(0xeab308);
        mat.emissiveIntensity = 0.8;
        mesh.scale.set(1.2, 1.2, 1.2);
        if (hMat && halo) {
          hMat.color.setHex(0xfef08a);
          hMat.opacity = 0.8;
          halo.scale.set(1.3, 1.3, 1.3);
        }
      } else if (isWinner) {
        mat.color.setHex(0xffd166);
        mat.emissive.setHex(0xeab308);
        mat.emissiveIntensity = 0.9;
        mesh.scale.set(1.25, 1.25, 1.25);
        if (hMat && halo) {
          hMat.color.setHex(0xffd166);
          hMat.opacity = 0.75;
          halo.scale.set(1.3, 1.3, 1.3);
        }
      } else {
        // Base Clean White Node
        mat.color.setHex(0xffffff);
        mat.emissive.setHex(0xd1d5db);
        mat.emissiveIntensity = 0.35;
        mesh.scale.set(1.0, 1.0, 1.0);
        if (hMat && halo) {
          hMat.color.setHex(0x94a3b8);
          hMat.opacity = 0.25;
          halo.scale.set(1.0, 1.0, 1.0);
        }
      }
    });

    // Update Curved Synapse Lines (Light Grey -> Electric Yellow for Active!)
    synapseLinesRef.current.forEach((line) => {
      const syn = line.userData as {
        link: SynapseLink;
        baseOpacity: number;
        srcPt: Node3DPoint;
        tgtPt: Node3DPoint;
      };
      const lineMat = line.material as THREE.LineBasicMaterial;
      const isConnected =
        activeId &&
        (syn.link.source_id === activeId || syn.link.target_id === activeId);

      if (isConnected) {
        // Connected: Vibrant Electric Yellow!
        lineMat.color.setHex(0xeab308);
        lineMat.opacity = 0.95;
      } else if (!activeId) {
        // Default: Soft Light Grey (#94a3b8 / #cbd5e1)
        lineMat.color.setHex(0x94a3b8);
        lineMat.opacity = syn.baseOpacity;
      } else {
        // Inactive during focus: Very faint light grey
        lineMat.color.setHex(0xd1d5db);
        lineMat.opacity = 0.08;
      }
    });
  }, [hoveredNodeId, selectedNodeId, graphData]);

  // Raycaster for Pointer Hover & Selection
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
        const pt = nodePointMap.get(nodeId);
        setTooltip({
          visible: true,
          x: e.clientX - rect.left,
          y: e.clientY - rect.top,
          node: pt ? pt.node : null,
        });
      } else {
        setHoveredNodeId(null);
        setTooltip((prev) => ({ ...prev, visible: false }));
      }
    },
    [nodePointMap]
  );

  const handleClick = useCallback(() => {
    if (hoveredNodeId) {
      setSelectedNodeId((prev) => (prev === hoveredNodeId ? null : hoveredNodeId));
    }
  }, [hoveredNodeId]);

  // 3D Animation Loop (Pulse Waves, OrbitControls)
  useEffect(() => {
    let lastTime = performance.now();
    let progress = 0;

    const animate = (currentTime: number) => {
      animFrameRef.current = requestAnimationFrame(animate);

      const delta = (currentTime - lastTime) / 1000;
      lastTime = currentTime;

      if (controlsRef.current) {
        controlsRef.current.autoRotate = autoRotate;
        controlsRef.current.autoRotateSpeed = 0.8;
        controlsRef.current.update();
      }

      // Animate Pulse Wave along Bézier Curves
      if (isPlaying && curvedSynapses.length > 0) {
        progress = (progress + delta * pulseSpeed * 0.45) % 1.0;

        const particles = pulseParticlesRef.current;
        if (particles) {
          const positions = particles.geometry.attributes.position.array as Float32Array;
          const count = positions.length / 3;

          for (let i = 0; i < count; i++) {
            const syn = curvedSynapses[i % curvedSynapses.length];
            const pPhase = (progress + (i / count) * 0.8) % 1.0;
            const pos = syn.curve.getPoint(pPhase);

            positions[i * 3] = pos.x;
            positions[i * 3 + 1] = pos.y;
            positions[i * 3 + 2] = pos.z;
          }
          particles.geometry.attributes.position.needsUpdate = true;
        }
      }

      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };

    animFrameRef.current = requestAnimationFrame(animate);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, autoRotate, pulseSpeed, curvedSynapses]);

  // Camera Presets
  const setCameraView = (view: 'perspective' | 'front' | 'top') => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;

    if (view === 'perspective') {
      camera.position.set(0, 0.4, 26);
      controls.target.set(0, 0, 0);
    } else if (view === 'front') {
      camera.position.set(0, 0, 26);
      controls.target.set(0, 0, 0);
    } else if (view === 'top') {
      camera.position.set(0, 27, 0.1);
      controls.target.set(0, 0, 0);
    }
    controls.update();
  };

  const toggleSilence = (id: string) => {
    setSilencedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  return (
    <div className="synaptic-3d-panel">
      {/* Query Bar */}
      <div className="synaptic-3d-input-bar">
        <input
          type="text"
          className="synaptic-3d-input"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter any query or sentence to inspect the model's inner thoughts..."
          onKeyDown={(e) => {
            if (e.key === 'Enter') void fetchGraph(prompt);
          }}
        />
        <button
          type="button"
          className="synaptic-3d-run-btn"
          onClick={() => void fetchGraph(prompt)}
          disabled={loading || !prompt.trim()}
        >
          {loading ? 'COMPUTING TENSORS...' : 'RUN 3D NEURAL FLOW ⚡'}
        </button>
      </div>

      {/* Presets */}
      <div className="synaptic-3d-presets">
        <span className="muted-label">TEST QUERIES:</span>
        <button
          type="button"
          className="synaptic-3d-chip"
          onClick={() => {
            setPrompt('The capital of Bihar is');
            void fetchGraph('The capital of Bihar is');
          }}
        >
          The capital of Bihar is ➔ "Pat" (Patna)
        </button>
        <button
          type="button"
          className="synaptic-3d-chip"
          onClick={() => {
            setPrompt('capital of bihar is');
            void fetchGraph('capital of bihar is');
          }}
        >
          capital of bihar is ➔ "pat" (patna)
        </button>
        <button
          type="button"
          className="synaptic-3d-chip"
          onClick={() => {
            setPrompt('Mumbai is the capital of');
            void fetchGraph('Mumbai is the capital of');
          }}
        >
          Mumbai is the capital of ➔ "Mah" (Maharashtra)
        </button>
        <button
          type="button"
          className="synaptic-3d-chip"
          onClick={() => {
            setPrompt('The capital of India is');
            void fetchGraph('The capital of India is');
          }}
        >
          The capital of India is ➔ "New"
        </button>
        <button
          type="button"
          className="synaptic-3d-chip"
          onClick={() => {
            setPrompt('7 + 1 = ');
            void fetchGraph('7 + 1 = ');
          }}
        >
          7 + 1 = ➔ "8"
        </button>
        <button
          type="button"
          className="synaptic-3d-chip"
          onClick={() => {
            setPrompt('The capital of France is');
            void fetchGraph('The capital of France is');
          }}
        >
          The capital of France is ➔ "Paris"
        </button>
      </div>

      {/* Logit Lens Timeline Ribbon */}
      {graphData?.thinking_steps && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ font: '10px "DM Mono", monospace', color: 'var(--muted, #797c72)', textTransform: 'uppercase', marginBottom: 6, letterSpacing: '0.08em' }}>
            🧠 MODEL LOGIT LENS EVOLUTION (How intermediate layers form the thought):
          </div>
          <div className="synaptic-3d-thinking-ribbon">
            {graphData.thinking_steps.map((step) => (
              <div
                key={step.step_index}
                className={`synaptic-3d-thinking-card ${step.step_index === 4 ? 'active' : ''}`}
              >
                <div className="synaptic-3d-stage-title">
                  <span>{step.stage_name}</span>
                  <span>L{String(step.layer_num).padStart(2, '0')}</span>
                </div>
                <div className="synaptic-3d-stage-token">{step.top_hypothesized_token}</div>
                <div className="synaptic-3d-stage-conf">
                  {step.step_index === 0 ? 'Prompt Ingestion' : `${(step.confidence * 100).toFixed(1)}% hypothesis`}
                </div>
                <div className="synaptic-3d-stage-desc">{step.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Autoregressive Continuation Banner */}
      {graphData?.autoregressive_continuation && (
        <div className="synaptic-3d-continuation-banner">
          <div className="header">
            <span className="tag">AUTOREGRESSIVE DECODED SEQUENCE (NEXT TOKENS)</span>
            <span className="continuation-text">
              Full continuation: <strong>"{graphData.autoregressive_continuation}"</strong>
            </span>
          </div>
          {graphData.subword_explanation && (
            <div className="subword-note">
              💡 <strong>Subword Tokenization (BPE):</strong> {graphData.subword_explanation}
            </div>
          )}
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {/* 3D Main Viewport Grid */}
      <div className="synaptic-3d-viewport-grid">
        <div className="synaptic-3d-canvas-container">
          {/* WebGL Canvas */}
        <div
          className="synaptic-3d-canvas"
          ref={mountRef}
          onPointerMove={handlePointerMove}
          onClick={handleClick}
        />

        {/* 3D Floating Tooltip */}
        {tooltip.visible && tooltip.node && (
          <div className="synaptic-3d-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
            <strong>{tooltip.node.label}</strong> &bull; {tooltip.node.layer_name}
            <span className="sub">
              Scalar Act: {tooltip.node.activation.toFixed(4)} | Normalized: {(tooltip.node.normalized * 100).toFixed(0)}%
            </span>
          </div>
        )}

        {/* Floating Layer Headers Ribbon (Top of Canvas) */}
        {graphData && (
          <div className="synaptic-3d-layer-panes-bar">
            {graphData.columns.map((colName, cIdx) => (
              <div
                key={cIdx}
                className={`synaptic-3d-layer-card ${
                  selectedNodeObj?.layer_idx === cIdx ? 'active' : ''
                }`}
              >
                <div className="title">
                  <span>{colName}</span>
                  <span style={{ color: 'var(--muted, #797c72)', fontSize: 9 }}>L{cIdx + 1}</span>
                </div>
                <div className="neurons-count">
                  Active Units: {graphData.nodes.filter((n) => n.layer_idx === cIdx).length}
                </div>
                <span className="activation-tag">
                  {cIdx === 0 ? 'EMBEDDING' : cIdx === 4 ? 'SOFTMAX' : 'CHECKPOINT'}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Floating Bottom Controls in Canvas */}
        <div className="synaptic-3d-controls-bar">
          <button
            type="button"
            className={`synaptic-3d-ctrl-btn ${isPlaying ? 'active' : ''}`}
            onClick={() => setIsPlaying((p) => !p)}
          >
            {isPlaying ? '⏸ PAUSE PULSE' : '▶ RESUME PULSE'}
          </button>
          <button
            type="button"
            className={`synaptic-3d-ctrl-btn ${autoRotate ? 'active' : ''}`}
            onClick={() => setAutoRotate((r) => !r)}
          >
            🔄 {autoRotate ? 'ROTATING' : 'AUTO-ROTATE'}
          </button>
          <div className="synaptic-3d-slider-group">
            <span>SPEED</span>
            <input
              type="range"
              min="0.4"
              max="3.0"
              step="0.2"
              value={pulseSpeed}
              onChange={(e) => setPulseSpeed(parseFloat(e.target.value))}
            />
            <span>{pulseSpeed.toFixed(1)}x</span>
          </div>
          <button
            type="button"
            className="synaptic-3d-ctrl-btn"
            onClick={() => setCameraView('perspective')}
          >
            📐 3D ANGLE
          </button>
          <button
            type="button"
            className="synaptic-3d-ctrl-btn"
            onClick={() => setCameraView('front')}
          >
            🔲 FRONT
          </button>
          <button
            type="button"
            className="synaptic-3d-ctrl-btn"
            onClick={() => setCameraView('top')}
          >
            🔝 TOP
          </button>
          <button
            type="button"
            className="synaptic-3d-ctrl-btn"
            onClick={() => {
              const el = mountRef.current?.parentElement;
              if (!document.fullscreenElement && el) {
                void el.requestFullscreen();
              } else if (document.fullscreenElement) {
                void document.exitFullscreen();
              }
            }}
          >
            ⛶ FULLSCREEN
          </button>
        </div>
      </div>

      {/* Right Side Telemetry HUD (Clean Light Mode) */}
      <div className="synaptic-3d-hud-sidebar">
          {selectedNodeObj ? (
            <div>
              <div className="synaptic-3d-hud-header">
                <h3>
                  <span>{selectedNodeObj.layer_name}</span>
                  <span
                    className={`synaptic-3d-hud-badge ${
                      selectedNodeObj.id === graphData?.winner_node_id ? 'winner' : ''
                    }`}
                  >
                    {selectedNodeObj.id === graphData?.winner_node_id
                      ? '★ DECIDED WINNER'
                      : selectedNodeObj.role === 'input'
                      ? `TOKEN #${selectedNodeObj.neuron_idx}`
                      : selectedNodeObj.role === 'output'
                      ? `RANK #${selectedNodeObj.neuron_idx + 1}`
                      : `NEURON #${selectedNodeObj.neuron_idx + 1}`}
                  </span>
                </h3>
                <div style={{ margin: '8px 0 4px', fontSize: 16, fontWeight: 'bold', color: 'var(--ink, #20221f)' }}>
                  {selectedNodeObj.label}
                </div>
                <span style={{ fontSize: 10, color: 'var(--muted, #797c72)' }}>
                  ID: <code>{selectedNodeObj.id}</code>
                </span>
              </div>

              {silencedNodes.has(selectedNodeObj.id) && (
                <div style={{ background: '#fef2f2', border: '1px solid #f87171', color: '#991b1b', padding: 8, fontSize: 10, marginBottom: 10 }}>
                  ⚠ NEURON SILENCED (Downstream transmission blocked)
                </div>
              )}

              {/* INPUT SPECIFICS */}
              {selectedNodeObj.role === 'input' && (
                <>
                  <div className="synaptic-3d-metric-row">
                    <span>SEQUENCE POSITION</span>
                    <strong>#{selectedNodeObj.details?.token_position}</strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>VOCABULARY ID</span>
                    <strong>ID #{selectedNodeObj.details?.token_id}</strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>EMBEDDING L2 NORM</span>
                    <strong>{selectedNodeObj.activation.toFixed(4)}</strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>QUERY ATTENTION</span>
                    <strong className="accent">{selectedNodeObj.details?.attention_received}</strong>
                  </div>
                </>
              )}

              {/* HIDDEN SPECIFICS */}
              {selectedNodeObj.role === 'hidden' && (
                <>
                  <div className="synaptic-3d-metric-row">
                    <span>SCALAR ACTIVATION</span>
                    <strong className="accent">
                      {selectedNodeObj.activation >= 0 ? `+${selectedNodeObj.activation.toFixed(4)}` : selectedNodeObj.activation.toFixed(4)}
                    </strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>NORMALIZED ACTIVITY</span>
                    <strong>{(selectedNodeObj.normalized * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>CHANNEL DIMENSION</span>
                    <strong>#{selectedNodeObj.channel_idx} / 2048</strong>
                  </div>
                  {selectedNodeObj.details?.logit_contribution_to_winner !== undefined && (
                    <div className="synaptic-3d-metric-row">
                      <span>CONTRIBUTION TO WINNER</span>
                      <strong className={selectedNodeObj.details.logit_contribution_to_winner >= 0 ? 'winner-val' : 'accent'}>
                        {selectedNodeObj.details.logit_contribution_to_winner >= 0 ? '+' : ''}
                        {selectedNodeObj.details.logit_contribution_to_winner.toFixed(3)} logits
                      </strong>
                    </div>
                  )}
                </>
              )}

              {/* OUTPUT SPECIFICS */}
              {selectedNodeObj.role === 'output' && (
                <>
                  <div className="synaptic-3d-metric-row">
                    <span>TARGET TOKEN</span>
                    <strong className="accent" style={{ fontSize: 13 }}>
                      "{selectedNodeObj.details?.token_string}"
                    </strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>PREDICTION PROBABILITY</span>
                    <strong className="winner-val">{selectedNodeObj.details?.percentage}</strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>VOCABULARY LOGIT</span>
                    <strong>{selectedNodeObj.activation.toFixed(4)}</strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>MARGIN OVER RUNNER-UP</span>
                    <strong className="accent">{selectedNodeObj.details?.margin_over_runner_up}</strong>
                  </div>
                </>
              )}

              {/* Connected Synapses Stats */}
              {selectedStats && (
                <>
                  <div className="synaptic-3d-metric-row" style={{ marginTop: 6 }}>
                    <span>INCOMING SYNAPSES</span>
                    <strong>{selectedStats.incomingCount} links (max {selectedStats.maxInWeight.toFixed(2)})</strong>
                  </div>
                  <div className="synaptic-3d-metric-row">
                    <span>OUTGOING SYNAPSES</span>
                    <strong>{selectedStats.outgoingCount} links (max {selectedStats.maxOutWeight.toFixed(2)})</strong>
                  </div>
                </>
              )}

              {/* Detailed Breakdown Card */}
              <div className="synaptic-3d-detail-card">
                <span style={{ fontWeight: 700, color: 'var(--ink, #20221f)', display: 'block', marginBottom: 4 }}>
                  MECHANISTIC ROLE:
                </span>
                <div style={{ color: '#4b5563', marginBottom: 6 }}>
                  {selectedNodeObj.role === 'input'
                    ? 'TOKEN EMBEDDING — Sequence Prompt Feature'
                    : selectedNodeObj.role === 'output'
                    ? `DECISION LOGIT — Vocabulary Candidate #${selectedNodeObj.neuron_idx + 1}`
                    : `RESIDUAL FEATURE — Transformer Layer ${selectedNodeObj.details?.layer_number}`}
                </div>

                {/* Vocabulary Projections */}
                {selectedNodeObj.details?.promoted_tokens && (
                  <div style={{ marginTop: 8 }}>
                    <span style={{ fontWeight: 700, color: '#047857', display: 'block' }}>
                      DIRECTLY PROMOTES IN VOCABULARY:
                    </span>
                    <div className="synaptic-3d-pill-list">
                      {selectedNodeObj.details.promoted_tokens.map((tok: string, i: number) => (
                        <span key={i} className="synaptic-3d-promote-pill">+{tok}</span>
                      ))}
                    </div>
                  </div>
                )}

                {selectedNodeObj.details?.suppressed_tokens && (
                  <div style={{ marginTop: 8 }}>
                    <span style={{ fontWeight: 700, color: '#b91c1c', display: 'block' }}>
                      DIRECTLY SUPPRESSES IN VOCABULARY:
                    </span>
                    <div className="synaptic-3d-pill-list">
                      {selectedNodeObj.details.suppressed_tokens.map((tok: string, i: number) => (
                        <span key={i} className="synaptic-3d-suppress-pill">-{tok}</span>
                      ))}
                    </div>
                  </div>
                )}

                <div style={{ marginTop: 8, paddingTop: 6, borderTop: '1px dashed #d7d5cc' }}>
                  <span style={{ fontWeight: 700, color: 'var(--ink, #20221f)' }}>WHAT THIS CARRIES: </span>
                  <span>{selectedNodeObj.details?.description}</span>
                </div>
              </div>

              {/* Silence Action Button */}
              <button
                type="button"
                className="synaptic-3d-run-btn"
                style={{
                  width: '100%',
                  marginTop: 12,
                  background: silencedNodes.has(selectedNodeObj.id) ? '#15803d' : '#dc2626',
                  color: '#ffffff',
                }}
                onClick={() => toggleSilence(selectedNodeObj.id)}
              >
                {silencedNodes.has(selectedNodeObj.id)
                  ? '✓ RESTORE NEURON'
                  : '⚡ SILENCE / ABLATE THIS NEURON'}
              </button>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 10px', color: 'var(--muted, #797c72)' }}>
              <span style={{ fontSize: 32, display: 'block', marginBottom: 12 }}>🧠</span>
              <strong style={{ color: 'var(--ink, #20221f)', display: 'block', marginBottom: 6 }}>
                No Neuron Selected
              </strong>
              <p style={{ fontSize: 11, lineHeight: 1.6 }}>
                Click any node in the 3D network flow to inspect its scalar activation, what words it promotes/suppresses, and its causal contribution to the final answer.
              </p>
            </div>
          )}
          {onProceedToStep3 && (
            <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #e2e8f0' }}>
              <button
                type="button"
                className="proceed-next-step-btn"
                style={{
                  width: '100%',
                  background: 'var(--ink, #0f172a)',
                  color: '#ffffff',
                  padding: '10px 14px',
                  borderRadius: 6,
                  fontWeight: 600,
                  fontSize: 12,
                  cursor: 'pointer',
                  border: 'none',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 6,
                }}
                onClick={onProceedToStep3}
              >
                Proceed to Circuit Localization (Step 3) ➔
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
