import { Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router";
import * as THREE from "three";
import { createShadowMaterial, createTileMaterial } from "./shaders";
import {
  type BoundingBox,
  CAMERA_Z_DEFAULT,
  CAMERA_Z_MAX,
  CAMERA_Z_MIN,
  FLICK_VELOCITY_MULTIPLIER,
  FRICTION,
  MIN_DRAG_DISTANCE_FOR_CLICK,
  type PhotoWallProps,
  ROWS,
  SHADOW_BLUR,
  SHADOW_OFFSET_X,
  SHADOW_OFFSET_Y,
  SHADOW_OFFSET_Z,
  TILE_GAP,
  TILE_GAP_V,
  TILE_HEIGHT,
  TILE_WIDTH,
  type WallTile,
  ZOOM_SPEED,
} from "./types";

interface TileData {
  mesh: THREE.Mesh;
  shadowMesh: THREE.Mesh;
  reflectionMesh: THREE.Mesh;
  labelMesh?: THREE.Mesh;
  tile: WallTile;
  column: number;
  row: number;
  baseX: number;
  baseY: number;
  isCircular: boolean;
}

// Create a label texture for text below tiles
// If label is provided, renders it bold and bright
// If only subtitle is provided (no label), renders it in the same position but with subtitle styling
function createLabelTexture(label: string, subtitle?: string): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  const width = 512;
  const height = 80;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return new THREE.CanvasTexture(canvas);

  // Transparent background
  ctx.clearRect(0, 0, width, height);

  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const centerY = height / 2;

  // If no label but has subtitle, render subtitle centered with subtitle styling
  if (!label && subtitle) {
    const subtitleFontSize = 28;
    ctx.font = `${subtitleFontSize}px system-ui, -apple-system, sans-serif`;

    // Shadow
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillText(subtitle, width / 2 + 1, centerY + 1);

    // Text (subtitle styling - muted)
    ctx.fillStyle = "rgba(255, 255, 255, 0.55)";
    ctx.fillText(subtitle, width / 2, centerY);
  } else if (label) {
    // Draw label centered
    const fontSize = 36;
    ctx.font = `600 ${fontSize}px system-ui, -apple-system, sans-serif`;

    // Shadow
    ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
    ctx.fillText(label, width / 2 + 1, centerY + 1);

    // Text (slightly muted white)
    ctx.fillStyle = "rgba(255, 255, 255, 0.75)";
    ctx.fillText(label, width / 2, centerY);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

// Create a collage texture from multiple images with an optional label and bbox for face cropping
async function createCollageTexture(
  imageUrls: string[],
  label?: string,
  bbox?: BoundingBox,
  isCircular: boolean = false,
): Promise<HTMLCanvasElement> {
  const canvas = document.createElement("canvas");
  const size = 512;
  // For circular (face) tiles, use square canvas; for rectangular tiles, use aspect ratio
  canvas.width = size;
  canvas.height = isCircular ? size : size * (TILE_HEIGHT / TILE_WIDTH);
  const ctx = canvas.getContext("2d");
  if (!ctx) return canvas;

  // Fill with dark background
  ctx.fillStyle = "#1a1a2e";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Load images
  const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = url;
    });
  };

  const images = await Promise.allSettled(imageUrls.map(loadImage));
  const loadedImages = images
    .filter((r) => r.status === "fulfilled")
    .map((r) => (r as PromiseFulfilledResult<HTMLImageElement>).value);

  if (loadedImages.length > 0) {
    // If bbox is provided, crop to the face region
    if (bbox && loadedImages.length === 1) {
      const img = loadedImages[0];

      // bbox coordinates are in pixels (relative to imageWidth/imageHeight)
      // We need to scale them to the actual loaded image dimensions
      const scaleX = img.width / bbox.imageWidth;
      const scaleY = img.height / bbox.imageHeight;

      // Add padding around the face (20% on each side for context)
      const padding = 0.2;

      // For circular tiles, make the crop square; for rectangular, use bbox aspect with padding
      const paddedWidth = bbox.width * (1 + padding * 2);
      const paddedHeight = bbox.height * (1 + padding * 2);

      // Center the crop on the face center
      const faceCenterX = bbox.x + bbox.width / 2;
      const faceCenterY = bbox.y + bbox.height / 2;
      const paddedX = faceCenterX - paddedWidth / 2;
      const paddedY = faceCenterY - paddedHeight / 2;

      // Convert to actual image pixel coordinates
      const sx = Math.max(0, paddedX * scaleX);
      const sy = Math.max(0, paddedY * scaleY);
      const sw = Math.min(paddedWidth * scaleX, img.width - sx);
      const sh = Math.min(paddedHeight * scaleY, img.height - sy);

      // Draw to fill the canvas (cover fit)
      const srcAspect = sw / sh;
      const destAspect = canvas.width / canvas.height;
      let drawX = 0,
        drawY = 0,
        drawW = canvas.width,
        drawH = canvas.height;

      if (srcAspect > destAspect) {
        // Source is wider - fit to height, center horizontally
        drawW = canvas.height * srcAspect;
        drawX = (canvas.width - drawW) / 2;
      } else if (srcAspect < destAspect) {
        // Source is taller - fit to width, center vertically
        drawH = canvas.width / srcAspect;
        drawY = (canvas.height - drawH) / 2;
      }

      ctx.drawImage(img, sx, sy, sw, sh, drawX, drawY, drawW, drawH);

      // For circular tiles, don't draw label - it will be rendered separately
      if (isCircular) {
        return canvas;
      }
      // For rectangular tiles with bbox, continue to draw the label below
    } else {
      // Standard collage behavior for multiple images or no bbox
      const cols = loadedImages.length === 1 ? 1 : 2;
      const rows = loadedImages.length <= 2 ? 1 : 2;
      const cellWidth = canvas.width / cols;
      const cellHeight = canvas.height / rows;
      const gap = 2;

      loadedImages.slice(0, 4).forEach((img, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = col * cellWidth + gap / 2;
        const y = row * cellHeight + gap / 2;
        const w = cellWidth - gap;
        const h = cellHeight - gap;

        // Draw image with cover fit
        const imgAspect = img.width / img.height;
        const cellAspect = w / h;

        let sx = 0,
          sy = 0,
          sw = img.width,
          sh = img.height;

        if (imgAspect > cellAspect) {
          sw = img.height * cellAspect;
          sx = (img.width - sw) / 2;
        } else {
          sh = img.width / cellAspect;
          sy = (img.height - sh) / 2;
        }

        ctx.drawImage(img, sx, sy, sw, sh, x, y, w, h);
      });
    }
  }

  // Draw label if provided
  if (label) {
    const fontSize = canvas.height * 0.25;
    ctx.font = `bold ${fontSize}px system-ui, -apple-system, sans-serif`;
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";

    // Shadow
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillText(label, canvas.width - 16, canvas.height - 8);

    // Text
    ctx.fillStyle = "rgba(255, 255, 255, 0.95)";
    ctx.fillText(label, canvas.width - 18, canvas.height - 10);
  }

  return canvas;
}

export function PhotoWall({ tiles, sessionKey, headerContent, onTileClick }: PhotoWallProps) {
  const navigate = useNavigate();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const tilesRef = useRef<TileData[]>([]);
  const wallContainerRef = useRef<THREE.Group | null>(null);
  const loadedTexturesRef = useRef<Map<string | number, THREE.Texture>>(new Map());
  const animationIdRef = useRef<number>(0);

  // Physics state
  const wallPositionRef = useRef({ x: 0, targetX: 0, velocityX: 0 });
  const cameraZRef = useRef({ z: CAMERA_Z_DEFAULT, targetZ: CAMERA_Z_DEFAULT });

  // Interaction state
  const isDraggingRef = useRef(false);
  const lastMouseXRef = useRef(0);
  const lastMouseTimeRef = useRef(0);
  const flickVelocityRef = useRef(0);
  const dragStartXRef = useRef(0);
  const totalDragDistanceRef = useRef(0);
  const wasDraggingRef = useRef(false);

  // Zoom transition state
  const zoomTransitionRef = useRef<{
    active: boolean;
    direction: "in" | "out";
    targetTile: WallTile | null;
    targetX: number;
    targetY: number;
    targetZ: number;
    targetWallX: number;
    progress: number;
    startX: number;
    startY: number;
    startZ: number;
    startWallX: number;
  }>({
    active: false,
    direction: "in",
    targetTile: null,
    targetX: 0,
    targetY: 0,
    targetZ: 0,
    targetWallX: 0,
    progress: 0,
    startX: 0,
    startY: 0,
    startZ: 0,
    startWallX: 0,
  });

  const [isLoading, setIsLoading] = useState(true);
  const [containerReady, setContainerReady] = useState(false);
  const [hoveredTile, setHoveredTile] = useState<WallTile | null>(null);
  const [transitionOpacity, setTransitionOpacity] = useState(0);

  // Calculate wall dimensions
  const columns = Math.ceil(tiles.length / ROWS);
  const wallWidth = columns * (TILE_WIDTH + TILE_GAP);
  // maxX is the maximum scroll distance from center; ensure it's never negative
  const maxX = Math.max(0, wallWidth / 2 - 2);
  const wallWidthRef = useRef(wallWidth);
  const maxXRef = useRef(maxX);
  wallWidthRef.current = wallWidth;
  maxXRef.current = maxX;

  const loadVisibleTextures = useCallback(() => {
    if (!cameraRef.current) return;

    const viewWidth = cameraZRef.current.z * 2;

    tilesRef.current.forEach((tileData) => {
      const tileWorldX = tileData.baseX + wallPositionRef.current.x;
      const isVisible = Math.abs(tileWorldX) < viewWidth;

      if (isVisible && !loadedTexturesRef.current.has(tileData.tile.id)) {
        // Mark as loading
        loadedTexturesRef.current.set(tileData.tile.id, null as unknown as THREE.Texture);

        // For tiles with bbox (face cropping), don't include label in texture (it's rendered separately)
        const hasBbox = !!tileData.tile.metadata?.bbox;
        const labelForTexture = hasBbox ? undefined : tileData.tile.label;

        // Create collage texture with optional bbox for face cropping
        createCollageTexture(
          tileData.tile.imageUrls,
          labelForTexture,
          tileData.tile.metadata?.bbox,
          tileData.isCircular,
        )
          .then((canvas) => {
            const texture = new THREE.CanvasTexture(canvas);
            loadedTexturesRef.current.set(tileData.tile.id, texture);

            // Circular tiles have 1:1 aspect ratio, rectangular have TILE_WIDTH/TILE_HEIGHT
            const imageAspect = tileData.isCircular ? 1.0 : TILE_WIDTH / TILE_HEIGHT;

            // Update main mesh
            const material = tileData.mesh.material as THREE.ShaderMaterial;
            material.uniforms.map.value = texture;
            material.uniforms.hasTexture.value = true;
            material.uniforms.imageAspect.value = imageAspect;
            material.needsUpdate = true;

            // Update reflection mesh
            const reflectionMaterial = tileData.reflectionMesh.material as THREE.ShaderMaterial;
            reflectionMaterial.uniforms.map.value = texture;
            reflectionMaterial.uniforms.hasTexture.value = true;
            reflectionMaterial.uniforms.imageAspect.value = imageAspect;
            reflectionMaterial.needsUpdate = true;
          })
          .catch(() => {
            loadedTexturesRef.current.delete(tileData.tile.id);
          });
      }
    });
  }, []);

  const updateTileCurve = useCallback(() => {
    if (!cameraRef.current) return;

    tilesRef.current.forEach((tileData) => {
      const worldX = tileData.baseX + wallPositionRef.current.x;
      const cylinderRadius = 40;
      const angle = worldX / cylinderRadius;
      const clampedAngle = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, angle));

      const curveX = cylinderRadius * Math.sin(clampedAngle);
      const offsetZ = -cylinderRadius * (1 - Math.cos(clampedAngle));
      const rotationY = clampedAngle;

      tileData.mesh.rotation.y = rotationY;
      tileData.mesh.position.z = offsetZ;
      tileData.mesh.position.x = curveX - wallPositionRef.current.x;

      const shadowOffsetX = clampedAngle * 0.5;
      tileData.shadowMesh.rotation.y = rotationY;
      tileData.shadowMesh.position.z = offsetZ + SHADOW_OFFSET_Z;
      tileData.shadowMesh.position.x = curveX - wallPositionRef.current.x + shadowOffsetX;
      tileData.shadowMesh.position.y = tileData.baseY + SHADOW_OFFSET_Y;

      tileData.reflectionMesh.rotation.y = rotationY;
      tileData.reflectionMesh.position.z = offsetZ;
      tileData.reflectionMesh.position.x = curveX - wallPositionRef.current.x;

      // Update label mesh position for circular tiles
      if (tileData.labelMesh) {
        tileData.labelMesh.rotation.y = rotationY;
        tileData.labelMesh.position.z = offsetZ + 0.01;
        tileData.labelMesh.position.x = curveX - wallPositionRef.current.x;
      }

      const distanceFromCenter = Math.abs(clampedAngle) / (Math.PI / 3);
      const baseOpacity = 0.12;
      const falloff = Math.max(0, 1 - distanceFromCenter * 0.8);
      const reflectionMaterial = tileData.reflectionMesh.material as THREE.ShaderMaterial;
      reflectionMaterial.uniforms.opacity.value = baseOpacity * falloff;
    });
  }, []);

  const animate = useCallback(() => {
    if (!sceneRef.current || !cameraRef.current || !rendererRef.current || !wallContainerRef.current) return;

    const transition = zoomTransitionRef.current;

    if (transition.active) {
      transition.progress += 0.025;

      const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
      const easedProgress = easeInOutCubic(Math.min(transition.progress, 1));

      cameraRef.current.position.x = transition.startX + (transition.targetX - transition.startX) * easedProgress;
      cameraRef.current.position.y = transition.startY + (transition.targetY - transition.startY) * easedProgress;
      cameraRef.current.position.z = transition.startZ + (transition.targetZ - transition.startZ) * easedProgress;

      wallPositionRef.current.x =
        transition.startWallX + (transition.targetWallX - transition.startWallX) * easedProgress;
      wallContainerRef.current.position.x = wallPositionRef.current.x;

      if (transition.direction === "in") {
        const fadeStart = 0.6;
        if (transition.progress > fadeStart) {
          const fadeProgress = (transition.progress - fadeStart) / (1 - fadeStart);
          setTransitionOpacity(fadeProgress);
        }

        if (transition.progress >= 1) {
          transition.active = false;
          if (transition.targetTile?.navigateTo) {
            const returnKey = `wall-return-${sessionKey}`;
            sessionStorage.setItem(
              returnKey,
              JSON.stringify({
                wallX: transition.startWallX,
                tileId: transition.targetTile.id,
              }),
            );
            navigate(transition.targetTile.navigateTo);
          } else if (onTileClick && transition.targetTile) {
            onTileClick(transition.targetTile);
          }
        }
      } else {
        const fadeEnd = 0.4;
        if (transition.progress < fadeEnd) {
          const fadeProgress = 1 - transition.progress / fadeEnd;
          setTransitionOpacity(fadeProgress);
        } else {
          setTransitionOpacity(0);
        }

        if (transition.progress >= 1) {
          transition.active = false;
          setTransitionOpacity(0);
        }
      }
    } else {
      if (!isDraggingRef.current) {
        wallPositionRef.current.velocityX *= FRICTION;
        if (Math.abs(wallPositionRef.current.velocityX) < 0.0001) {
          wallPositionRef.current.velocityX = 0;
        }
      }
      wallPositionRef.current.x += wallPositionRef.current.velocityX;

      const maxX = maxXRef.current;
      const softZone = 3;
      const pos = wallPositionRef.current.x;

      if (Math.abs(pos) > maxX - softZone) {
        const distanceIntoSoftZone = (Math.abs(pos) - (maxX - softZone)) / softZone;
        const edgeFriction = 1 - Math.min(distanceIntoSoftZone, 1) * 0.5;
        wallPositionRef.current.velocityX *= edgeFriction;
      }

      if (wallPositionRef.current.x > maxX) {
        wallPositionRef.current.x = maxX;
        wallPositionRef.current.velocityX = 0;
      } else if (wallPositionRef.current.x < -maxX) {
        wallPositionRef.current.x = -maxX;
        wallPositionRef.current.velocityX = 0;
      }

      const dz = cameraZRef.current.targetZ - cameraZRef.current.z;
      cameraZRef.current.z += dz * 0.1;
      cameraRef.current.position.z = cameraZRef.current.z;

      wallContainerRef.current.position.x = wallPositionRef.current.x;
    }

    updateTileCurve();
    loadVisibleTextures();
    rendererRef.current.render(sceneRef.current, cameraRef.current);
    animationIdRef.current = requestAnimationFrame(animate);
  }, [updateTileCurve, loadVisibleTextures, navigate, onTileClick, sessionKey]);

  const handleMouseDown = useCallback((e: MouseEvent) => {
    isDraggingRef.current = true;
    wasDraggingRef.current = false;
    lastMouseXRef.current = e.clientX;
    dragStartXRef.current = e.clientX;
    totalDragDistanceRef.current = 0;
    lastMouseTimeRef.current = performance.now();
    flickVelocityRef.current = 0;
    wallPositionRef.current.velocityX = 0;
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!containerRef.current || !cameraRef.current) return;

    if (isDraggingRef.current) {
      const deltaX = e.clientX - lastMouseXRef.current;
      const currentTime = performance.now();
      const deltaTime = Math.max(1, currentTime - lastMouseTimeRef.current);

      totalDragDistanceRef.current += Math.abs(deltaX);
      if (totalDragDistanceRef.current > MIN_DRAG_DISTANCE_FOR_CLICK) {
        wasDraggingRef.current = true;
      }

      flickVelocityRef.current = (deltaX / deltaTime) * FLICK_VELOCITY_MULTIPLIER;

      const dragSpeed = cameraZRef.current.z * 0.003;
      wallPositionRef.current.x += deltaX * dragSpeed;

      const maxX = maxXRef.current;
      wallPositionRef.current.x = Math.max(-maxX, Math.min(maxX, wallPositionRef.current.x));

      lastMouseXRef.current = e.clientX;
      lastMouseTimeRef.current = currentTime;
    }

    if (!sceneRef.current || !cameraRef.current) return;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (e.clientX / containerRef.current.clientWidth) * 2 - 1,
      -(e.clientY / containerRef.current.clientHeight) * 2 + 1,
    );

    raycaster.setFromCamera(mouse, cameraRef.current);
    const intersects = raycaster.intersectObjects(wallContainerRef.current?.children || [], false);

    const hit = intersects.find((i) => i.object.userData.tile && !i.object.userData.isReflection);
    if (hit) {
      setHoveredTile(hit.object.userData.tile);
      containerRef.current.style.cursor = "pointer";
    } else {
      setHoveredTile(null);
      containerRef.current.style.cursor = isDraggingRef.current ? "grabbing" : "grab";
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    if (isDraggingRef.current) {
      wallPositionRef.current.velocityX = flickVelocityRef.current;
    }
    isDraggingRef.current = false;
    if (containerRef.current) {
      containerRef.current.style.cursor = "grab";
    }
  }, []);

  const handleClick = useCallback((e: MouseEvent) => {
    if (!containerRef.current || !sceneRef.current || !cameraRef.current) return;

    if (zoomTransitionRef.current.active) return;
    if (wasDraggingRef.current) return;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (e.clientX / containerRef.current.clientWidth) * 2 - 1,
      -(e.clientY / containerRef.current.clientHeight) * 2 + 1,
    );

    raycaster.setFromCamera(mouse, cameraRef.current);
    const intersects = raycaster.intersectObjects(wallContainerRef.current?.children || [], false);

    const hit = intersects.find((i) => i.object.userData.tile && !i.object.userData.isReflection);
    if (hit) {
      const tile = hit.object.userData.tile as WallTile;
      const tileData = tilesRef.current.find((t) => t.tile.id === tile.id);

      if (tileData) {
        const transition = zoomTransitionRef.current;
        transition.active = true;
        transition.direction = "in";
        transition.targetTile = tile;
        transition.progress = 0;

        transition.startX = cameraRef.current.position.x;
        transition.startY = cameraRef.current.position.y;
        transition.startZ = cameraRef.current.position.z;
        transition.startWallX = wallPositionRef.current.x;

        transition.targetX = 0;
        transition.targetY = tileData.baseY;
        transition.targetZ = 0.5;
        transition.targetWallX = -tileData.baseX;

        wallPositionRef.current.velocityX = 0;
      }
    }
  }, []);

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();

    if (zoomTransitionRef.current.active) return;

    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      const panSpeed = cameraZRef.current.z * 0.002;
      wallPositionRef.current.x -= e.deltaX * panSpeed;

      const maxX = maxXRef.current;
      wallPositionRef.current.x = Math.max(-maxX, Math.min(maxX, wallPositionRef.current.x));
    } else {
      cameraZRef.current.targetZ += e.deltaY * 0.01 * ZOOM_SPEED;
      cameraZRef.current.targetZ = Math.max(CAMERA_Z_MIN, Math.min(CAMERA_Z_MAX, cameraZRef.current.targetZ));
    }
  }, []);

  const handleResize = useCallback(() => {
    if (!containerRef.current || !cameraRef.current || !rendererRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    cameraRef.current.aspect = width / height;
    cameraRef.current.updateProjectionMatrix();
    rendererRef.current.setSize(width, height);
  }, []);

  // Wait for container to have valid dimensions (needed on page reload/SSR hydration)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Check if already has dimensions
    if (container.clientWidth > 0 && container.clientHeight > 0) {
      setContainerReady(true);
      return;
    }

    // Wait for dimensions via ResizeObserver
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry && entry.contentRect.width > 0 && entry.contentRect.height > 0) {
        setContainerReady(true);
        observer.disconnect();
      }
    });
    observer.observe(container);

    return () => observer.disconnect();
  }, []);

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current || !containerReady) return;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    // Create radial gradient background
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const gradient = ctx.createRadialGradient(256, 256, 0, 256, 256, 400);
      gradient.addColorStop(0, "#1a1a2e");
      gradient.addColorStop(0.5, "#0f0f1a");
      gradient.addColorStop(1, "#050508");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 512, 512);
    }
    const backgroundTexture = new THREE.CanvasTexture(canvas);
    scene.background = backgroundTexture;

    const camera = new THREE.PerspectiveCamera(
      60,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000,
    );
    camera.position.z = CAMERA_Z_DEFAULT;
    camera.position.y = 0.1;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const wallContainer = new THREE.Group();
    scene.add(wallContainer);
    wallContainerRef.current = wallContainer;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, 5, 10);
    scene.add(directionalLight);

    const tileDataList: TileData[] = [];

    const bottomRowY = (ROWS / 2 - (ROWS - 1) - 0.5) * (TILE_HEIGHT + TILE_GAP_V);
    const wallBottom = bottomRowY - TILE_HEIGHT / 2;
    const reflectionGap = 0.15;
    const mirrorLine = wallBottom - reflectionGap;

    tiles.forEach((tile, index) => {
      const column = Math.floor(index / ROWS);
      const row = index % ROWS;

      // Check if this is a circular (face) tile - explicitly set via isCircular flag
      const isCircular = !!tile.metadata?.isCircular;
      // Check if tile has bbox (face cropping) - labels will be outside
      const hasBbox = !!tile.metadata?.bbox;

      // For circular tiles, use square dimensions
      const tileW = isCircular ? TILE_HEIGHT : TILE_WIDTH;
      const tileH = TILE_HEIGHT;

      const x = column * (TILE_WIDTH + TILE_GAP) - wallWidth / 2 + TILE_WIDTH / 2;
      // Use larger vertical gap for better spacing
      const y = (ROWS / 2 - row - 0.5) * (TILE_HEIGHT + TILE_GAP_V);

      // Create shadow mesh
      const shadowSize = isCircular ? tileH : tileW;
      const shadowGeometry = new THREE.PlaneGeometry(shadowSize + SHADOW_BLUR * 4, shadowSize + SHADOW_BLUR * 4);
      const shadowMaterial = createShadowMaterial(isCircular, shadowSize);
      const shadowMesh = new THREE.Mesh(shadowGeometry, shadowMaterial);
      shadowMesh.position.set(x + SHADOW_OFFSET_X, y + SHADOW_OFFSET_Y, SHADOW_OFFSET_Z);
      shadowMesh.userData = { isShadow: true };
      wallContainer.add(shadowMesh);

      // Create tile geometry
      const geometry = new THREE.PlaneGeometry(tileW, tileH);
      const material = createTileMaterial(false, isCircular);

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, y, 0);
      mesh.userData = { tile, index };
      wallContainer.add(mesh);

      // Create reflection mesh
      const reflectionGeometry = new THREE.PlaneGeometry(tileW, tileH);
      const reflectionMaterial = createTileMaterial(true, isCircular);
      const reflectionMesh = new THREE.Mesh(reflectionGeometry, reflectionMaterial);

      const reflectionY = 2 * mirrorLine - y;
      reflectionMesh.position.set(x, reflectionY, 0);
      reflectionMesh.scale.y = -1;
      reflectionMesh.userData = { isReflection: true };
      wallContainer.add(reflectionMesh);

      // Create label mesh for tiles with bbox (face cropping) - label is outside the face
      let labelMesh: THREE.Mesh | undefined;
      if (hasBbox && (tile.label || tile.metadata?.subtitle)) {
        const labelTexture = createLabelTexture(tile.label || "", tile.metadata?.subtitle);
        // Match canvas aspect ratio (512x80 = 6.4:1)
        const labelWidth = tileW * 1.4;
        const labelHeight = labelWidth * (80 / 512);
        const labelGeometry = new THREE.PlaneGeometry(labelWidth, labelHeight);
        const labelMaterial = new THREE.MeshBasicMaterial({
          map: labelTexture,
          transparent: true,
          side: THREE.FrontSide,
          depthWrite: false,
        });
        labelMesh = new THREE.Mesh(labelGeometry, labelMaterial);
        // Position label directly under tile, hugging the card above
        labelMesh.position.set(x, y - tileH / 2 - labelHeight / 2 + 0.08, 0.01);
        labelMesh.userData = { isLabel: true };
        wallContainer.add(labelMesh);
      }

      tileDataList.push({
        mesh,
        shadowMesh,
        reflectionMesh,
        labelMesh,
        tile,
        column,
        row,
        baseX: x,
        baseY: y,
        isCircular,
      });
    });

    tilesRef.current = tileDataList;

    // Check for return state
    const returnKey = `wall-return-${sessionKey}`;
    const positionKey = `wall-position-${sessionKey}`;
    const savedReturnState = sessionStorage.getItem(returnKey);
    const savedPosition = sessionStorage.getItem(positionKey);

    // For small walls, center the content (startPosition = 0)
    // For larger walls, start at the right edge
    const startPosition = maxXRef.current > 0 ? maxXRef.current : 0;

    let didSetupReturn = false;
    if (savedReturnState) {
      try {
        const { wallX, tileId } = JSON.parse(savedReturnState);
        sessionStorage.removeItem(returnKey);

        const returnTile = tileDataList.find((t) => t.tile.id === tileId);
        if (returnTile) {
          const zoomedWallX = -returnTile.baseX;
          wallPositionRef.current.x = zoomedWallX;
          wallContainer.position.x = zoomedWallX;
          camera.position.x = 0;
          camera.position.y = returnTile.baseY;
          camera.position.z = 0.5;

          const transition = zoomTransitionRef.current;
          transition.active = true;
          transition.direction = "out";
          transition.targetTile = returnTile.tile;
          transition.progress = 0;

          transition.startX = 0;
          transition.startY = returnTile.baseY;
          transition.startZ = 0.5;
          transition.startWallX = zoomedWallX;

          transition.targetX = 0;
          transition.targetY = 0.1;
          transition.targetZ = CAMERA_Z_DEFAULT;
          transition.targetWallX = wallX;

          setTransitionOpacity(1);
          didSetupReturn = true;
        } else if (typeof wallX === "number" && !Number.isNaN(wallX)) {
          wallPositionRef.current.x = wallX;
          wallContainer.position.x = wallX;
          didSetupReturn = true;
        }
      } catch {
        sessionStorage.removeItem(returnKey);
      }
    }

    if (!didSetupReturn) {
      if (savedPosition) {
        const wallX = parseFloat(savedPosition);
        if (!Number.isNaN(wallX)) {
          wallPositionRef.current.x = wallX;
          wallContainer.position.x = wallX;
        } else {
          wallPositionRef.current.x = startPosition;
          wallContainer.position.x = startPosition;
        }
      } else {
        wallPositionRef.current.x = startPosition;
        wallContainer.position.x = startPosition;
      }
    }

    setIsLoading(false);

    return () => {
      if (!zoomTransitionRef.current.active) {
        sessionStorage.setItem(positionKey, String(wallPositionRef.current.x));
      }

      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
        animationIdRef.current = 0;
      }

      if (rendererRef.current) {
        const domElement = rendererRef.current.domElement;
        if (domElement.parentNode) {
          domElement.parentNode.removeChild(domElement);
        }
        rendererRef.current.dispose();
        rendererRef.current = null;
      }

      loadedTexturesRef.current.forEach((texture) => {
        if (texture) texture.dispose();
      });
      loadedTexturesRef.current.clear();

      tilesRef.current.forEach((tileData) => {
        tileData.mesh.geometry.dispose();
        (tileData.mesh.material as THREE.Material).dispose();
        tileData.shadowMesh.geometry.dispose();
        (tileData.shadowMesh.material as THREE.Material).dispose();
        tileData.reflectionMesh.geometry.dispose();
        (tileData.reflectionMesh.material as THREE.Material).dispose();
        if (tileData.labelMesh) {
          tileData.labelMesh.geometry.dispose();
          (tileData.labelMesh.material as THREE.Material).dispose();
        }
      });

      sceneRef.current = null;
      cameraRef.current = null;
      wallContainerRef.current = null;
      tilesRef.current = [];
    };
  }, [tiles, wallWidth, sessionKey, containerReady]);

  // Start animation loop
  useEffect(() => {
    if (!isLoading) {
      animate();
    }
    return () => {
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
    };
  }, [isLoading, animate]);

  // Event listeners
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener("mousedown", handleMouseDown);
    container.addEventListener("mousemove", handleMouseMove);
    container.addEventListener("mouseup", handleMouseUp);
    container.addEventListener("mouseleave", handleMouseUp);
    container.addEventListener("click", handleClick);
    container.addEventListener("wheel", handleWheel, { passive: false });
    window.addEventListener("resize", handleResize);

    return () => {
      container.removeEventListener("mousedown", handleMouseDown);
      container.removeEventListener("mousemove", handleMouseMove);
      container.removeEventListener("mouseup", handleMouseUp);
      container.removeEventListener("mouseleave", handleMouseUp);
      container.removeEventListener("click", handleClick);
      container.removeEventListener("wheel", handleWheel);
      window.removeEventListener("resize", handleResize);
    };
  }, [handleMouseDown, handleMouseMove, handleMouseUp, handleClick, handleWheel, handleResize]);

  // Touch support
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isDraggingRef.current = true;
        wasDraggingRef.current = false;
        const touchStartX = e.touches[0].clientX;
        lastMouseXRef.current = touchStartX;
        dragStartXRef.current = touchStartX;
        totalDragDistanceRef.current = 0;
        lastMouseTimeRef.current = performance.now();
        flickVelocityRef.current = 0;
        wallPositionRef.current.velocityX = 0;
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 1 && isDraggingRef.current) {
        const touch = e.touches[0];
        const deltaX = touch.clientX - lastMouseXRef.current;
        const currentTime = performance.now();
        const deltaTime = Math.max(1, currentTime - lastMouseTimeRef.current);

        totalDragDistanceRef.current += Math.abs(deltaX);
        if (totalDragDistanceRef.current > MIN_DRAG_DISTANCE_FOR_CLICK) {
          wasDraggingRef.current = true;
        }

        flickVelocityRef.current = (deltaX / deltaTime) * FLICK_VELOCITY_MULTIPLIER;

        const dragSpeed = cameraZRef.current.z * 0.003;
        wallPositionRef.current.x += deltaX * dragSpeed;

        const maxX = maxXRef.current;
        wallPositionRef.current.x = Math.max(-maxX, Math.min(maxX, wallPositionRef.current.x));

        lastMouseXRef.current = touch.clientX;
        lastMouseTimeRef.current = currentTime;
      }
    };

    const handleTouchEnd = (e: TouchEvent) => {
      if (isDraggingRef.current) {
        wallPositionRef.current.velocityX = flickVelocityRef.current;

        if (!wasDraggingRef.current && e.changedTouches.length > 0) {
          const touch = e.changedTouches[0];
          const clickEvent = new MouseEvent("click", {
            clientX: touch.clientX,
            clientY: touch.clientY,
            bubbles: true,
          });
          container.dispatchEvent(clickEvent);
        }
      }
      isDraggingRef.current = false;
    };

    container.addEventListener("touchstart", handleTouchStart, { passive: true });
    container.addEventListener("touchmove", handleTouchMove, { passive: true });
    container.addEventListener("touchend", handleTouchEnd, { passive: true });

    return () => {
      container.removeEventListener("touchstart", handleTouchStart);
      container.removeEventListener("touchmove", handleTouchMove);
      container.removeEventListener("touchend", handleTouchEnd);
    };
  }, []);

  // Pinch zoom support
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let initialPinchDistance = 0;
    let initialZoom = CAMERA_Z_DEFAULT;

    const handleTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        initialPinchDistance = Math.sqrt(dx * dx + dy * dy);
        initialZoom = cameraZRef.current.targetZ;
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const scale = initialPinchDistance / distance;

        cameraZRef.current.targetZ = Math.max(CAMERA_Z_MIN, Math.min(CAMERA_Z_MAX, initialZoom * scale));
      }
    };

    container.addEventListener("touchstart", handleTouchStart, { passive: true });
    container.addEventListener("touchmove", handleTouchMove, { passive: true });

    return () => {
      container.removeEventListener("touchstart", handleTouchStart);
      container.removeEventListener("touchmove", handleTouchMove);
    };
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden relative bg-black">
      <div ref={containerRef} className="w-full h-full" style={{ cursor: "grab" }} />

      {/* Header overlay */}
      {headerContent && (
        <div className="absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/80 to-transparent pointer-events-none">
          <div className="pointer-events-auto">{headerContent}</div>
        </div>
      )}

      {/* Tile info overlay */}
      {hoveredTile && (
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent pointer-events-none">
          <div className="max-w-md">
            <p className="text-white font-medium truncate">{hoveredTile.label}</p>
            {hoveredTile.metadata?.subtitle && (
              <p className="text-white/70 text-sm truncate">{hoveredTile.metadata.subtitle}</p>
            )}
          </div>
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
          <div className="flex items-center gap-3 text-white">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span>Loading Photo Wall...</span>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 text-white/40 text-xs pointer-events-none">
        <p>Drag to pan | Scroll to zoom | Click to view</p>
      </div>

      {/* Zoom transition overlay */}
      {transitionOpacity > 0 && (
        <div className="absolute inset-0 bg-white pointer-events-none" style={{ opacity: transitionOpacity }} />
      )}
    </div>
  );
}
