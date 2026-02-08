import { ArrowLeft, Camera, Grid, Loader2, User, Users } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { data, Link, useLocation, useNavigate } from "react-router";
import * as THREE from "three";
import { CoverflowIcon } from "~/components/coverflow-icon";
import { Header } from "~/components/header";
import { ViewSwitcher } from "~/components/view-switcher";
import { getPhotoCountByMonth, getPhotosByMonth } from "~/lib/db.server";
import type { Route } from "./+types/month.wall";

export function meta({ params }: Route.MetaArgs) {
  const monthNames = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  const monthName = monthNames[parseInt(params.month, 10)] || params.month;

  return [
    { title: `PhotoDB - ${monthName} ${params.year} - 3D Wall` },
    {
      name: "description",
      content: `Browse photos from ${monthName} ${params.year} in 3D wall view`,
    },
  ];
}

import { dataWithViewMode } from "~/lib/cookies.server";

export async function loader({ params }: Route.LoaderArgs) {
  const year = parseInt(params.year, 10);
  const month = parseInt(params.month, 10);

  const monthNames = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];
  const monthName = monthNames[month] || `Month ${month}`;

  try {
    // Load all photos for the 3D wall (up to a reasonable limit)
    const photos = await getPhotosByMonth(year, month, 500, 0);
    const totalPhotos = await getPhotoCountByMonth(year, month);

    return dataWithViewMode(
      {
        photos,
        totalPhotos,
        year: params.year,
        month: params.month,
        monthName,
      },
      "wall",
    );
  } catch (error) {
    console.error(`Failed to load photos for ${year}-${month}:`, error);
    return dataWithViewMode(
      {
        photos: [],
        totalPhotos: 0,
        year: params.year,
        month: params.month,
        monthName,
      },
      "wall",
    );
  }
}

type Photo = Route.ComponentProps["loaderData"]["photos"][number];

// Physics constants - friction-based deceleration (no spring)
const FRICTION = 0.85; // Velocity multiplier per frame (lower = more friction, faster stop)
const FLICK_VELOCITY_MULTIPLIER = 3; // Initial velocity from flick gesture
const MIN_DRAG_DISTANCE_FOR_CLICK = 8; // Pixels - drags shorter than this count as clicks

// Layout constants
const ROWS = 4;
const TILE_WIDTH = 2.0;
const TILE_HEIGHT = 1.5;
const TILE_GAP = 0.15;
const CURVE_FACTOR = 0.04;
const Z_OFFSET_FACTOR = 0.15;

// Camera constants
const CAMERA_Z_DEFAULT = 8;
const CAMERA_Z_MIN = 3;
const CAMERA_Z_MAX = 20;
const ZOOM_SPEED = 0.5;

// Visual effect constants
const CORNER_RADIUS = 0.06; // Subtle rounded corners (0.0 to 0.5)
const VIGNETTE_STRENGTH = 0.25; // Subtle edge darkening
const SHADOW_OFFSET_X = 0.04; // Shadow offset right
const SHADOW_OFFSET_Y = -0.05; // Shadow offset down
const SHADOW_OFFSET_Z = -0.02; // Shadow offset back
const SHADOW_BLUR = 0.1; // Shadow blur/spread amount
const SHADOW_OPACITY = 0.15; // Shadow darkness

// Custom shader for rounded corners and vignette effect
const tileVertexShader = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;

  void main() {
    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const tileFragmentShader = `
  uniform sampler2D map;
  uniform vec3 baseColor;
  uniform bool hasTexture;
  uniform float opacity;
  uniform float cornerRadius;
  uniform float vignetteStrength;
  uniform float imageAspect;
  uniform float tileAspect;

  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vWorldPosition;

  // Signed distance function for rounded rectangle
  float roundedBoxSDF(vec2 p, vec2 halfSize, float radius) {
    vec2 q = abs(p) - halfSize + radius;
    return length(max(q, 0.0)) - radius;
  }

  void main() {
    // Calculate distance from rounded rectangle edge
    vec2 centeredUv = vUv * 2.0 - 1.0;
    float halfSize = 1.0 - cornerRadius;
    float d = roundedBoxSDF(centeredUv, vec2(halfSize), cornerRadius);

    // Discard pixels outside rounded corners
    if (d > 0.0) discard;

    // Anti-aliased edge for rounded corners
    float edgeSoftness = fwidth(d) * 1.5;
    float edgeAlpha = 1.0 - smoothstep(-edgeSoftness, edgeSoftness, d);

    // Distance from center for vignette
    float dist = length(vUv - 0.5) * 2.0;

    // Adjust UVs for aspect ratio (object-fit: cover)
    vec2 adjustedUv = vUv;
    if (hasTexture && imageAspect > 0.0) {
      float aspectRatio = imageAspect / tileAspect;
      if (aspectRatio > 1.0) {
        // Image is wider than tile - crop sides
        adjustedUv.x = (vUv.x - 0.5) / aspectRatio + 0.5;
      } else {
        // Image is taller than tile - crop top/bottom
        adjustedUv.y = (vUv.y - 0.5) * aspectRatio + 0.5;
      }
    }

    // Get base color from texture or uniform
    vec3 texColor = hasTexture ? texture2D(map, adjustedUv).rgb : baseColor;

    // For placeholder tiles (no texture), apply simple lighting for depth
    // For actual photos, show at full brightness
    vec3 litColor = texColor;
    if (!hasTexture) {
      vec3 ambient = texColor * 0.6;
      vec3 lightDir = normalize(vec3(0.0, 5.0, 10.0));
      float diff = max(dot(vNormal, lightDir), 0.0);
      vec3 diffuse = texColor * diff * 0.8;
      litColor = ambient + diffuse;
    }

    // Vignette effect - darken edges only
    float vignette = 1.0 - pow(dist, 2.5) * vignetteStrength;

    vec3 finalColor = litColor * vignette;

    gl_FragColor = vec4(finalColor, opacity * edgeAlpha);
  }
`;

// Shadow shader - soft blurred dark rectangle
const shadowFragmentShader = `
  uniform float opacity;
  uniform float blur;
  uniform float cornerRadius;
  uniform float innerScale;

  varying vec2 vUv;

  // Signed distance function for rounded rectangle
  float roundedBoxSDF(vec2 p, vec2 halfSize, float radius) {
    vec2 q = abs(p) - halfSize + radius;
    return length(max(q, 0.0)) - radius;
  }

  void main() {
    vec2 centeredUv = vUv * 2.0 - 1.0;
    // Scale the box to match tile size within larger geometry
    float halfSize = innerScale - cornerRadius;
    float d = roundedBoxSDF(centeredUv, vec2(halfSize), cornerRadius);

    // Soft shadow falloff - blur is in UV space
    float alpha = 1.0 - smoothstep(-blur * 0.5, blur, d);

    gl_FragColor = vec4(0.0, 0.0, 0.0, opacity * alpha);
  }
`;

// Create shadow material
function createShadowMaterial(): THREE.ShaderMaterial {
  // Scale to match tile size within larger shadow geometry
  const innerScale = TILE_WIDTH / (TILE_WIDTH + SHADOW_BLUR * 4);
  const shadowCornerRadius = CORNER_RADIUS * innerScale;
  // Blur in UV space (geometry spans -1 to 1, so 2 units)
  const blurUV = ((SHADOW_BLUR * 2) / (TILE_WIDTH + SHADOW_BLUR * 4)) * 2;

  return new THREE.ShaderMaterial({
    uniforms: {
      opacity: { value: SHADOW_OPACITY },
      blur: { value: blurUV },
      cornerRadius: { value: shadowCornerRadius },
      innerScale: { value: innerScale },
    },
    vertexShader: tileVertexShader,
    fragmentShader: shadowFragmentShader,
    transparent: true,
    side: THREE.FrontSide,
    depthWrite: false,
  });
}

// Create shader material for tiles
function createTileMaterial(isReflection: boolean = false): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      map: { value: null },
      baseColor: { value: new THREE.Color(0x2a2a3e) },
      hasTexture: { value: false },
      opacity: { value: isReflection ? 0.12 : 1.0 },
      cornerRadius: { value: CORNER_RADIUS },
      vignetteStrength: { value: VIGNETTE_STRENGTH },
      imageAspect: { value: 0.0 },
      tileAspect: { value: TILE_WIDTH / TILE_HEIGHT },
    },
    vertexShader: tileVertexShader,
    fragmentShader: tileFragmentShader,
    transparent: true,
    side: THREE.FrontSide,
  });
}

interface TileData {
  mesh: THREE.Mesh;
  shadowMesh: THREE.Mesh;
  reflectionMesh: THREE.Mesh;
  photo: Photo;
  column: number;
  row: number;
  baseX: number;
  baseY: number;
}

interface ThreeWallProps {
  photos: Photo[];
  year: string;
  month: string;
  totalPhotos: number;
  monthName: string;
}

function ThreeWall({ photos, year, month, totalPhotos, monthName }: ThreeWallProps) {
  const navigate = useNavigate();

  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const tilesRef = useRef<TileData[]>([]);
  const wallContainerRef = useRef<THREE.Group | null>(null);
  const textureLoaderRef = useRef<THREE.TextureLoader | null>(null);
  const loadedTexturesRef = useRef<Map<number, THREE.Texture>>(new Map());
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
  const wasDraggingRef = useRef(false); // Track if a real drag occurred

  // Zoom transition state
  const zoomTransitionRef = useRef<{
    active: boolean;
    direction: "in" | "out";
    targetPhoto: Photo | null;
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
    targetPhoto: null,
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
  const [hoveredPhoto, setHoveredPhoto] = useState<Photo | null>(null);
  const [transitionOpacity, setTransitionOpacity] = useState(0);

  // Calculate wall dimensions
  const columns = Math.ceil(photos.length / ROWS);
  const wallWidth = columns * (TILE_WIDTH + TILE_GAP);

  // Store wallWidth in a ref to avoid stale closure issues
  const wallWidthRef = useRef(wallWidth);
  wallWidthRef.current = wallWidth;

  // maxX is the maximum scroll distance from center; ensure it's never negative
  const maxX = Math.max(0, wallWidth / 2 - 2);
  const maxXRef = useRef(maxX);
  maxXRef.current = maxX;

  const loadVisibleTextures = useCallback(() => {
    if (!textureLoaderRef.current || !cameraRef.current) return;

    const viewWidth = cameraZRef.current.z * 2;

    tilesRef.current.forEach((tile) => {
      // Check if tile is near the visible area
      const tileWorldX = tile.baseX + wallPositionRef.current.x;
      const isVisible = Math.abs(tileWorldX) < viewWidth;

      if (isVisible && !loadedTexturesRef.current.has(tile.photo.id)) {
        // Mark as loading
        loadedTexturesRef.current.set(tile.photo.id, null as unknown as THREE.Texture);

        textureLoaderRef.current?.load(
          `/api/image/${tile.photo.id}`,
          (texture) => {
            // Don't set colorSpace - let the texture pass through as-is
            // (sRGB conversion would require gamma correction in shader output)
            loadedTexturesRef.current.set(tile.photo.id, texture);

            // Calculate image aspect ratio
            const imageAspect = texture.image.width / texture.image.height;

            // Update main mesh shader material
            const material = tile.mesh.material as THREE.ShaderMaterial;
            material.uniforms.map.value = texture;
            material.uniforms.hasTexture.value = true;
            material.uniforms.imageAspect.value = imageAspect;
            material.needsUpdate = true;

            // Update reflection mesh shader material
            const reflectionMaterial = tile.reflectionMesh.material as THREE.ShaderMaterial;
            reflectionMaterial.uniforms.map.value = texture;
            reflectionMaterial.uniforms.hasTexture.value = true;
            reflectionMaterial.uniforms.imageAspect.value = imageAspect;
            reflectionMaterial.needsUpdate = true;
          },
          undefined,
          () => {
            // On error, remove from map so it can retry
            loadedTexturesRef.current.delete(tile.photo.id);
          },
        );
      }
    });
  }, []);

  const updateTileCurve = useCallback(() => {
    if (!cameraRef.current) return;

    tilesRef.current.forEach((tile) => {
      // Calculate world position of tile
      const worldX = tile.baseX + wallPositionRef.current.x;

      // True cylindrical curve - position and rotation derived from same angle
      const cylinderRadius = 40; // Fixed radius of the virtual cylinder
      const angle = worldX / cylinderRadius; // Angle on the cylinder surface
      const clampedAngle = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, angle));

      // Position on cylinder surface (in world space)
      const curveX = cylinderRadius * Math.sin(clampedAngle);
      const offsetZ = -cylinderRadius * (1 - Math.cos(clampedAngle));

      // Rotation matches the surface tangent
      const rotationY = clampedAngle;

      tile.mesh.rotation.y = rotationY;
      tile.mesh.position.z = offsetZ;
      // Convert world curveX back to local position (relative to wall container)
      tile.mesh.position.x = curveX - wallPositionRef.current.x;

      // Update shadow (same rotation, offset position relative to center light)
      // Shadow X offset based on position - shadows point away from center
      const shadowOffsetX = clampedAngle * 0.5; // Shadows spread outward from center
      tile.shadowMesh.rotation.y = rotationY;
      tile.shadowMesh.position.z = offsetZ + SHADOW_OFFSET_Z;
      tile.shadowMesh.position.x = curveX - wallPositionRef.current.x + shadowOffsetX;
      tile.shadowMesh.position.y = tile.baseY + SHADOW_OFFSET_Y;

      // Update reflection
      tile.reflectionMesh.rotation.y = rotationY;
      tile.reflectionMesh.position.z = offsetZ;
      tile.reflectionMesh.position.x = curveX - wallPositionRef.current.x;

      // Dim reflection based on distance from center (simulates overhead lighting)
      const distanceFromCenter = Math.abs(clampedAngle) / (Math.PI / 3);
      const baseOpacity = 0.12;
      const falloff = Math.max(0, 1 - distanceFromCenter * 0.8);
      const reflectionMaterial = tile.reflectionMesh.material as THREE.ShaderMaterial;
      reflectionMaterial.uniforms.opacity.value = baseOpacity * falloff;
    });
  }, []);

  const animate = useCallback(() => {
    if (!sceneRef.current || !cameraRef.current || !rendererRef.current || !wallContainerRef.current) return;

    const transition = zoomTransitionRef.current;

    if (transition.active) {
      // Zoom transition animation
      transition.progress += 0.025; // Speed of transition

      // Easing function (ease-in-out cubic)
      const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
      const easedProgress = easeInOutCubic(Math.min(transition.progress, 1));

      // Interpolate camera position
      cameraRef.current.position.x = transition.startX + (transition.targetX - transition.startX) * easedProgress;
      cameraRef.current.position.y = transition.startY + (transition.targetY - transition.startY) * easedProgress;
      cameraRef.current.position.z = transition.startZ + (transition.targetZ - transition.startZ) * easedProgress;

      // Interpolate wall position
      wallPositionRef.current.x =
        transition.startWallX + (transition.targetWallX - transition.startWallX) * easedProgress;
      wallContainerRef.current.position.x = wallPositionRef.current.x;

      if (transition.direction === "in") {
        // Zoom IN: Fade to white as we get closer, then navigate
        const fadeStart = 0.6;
        if (transition.progress > fadeStart) {
          const fadeProgress = (transition.progress - fadeStart) / (1 - fadeStart);
          setTransitionOpacity(fadeProgress);
        }

        // Navigate when transition completes
        if (transition.progress >= 1) {
          transition.active = false;
          if (transition.targetPhoto) {
            // Save state before navigating (for zoom-out animation on return)
            const photoReturnKey = `wall-photo-return-${year}-${month}`;
            sessionStorage.setItem(
              photoReturnKey,
              JSON.stringify({
                wallX: transition.startWallX,
                photoId: transition.targetPhoto.id,
              }),
            );
            navigate(`/photo/${transition.targetPhoto.id}`, { state: { fromWall: true } });
          }
        }
      } else {
        // Zoom OUT: Fade from white at the start
        const fadeEnd = 0.4;
        if (transition.progress < fadeEnd) {
          const fadeProgress = 1 - transition.progress / fadeEnd;
          setTransitionOpacity(fadeProgress);
        } else {
          setTransitionOpacity(0);
        }

        // Just end the transition when complete
        if (transition.progress >= 1) {
          transition.active = false;
          setTransitionOpacity(0);
        }
      }
    } else {
      // Normal wall animation
      // Apply friction-based deceleration (no spring - just velocity decay)
      if (!isDraggingRef.current) {
        wallPositionRef.current.velocityX *= FRICTION;
        // Stop when velocity is negligible
        if (Math.abs(wallPositionRef.current.velocityX) < 0.0001) {
          wallPositionRef.current.velocityX = 0;
        }
      }
      wallPositionRef.current.x += wallPositionRef.current.velocityX;

      // Soft boundary - gradually slow down near edges
      const maxX = maxXRef.current;
      const softZone = 3; // Start slowing down this far from the edge
      const pos = wallPositionRef.current.x;

      if (Math.abs(pos) > maxX - softZone) {
        // Calculate how far into the soft zone we are (0 to 1)
        const distanceIntoSoftZone = (Math.abs(pos) - (maxX - softZone)) / softZone;
        // Apply extra friction based on how deep into soft zone
        const edgeFriction = 1 - Math.min(distanceIntoSoftZone, 1) * 0.5;
        wallPositionRef.current.velocityX *= edgeFriction;
      }

      // Hard clamp as failsafe
      if (wallPositionRef.current.x > maxX) {
        wallPositionRef.current.x = maxX;
        wallPositionRef.current.velocityX = 0;
      } else if (wallPositionRef.current.x < -maxX) {
        wallPositionRef.current.x = -maxX;
        wallPositionRef.current.velocityX = 0;
      }

      // Smooth camera zoom
      const dz = cameraZRef.current.targetZ - cameraZRef.current.z;
      cameraZRef.current.z += dz * 0.1;
      cameraRef.current.position.z = cameraZRef.current.z;

      // Update wall container position
      wallContainerRef.current.position.x = wallPositionRef.current.x;
    }

    // Update tile curves based on their position relative to camera
    updateTileCurve();

    // Load textures for visible tiles
    loadVisibleTextures();

    // Render
    rendererRef.current.render(sceneRef.current, cameraRef.current);

    animationIdRef.current = requestAnimationFrame(animate);
  }, [updateTileCurve, loadVisibleTextures]);

  const handleMouseDown = useCallback((e: MouseEvent) => {
    isDraggingRef.current = true;
    wasDraggingRef.current = false;
    lastMouseXRef.current = e.clientX;
    dragStartXRef.current = e.clientX;
    totalDragDistanceRef.current = 0;
    lastMouseTimeRef.current = performance.now();
    flickVelocityRef.current = 0;
    // Stop any ongoing momentum
    wallPositionRef.current.velocityX = 0;
  }, []);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!containerRef.current || !cameraRef.current) return;

    if (isDraggingRef.current) {
      const deltaX = e.clientX - lastMouseXRef.current;
      const currentTime = performance.now();
      const deltaTime = Math.max(1, currentTime - lastMouseTimeRef.current);

      // Track total drag distance
      totalDragDistanceRef.current += Math.abs(deltaX);
      if (totalDragDistanceRef.current > MIN_DRAG_DISTANCE_FOR_CLICK) {
        wasDraggingRef.current = true;
      }

      // Calculate flick velocity
      flickVelocityRef.current = (deltaX / deltaTime) * FLICK_VELOCITY_MULTIPLIER;

      // Update position directly based on drag
      const dragSpeed = cameraZRef.current.z * 0.003;
      wallPositionRef.current.x += deltaX * dragSpeed;

      // Clamp position to wall bounds
      const maxX = maxXRef.current;
      wallPositionRef.current.x = Math.max(-maxX, Math.min(maxX, wallPositionRef.current.x));

      lastMouseXRef.current = e.clientX;
      lastMouseTimeRef.current = currentTime;
    }

    // Raycasting for hover
    if (!sceneRef.current || !cameraRef.current) return;

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (e.clientX / containerRef.current.clientWidth) * 2 - 1,
      -(e.clientY / containerRef.current.clientHeight) * 2 + 1,
    );

    raycaster.setFromCamera(mouse, cameraRef.current);
    const intersects = raycaster.intersectObjects(wallContainerRef.current?.children || [], false);

    const hit = intersects.find((i) => i.object.userData.photo && !i.object.userData.isReflection);
    if (hit) {
      setHoveredPhoto(hit.object.userData.photo);
      containerRef.current.style.cursor = "pointer";
    } else {
      setHoveredPhoto(null);
      containerRef.current.style.cursor = isDraggingRef.current ? "grabbing" : "grab";
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    if (isDraggingRef.current) {
      // Apply flick velocity
      wallPositionRef.current.velocityX = flickVelocityRef.current;
    }
    isDraggingRef.current = false;
    if (containerRef.current) {
      containerRef.current.style.cursor = "grab";
    }
  }, []);

  const handleClick = useCallback(
    (e: MouseEvent) => {
      if (!containerRef.current || !sceneRef.current || !cameraRef.current) return;

      // Don't handle clicks during transition
      if (zoomTransitionRef.current.active) return;

      // Only navigate if this wasn't a drag (drag distance was minimal)
      if (wasDraggingRef.current) {
        return;
      }

      const raycaster = new THREE.Raycaster();
      const mouse = new THREE.Vector2(
        (e.clientX / containerRef.current.clientWidth) * 2 - 1,
        -(e.clientY / containerRef.current.clientHeight) * 2 + 1,
      );

      raycaster.setFromCamera(mouse, cameraRef.current);
      const intersects = raycaster.intersectObjects(wallContainerRef.current?.children || [], false);

      const hit = intersects.find((i) => i.object.userData.photo && !i.object.userData.isReflection);
      if (hit) {
        const photo = hit.object.userData.photo as Photo;
        const tile = tilesRef.current.find((t) => t.photo.id === photo.id);

        if (tile) {
          // Start zoom transition
          const transition = zoomTransitionRef.current;
          transition.active = true;
          transition.direction = "in";
          transition.targetPhoto = photo;
          transition.progress = 0;

          // Store starting positions
          transition.startX = cameraRef.current.position.x;
          transition.startY = cameraRef.current.position.y;
          transition.startZ = cameraRef.current.position.z;
          transition.startWallX = wallPositionRef.current.x;

          // Calculate target position (fly into the tile)
          // Target X: center on the tile
          transition.targetX = 0;
          // Target Y: tile's Y position
          transition.targetY = tile.baseY;
          // Target Z: very close to the tile
          transition.targetZ = 0.5;
          // Target wall X: center the tile
          transition.targetWallX = -tile.baseX;

          // Stop any ongoing momentum
          wallPositionRef.current.velocityX = 0;
        }
      }
    },
    [navigate],
  );

  const handleWheel = useCallback((e: WheelEvent) => {
    e.preventDefault();

    // Don't handle during transition
    if (zoomTransitionRef.current.active) return;

    // Horizontal scroll pans the wall
    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      const panSpeed = cameraZRef.current.z * 0.002;
      wallPositionRef.current.x -= e.deltaX * panSpeed;

      // Clamp position to wall bounds
      const maxX = maxXRef.current;
      wallPositionRef.current.x = Math.max(-maxX, Math.min(maxX, wallPositionRef.current.x));
    } else {
      // Vertical scroll zooms in/out
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

  // Initialize scene
  useEffect(() => {
    if (!containerRef.current) return;

    // Create scene
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

    // Create camera
    const camera = new THREE.PerspectiveCamera(
      60,
      containerRef.current.clientWidth / containerRef.current.clientHeight,
      0.1,
      1000,
    );
    camera.position.z = CAMERA_Z_DEFAULT;
    camera.position.y = 0.1; // Shifted to center wall between header and bottom
    cameraRef.current = camera;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Create wall container
    const wallContainer = new THREE.Group();
    scene.add(wallContainer);
    wallContainerRef.current = wallContainer;

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    // Add directional light
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, 5, 10);
    scene.add(directionalLight);

    // Create texture loader
    textureLoaderRef.current = new THREE.TextureLoader();

    // Create tiles
    const tiles: TileData[] = [];

    // Calculate the bottom edge of the wall for reflection positioning
    const bottomRowY = (ROWS / 2 - (ROWS - 1) - 0.5) * (TILE_HEIGHT + TILE_GAP);
    const wallBottom = bottomRowY - TILE_HEIGHT / 2;
    const reflectionGap = 0.15; // Gap between wall and reflection
    const mirrorLine = wallBottom - reflectionGap;

    photos.forEach((photo, index) => {
      const column = Math.floor(index / ROWS);
      const row = index % ROWS;

      const x = column * (TILE_WIDTH + TILE_GAP) - wallWidth / 2 + TILE_WIDTH / 2;
      const y = (ROWS / 2 - row - 0.5) * (TILE_HEIGHT + TILE_GAP);

      // Create shadow mesh (rendered behind the tile)
      // Extra padding (*4) to ensure blur fades out completely without hard edges
      const shadowGeometry = new THREE.PlaneGeometry(TILE_WIDTH + SHADOW_BLUR * 4, TILE_HEIGHT + SHADOW_BLUR * 4);
      const shadowMaterial = createShadowMaterial();
      const shadowMesh = new THREE.Mesh(shadowGeometry, shadowMaterial);
      shadowMesh.position.set(x + SHADOW_OFFSET_X, y + SHADOW_OFFSET_Y, SHADOW_OFFSET_Z);
      shadowMesh.userData = { isShadow: true };
      wallContainer.add(shadowMesh);

      // Create tile geometry
      const geometry = new THREE.PlaneGeometry(TILE_WIDTH, TILE_HEIGHT);

      // Create shader material with rounded corners and vignette
      const material = createTileMaterial(false);

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, y, 0);
      mesh.userData = { photo, index };
      wallContainer.add(mesh);

      // Create reflection mesh
      const reflectionGeometry = new THREE.PlaneGeometry(TILE_WIDTH, TILE_HEIGHT);
      const reflectionMaterial = createTileMaterial(true);
      const reflectionMesh = new THREE.Mesh(reflectionGeometry, reflectionMaterial);

      // Position reflection mirrored around the wall's bottom edge
      const reflectionY = 2 * mirrorLine - y;
      reflectionMesh.position.set(x, reflectionY, 0);
      reflectionMesh.scale.y = -1;
      reflectionMesh.userData = { isReflection: true };
      wallContainer.add(reflectionMesh);

      tiles.push({
        mesh,
        shadowMesh,
        reflectionMesh,
        photo,
        column,
        row,
        baseX: x,
        baseY: y,
      });
    });

    tilesRef.current = tiles;

    // Check if returning from a photo - set up zoom-out animation
    const photoReturnKey = `wall-photo-return-${year}-${month}`;
    const positionKey = `wall-position-${year}-${month}`;
    const savedPhotoState = sessionStorage.getItem(photoReturnKey);
    const savedPosition = sessionStorage.getItem(positionKey);

    // Default to start position (left side of wall)
    // For small walls, center the content (startPosition = 0)
    // For larger walls, start at the right edge
    const startPosition = maxXRef.current > 0 ? maxXRef.current : 0;

    let didSetupPhotoReturn = false;
    if (savedPhotoState) {
      // Returning from a photo view - do zoom-out animation
      try {
        const { wallX, photoId } = JSON.parse(savedPhotoState);
        sessionStorage.removeItem(photoReturnKey);

        // Find the tile for the photo we're returning from
        const returnTile = tiles.find((t) => t.photo.id === photoId);
        if (returnTile) {
          // Start zoomed into the photo
          const zoomedWallX = -returnTile.baseX;
          wallPositionRef.current.x = zoomedWallX;
          wallContainer.position.x = zoomedWallX;
          camera.position.x = 0;
          camera.position.y = returnTile.baseY;
          camera.position.z = 0.5;

          // Set up zoom-out transition
          const transition = zoomTransitionRef.current;
          transition.active = true;
          transition.direction = "out";
          transition.targetPhoto = returnTile.photo;
          transition.progress = 0;

          // Start positions (zoomed in)
          transition.startX = 0;
          transition.startY = returnTile.baseY;
          transition.startZ = 0.5;
          transition.startWallX = zoomedWallX;

          // Target positions (normal view, restored wall position)
          transition.targetX = 0;
          transition.targetY = 0.1;
          transition.targetZ = CAMERA_Z_DEFAULT;
          transition.targetWallX = wallX; // Restore to original position

          // Start with white overlay
          setTransitionOpacity(1);
          didSetupPhotoReturn = true;
        } else if (typeof wallX === "number" && !isNaN(wallX)) {
          // Tile not found, just restore position without animation
          wallPositionRef.current.x = wallX;
          wallContainer.position.x = wallX;
          didSetupPhotoReturn = true;
        }
      } catch {
        // Invalid JSON, will fall through to other cases
        sessionStorage.removeItem(photoReturnKey);
      }
    }

    if (!didSetupPhotoReturn) {
      if (savedPosition) {
        // Returning to wall (not from photo) - restore last position
        const wallX = parseFloat(savedPosition);
        if (!isNaN(wallX)) {
          wallPositionRef.current.x = wallX;
          wallContainer.position.x = wallX;
        } else {
          // Invalid saved position, use default
          wallPositionRef.current.x = startPosition;
          wallContainer.position.x = startPosition;
        }
      } else {
        // First visit - start at the beginning (left side)
        wallPositionRef.current.x = startPosition;
        wallContainer.position.x = startPosition;
      }
    }

    setIsLoading(false);

    return () => {
      // Save current position for when returning to this wall
      // Only save if not in the middle of a zoom transition (which saves its own state)
      if (!zoomTransitionRef.current.active) {
        const positionKey = `wall-position-${year}-${month}`;
        sessionStorage.setItem(positionKey, String(wallPositionRef.current.x));
      }

      // Cancel animation
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
        animationIdRef.current = 0;
      }

      // Dispose and remove renderer
      if (rendererRef.current) {
        const domElement = rendererRef.current.domElement;
        if (domElement.parentNode) {
          domElement.parentNode.removeChild(domElement);
        }
        rendererRef.current.dispose();
        rendererRef.current = null;
      }

      // Dispose textures
      loadedTexturesRef.current.forEach((texture) => {
        if (texture) texture.dispose();
      });
      loadedTexturesRef.current.clear();

      // Dispose geometries and materials from tiles
      tilesRef.current.forEach((tile) => {
        tile.mesh.geometry.dispose();
        (tile.mesh.material as THREE.Material).dispose();
        tile.shadowMesh.geometry.dispose();
        (tile.shadowMesh.material as THREE.Material).dispose();
        tile.reflectionMesh.geometry.dispose();
        (tile.reflectionMesh.material as THREE.Material).dispose();
      });

      // Clear all refs
      sceneRef.current = null;
      cameraRef.current = null;
      wallContainerRef.current = null;
      textureLoaderRef.current = null;
      tilesRef.current = [];
    };
  }, [photos, wallWidth]);

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

    let touchStartX = 0;

    const handleTouchStart = (e: TouchEvent) => {
      if (e.touches.length === 1) {
        isDraggingRef.current = true;
        wasDraggingRef.current = false;
        touchStartX = e.touches[0].clientX;
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

        // Track total drag distance
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

        // Check for tap (not a drag)
        if (!wasDraggingRef.current && e.changedTouches.length > 0) {
          // Simulate click for tap
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

    container.addEventListener("touchstart", handleTouchStart, {
      passive: true,
    });
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

    container.addEventListener("touchstart", handleTouchStart, {
      passive: true,
    });
    container.addEventListener("touchmove", handleTouchMove, { passive: true });

    return () => {
      container.removeEventListener("touchstart", handleTouchStart);
      container.removeEventListener("touchmove", handleTouchMove);
    };
  }, []);

  return (
    <div className="h-screen w-screen overflow-hidden relative bg-black">
      {/* Three.js container */}
      <div ref={containerRef} className="w-full h-full" style={{ cursor: "grab" }} />

      {/* Header overlay */}
      <Header
        homeTo="/wall"
        breadcrumbs={[{ label: year.toString(), to: `/year/${year}/wall` }, { label: monthName }]}
        viewAction={
          <ViewSwitcher
            modes={[
              {
                key: "grid",
                label: "Grid View",
                icon: <Grid className="h-4 w-4" />,
                to: `/year/${year}/month/${month}/grid`,
                isActive: false,
              },
              {
                key: "wall",
                label: "3D Wall",
                icon: <CoverflowIcon className="h-4 w-4" />,
                isActive: true,
              },
            ]}
          />
        }
      />

      {/* Photo info overlay */}
      {hoveredPhoto && (
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent pointer-events-none">
          <div className="max-w-md">
            <p className="text-white font-medium truncate">{hoveredPhoto.filename_only}</p>
            {hoveredPhoto.short_description && (
              <p className="text-white/70 text-sm truncate">{hoveredPhoto.short_description}</p>
            )}
          </div>
        </div>
      )}

      {/* Loading overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center">
          <div className="flex items-center gap-3 text-white">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span>Loading 3D Wall...</span>
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="absolute bottom-4 right-4 text-white/40 text-xs pointer-events-none">
        <p>Drag to pan | Scroll to zoom | Click photo to view</p>
      </div>

      {/* Zoom transition overlay */}
      {transitionOpacity > 0 && (
        <div className="absolute inset-0 bg-white pointer-events-none" style={{ opacity: transitionOpacity }} />
      )}
    </div>
  );
}

export default function MonthWallView({ loaderData }: Route.ComponentProps) {
  const { photos, totalPhotos, year, month, monthName } = loaderData;
  const location = useLocation();

  if (photos.length === 0) {
    return (
      <div className="h-screen w-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-400 text-lg mb-4">
            No photos found for {monthName} {year}.
          </p>
          <Link to={`/year/${year}`} className="text-blue-400 hover:text-blue-300 inline-flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to {year}
          </Link>
        </div>
      </div>
    );
  }

  // Use location.key to force complete remount when navigating back
  // This ensures Three.js is properly re-initialized after navigation
  return (
    <ThreeWall
      key={location.key}
      photos={photos}
      year={year}
      month={month}
      totalPhotos={totalPhotos}
      monthName={monthName}
    />
  );
}
