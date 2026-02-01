import * as THREE from "three";
import { CORNER_RADIUS, SHADOW_BLUR, SHADOW_OPACITY, TILE_HEIGHT, TILE_WIDTH, VIGNETTE_STRENGTH } from "./types";

export const tileVertexShader = `
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

export const tileFragmentShader = `
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

export const shadowFragmentShader = `
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

export function createShadowMaterial(): THREE.ShaderMaterial {
  const innerScale = TILE_WIDTH / (TILE_WIDTH + SHADOW_BLUR * 4);
  const shadowCornerRadius = CORNER_RADIUS * innerScale;
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

export function createTileMaterial(isReflection: boolean = false): THREE.ShaderMaterial {
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
