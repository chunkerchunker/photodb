export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  imageWidth: number;
  imageHeight: number;
}

export interface WallTile {
  id: string | number;
  imageUrls: string[];
  label?: string;
  navigateTo?: string;
  metadata?: {
    subtitle?: string;
    count?: number;
    bbox?: BoundingBox;
    isCircular?: boolean; // Set to true for circular face tiles, defaults to false (rectangular)
  };
}

export interface PhotoWallProps {
  tiles: WallTile[];
  sessionKey: string;
  headerContent?: React.ReactNode;
  onTileClick?: (tile: WallTile) => void;
}

// Layout constants
export const ROWS = 4;
export const TILE_WIDTH = 2.0;
export const TILE_HEIGHT = 1.5;
export const TILE_GAP = 0.15;
export const TILE_GAP_V = 0.35; // Extra vertical gap for face views with labels

// Camera constants
export const CAMERA_Z_DEFAULT = 8;
export const CAMERA_Z_MIN = 3;
export const CAMERA_Z_MAX = 20;
export const ZOOM_SPEED = 0.5;

// Physics constants
export const FRICTION = 0.85;
export const FLICK_VELOCITY_MULTIPLIER = 3;
export const MIN_DRAG_DISTANCE_FOR_CLICK = 8;

// Visual effect constants
export const CORNER_RADIUS = 0.06;
export const VIGNETTE_STRENGTH = 0.25;
export const SHADOW_OFFSET_X = 0.04;
export const SHADOW_OFFSET_Y = -0.05;
export const SHADOW_OFFSET_Z = -0.02;
export const SHADOW_BLUR = 0.1;
export const SHADOW_OPACITY = 0.15;
