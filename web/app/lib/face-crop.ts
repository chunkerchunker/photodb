/**
 * Bounding box coordinates for a face crop.
 */
export interface FaceBbox {
  bbox_x: number;
  bbox_y: number;
  bbox_width: number;
  bbox_height: number;
}

/**
 * CSS styles for positioning and scaling an image to show a face crop.
 */
export interface FaceCropStyle {
  transform: string;
  transformOrigin: string;
  width: string;
  height: string;
}

/**
 * Calculates CSS styles to crop and display a face from a larger image.
 * The face is scaled to fit within the specified container size while
 * maintaining its aspect ratio.
 *
 * @param bbox - The bounding box coordinates of the face
 * @param imageWidth - The width of the full image in pixels
 * @param imageHeight - The height of the full image in pixels
 * @param containerSize - The size of the container to fit the face into (default: 128)
 * @returns CSS style properties for positioning the cropped face
 */
export function getFaceCropStyle(
  bbox: FaceBbox,
  imageWidth: number,
  imageHeight: number,
  containerSize = 128,
): FaceCropStyle {
  if (bbox.bbox_width <= 0 || bbox.bbox_height <= 0) {
    return { transform: "none", transformOrigin: "0 0", width: "100%", height: "100%" };
  }

  const scaleX = containerSize / bbox.bbox_width;
  const scaleY = containerSize / bbox.bbox_height;

  const left = -bbox.bbox_x * scaleX;
  const top = -bbox.bbox_y * scaleY;
  const width = imageWidth * scaleX;
  const height = imageHeight * scaleY;

  return {
    transform: `translate(${left}px, ${top}px)`,
    transformOrigin: "0 0",
    width: `${width}px`,
    height: `${height}px`,
  };
}
