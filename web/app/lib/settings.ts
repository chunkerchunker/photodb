/**
 * Project-level display settings for the PhotoDB web UI.
 * These settings control how data is filtered and displayed.
 */

export const displaySettings = {
  /**
   * Minimum confidence threshold for displaying tags (0-1).
   * Tags with confidence below this value are hidden.
   */
  minTagConfidence: 0.7,

  /**
   * Minimum face detection confidence for showing face tags (0-1).
   * Face tags are hidden if the face detection confidence is below this value.
   */
  minFaceConfidenceForTags: 0.5,

  /**
   * Minimum confidence for Apple Vision taxonomy labels (0-1).
   */
  minTaxonomyConfidence: 0.7,
};
