import path from "node:path";
import dotenv from "dotenv";
import { Pool } from "pg";

// Load environment variables from .env file
dotenv.config({ path: path.join(process.cwd(), "..", ".env") });

// Create a connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || "postgresql://localhost/photodb",
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

/**
 * Get the default collection ID from environment variable.
 * Used for backward compatibility (e.g., CLI tools) when collectionId is not provided.
 */
function getDefaultCollectionId(): number {
  const raw = process.env.COLLECTION_ID || "1";
  const parsed = Number.parseInt(raw, 10);
  return Number.isNaN(parsed) ? 1 : parsed;
}

export { getDefaultCollectionId };

// Query functions that match the Python PhotoQueries class

export async function getYearsWithPhotos(collectionId: number) {
  // Single query with LATERAL join for sample photos
  // Uses day-of-year to create daily-varying but deterministic selection
  // The modulo with a prime (997) spreads photos across the ID space
  const query = `
    SELECT
      y.year,
      y.photo_count,
      COALESCE(s.sample_ids, ARRAY[]::int[]) as sample_photo_ids
    FROM (
      SELECT
        EXTRACT(YEAR FROM m.captured_at AT TIME ZONE 'UTC')::int as year,
        COUNT(*)::int as photo_count
      FROM metadata m
      WHERE m.captured_at IS NOT NULL
        AND m.collection_id = $1
      GROUP BY year
    ) y
    LEFT JOIN LATERAL (
      SELECT array_agg(id ORDER BY rn) as sample_ids
      FROM (
        SELECT p.id, ROW_NUMBER() OVER (
          ORDER BY (p.id + EXTRACT(DOY FROM current_date)::int * 127) % 997
        ) as rn
        FROM photo p
        JOIN metadata m ON p.id = m.photo_id
        WHERE EXTRACT(YEAR FROM m.captured_at AT TIME ZONE 'UTC') = y.year
          AND m.collection_id = $1
          AND p.collection_id = $1
        LIMIT 4
      ) ranked
    ) s ON true
    ORDER BY y.year DESC
  `;

  const result = await pool.query(query, [collectionId]);

  // Add backward compatibility field
  return result.rows.map((row) => ({
    ...row,
    sample_photo_id: row.sample_photo_ids?.[0] || null,
  }));
}

export async function getMonthsInYear(collectionId: number, year: number) {
  // Single query with LATERAL join for sample photos
  // Uses daily-varying deterministic selection like getYearsWithPhotos
  const query = `
    SELECT
      mo.month,
      mo.photo_count,
      COALESCE(s.sample_ids, ARRAY[]::int[]) as sample_photo_ids
    FROM (
      SELECT
        EXTRACT(MONTH FROM m.captured_at AT TIME ZONE 'UTC')::int as month,
        COUNT(*)::int as photo_count
      FROM metadata m
      WHERE EXTRACT(YEAR FROM m.captured_at AT TIME ZONE 'UTC') = $1
        AND m.captured_at IS NOT NULL
        AND m.collection_id = $2
      GROUP BY month
    ) mo
    LEFT JOIN LATERAL (
      SELECT array_agg(id ORDER BY rn) as sample_ids
      FROM (
        SELECT p.id, ROW_NUMBER() OVER (
          ORDER BY (p.id + EXTRACT(DOY FROM current_date)::int * 127) % 997
        ) as rn
        FROM photo p
        JOIN metadata m ON p.id = m.photo_id
        WHERE EXTRACT(YEAR FROM m.captured_at AT TIME ZONE 'UTC') = $1
          AND EXTRACT(MONTH FROM m.captured_at AT TIME ZONE 'UTC') = mo.month
          AND m.collection_id = $2
          AND p.collection_id = $2
        LIMIT 4
      ) ranked
    ) s ON true
    ORDER BY mo.month
  `;

  const result = await pool.query(query, [year, collectionId]);

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

  return result.rows.map((row) => ({
    ...row,
    month_name: monthNames[row.month],
  }));
}

export async function getPhotosByMonth(collectionId: number, year: number, month: number, limit = 100, offset = 0) {
  const query = `
    SELECT p.id, p.filename, p.normalized_path,
           m.captured_at, m.latitude, m.longitude,
           la.description, la.emotional_tone,
           la.objects, la.people_count
    FROM photo p
    JOIN metadata m ON p.id = m.photo_id
    LEFT JOIN llm_analysis la ON p.id = la.photo_id
    WHERE EXTRACT(YEAR FROM m.captured_at AT TIME ZONE 'UTC') = $1
      AND EXTRACT(MONTH FROM m.captured_at AT TIME ZONE 'UTC') = $2
      AND m.collection_id = $3
      AND p.collection_id = $3
    ORDER BY m.captured_at, p.filename
    LIMIT $4 OFFSET $5
  `;

  const result = await pool.query(query, [year, month, collectionId, limit, offset]);

  // Process photos to add computed fields
  const photos = result.rows.map((photo) => ({
    ...photo,
    filename_only: path.basename(photo.filename),
    short_description: photo.description
      ? photo.description.length > 50
        ? `${photo.description.substring(0, 47)}...`
        : photo.description
      : null,
  }));

  return photos;
}

export async function getPhotoCountByMonth(collectionId: number, year: number, month: number) {
  const query = `
    SELECT COUNT(*) as count
    FROM metadata m
    WHERE EXTRACT(YEAR FROM m.captured_at AT TIME ZONE 'UTC') = $1
      AND EXTRACT(MONTH FROM m.captured_at AT TIME ZONE 'UTC') = $2
      AND m.collection_id = $3
  `;

  const result = await pool.query(query, [year, month, collectionId]);
  return parseInt(result.rows[0].count, 10);
}

export async function getPhotoDetails(collectionId: number, photoId: number) {
  const query = `
    SELECT p.id, p.filename, p.normalized_path, 
           p.created_at as photo_created_at, p.updated_at as photo_updated_at,
           p.width, p.height, p.normalized_width, p.normalized_height,
           m.captured_at, m.latitude, m.longitude, 
           m.extra as metadata_extra,
           la.description, la.analysis, la.objects, la.people_count,
           la.location_description, la.emotional_tone,
           la.model_name, la.processed_at as analysis_processed_at,
           la.confidence_score
    FROM photo p
    LEFT JOIN metadata m ON p.id = m.photo_id
    LEFT JOIN llm_analysis la ON p.id = la.photo_id
    WHERE p.id = $1 AND p.collection_id = $2
  `;

  const result = await pool.query(query, [photoId, collectionId]);
  if (result.rows.length === 0) {
    return null;
  }

  const photo = result.rows[0];

  // Get person detections for this photo with match candidates
  // Join to both detection's person and cluster's person, prefer detection's person if set
  // Only include detections that have a face bounding box (exclude body-only detections)
  const facesQuery = `
    SELECT pd.id, pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.face_confidence as confidence, pd.person_id,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           pd.body_bbox_x, pd.body_bbox_y, pd.body_bbox_width, pd.body_bbox_height,
           COALESCE(
             NULLIF(TRIM(CONCAT(p.first_name, ' ', COALESCE(p.last_name, ''))), ''),
             NULLIF(TRIM(CONCAT(cp.first_name, ' ', COALESCE(cp.last_name, ''))), '')
           ) as person_name,
           pd.cluster_id, pd.cluster_status, pd.cluster_confidence
    FROM person_detection pd
    LEFT JOIN person p ON pd.person_id = p.id
    LEFT JOIN "cluster" c ON pd.cluster_id = c.id
    LEFT JOIN person cp ON c.person_id = cp.id
    WHERE pd.photo_id = $1
      AND pd.face_bbox_x IS NOT NULL
    ORDER BY pd.face_confidence DESC NULLS LAST
  `;

  const facesResult = await pool.query(facesQuery, [photoId]);

  // Get face match candidates for each detection
  for (const face of facesResult.rows) {
    const candidatesQuery = `
      SELECT fmc.cluster_id, fmc.similarity, fmc.status,
             c.person_id,
             TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
             c.face_count
      FROM face_match_candidate fmc
      LEFT JOIN "cluster" c ON fmc.cluster_id = c.id
      LEFT JOIN person per ON c.person_id = per.id
      WHERE fmc.detection_id = $1 AND fmc.status = 'pending'
      ORDER BY fmc.similarity DESC
      LIMIT 3
    `;

    const candidatesResult = await pool.query(candidatesQuery, [face.id]);
    face.match_candidates = candidatesResult.rows;
  }

  // Get detection tags (face tags) for each face
  for (const face of facesResult.rows) {
    const faceTagsQuery = `
      SELECT dt.confidence, dt.rank_in_category,
             pe.label, pe.display_name, pe.prompt_text,
             pc.name as category_name, pc.target
      FROM detection_tag dt
      JOIN prompt_embedding pe ON dt.prompt_id = pe.id
      JOIN prompt_category pc ON pe.category_id = pc.id
      WHERE dt.detection_id = $1
      ORDER BY pc.display_order, dt.confidence DESC
    `;
    const faceTagsResult = await pool.query(faceTagsQuery, [face.id]);
    face.tags = faceTagsResult.rows;
  }

  photo.faces = facesResult.rows;
  photo.face_count = facesResult.rows.length;

  // Get photo-level scene tags
  const photoTagsQuery = `
    SELECT pt.confidence, pt.rank_in_category,
           pe.label, pe.display_name, pe.prompt_text,
           pc.name as category_name, pc.target
    FROM photo_tag pt
    JOIN prompt_embedding pe ON pt.prompt_id = pe.id
    JOIN prompt_category pc ON pe.category_id = pc.id
    WHERE pt.photo_id = $1
    ORDER BY pc.display_order, pt.confidence DESC
  `;
  const photoTagsResult = await pool.query(photoTagsQuery, [photoId]);
  photo.scene_tags = photoTagsResult.rows;

  // Get scene analysis data (Apple Vision taxonomy)
  const sceneAnalysisQuery = `
    SELECT sa.taxonomy_labels, sa.taxonomy_confidences
    FROM scene_analysis sa
    WHERE sa.photo_id = $1
  `;
  const sceneAnalysisResult = await pool.query(sceneAnalysisQuery, [photoId]);
  if (sceneAnalysisResult.rows.length > 0 && sceneAnalysisResult.rows[0].taxonomy_labels) {
    const row = sceneAnalysisResult.rows[0];
    // Convert parallel arrays to array of objects for easier frontend consumption
    const topLabels = row.taxonomy_labels.map((label: string, i: number) => ({
      label,
      confidence: row.taxonomy_confidences[i] || 0,
    }));
    photo.scene_taxonomy = { top_labels: topLabels };
  }

  // Add computed fields
  photo.filename_only = path.basename(photo.filename);

  // Format analysis and metadata if present
  if (photo.analysis) {
    try {
      photo.analysis_formatted = JSON.stringify(photo.analysis, null, 2);
    } catch {
      photo.analysis_formatted = String(photo.analysis);
    }
  }

  if (photo.metadata_extra) {
    try {
      photo.metadata_formatted = JSON.stringify(photo.metadata_extra, null, 2);
    } catch {
      photo.metadata_formatted = String(photo.metadata_extra);
    }
  }

  // Add date fields
  if (photo.captured_at) {
    const date = new Date(photo.captured_at);
    photo.year = date.getFullYear();
    photo.month = date.getMonth() + 1;
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
    photo.month_name = monthNames[photo.month];
  }

  // Set image dimensions from normalized image (face boxes use normalized coordinates)
  photo.image_width = photo.normalized_width || null;
  photo.image_height = photo.normalized_height || null;

  return photo;
}

export async function getPhotoById(collectionId: number, photoId: number) {
  const query = `
    SELECT id, filename, normalized_path
    FROM photo
    WHERE id = $1 AND collection_id = $2
  `;

  const result = await pool.query(query, [photoId, collectionId]);
  return result.rows[0] || null;
}

export type AppUser = {
  id: number;
  username: string;
  password_hash: string;
  first_name: string;
  last_name: string;
  default_collection_id: number | null;
};

export async function getUserByUsername(username: string): Promise<AppUser | null> {
  const result = await pool.query("SELECT * FROM app_user WHERE username = $1", [username]);
  return (result.rows[0] as AppUser) || null;
}

export async function getUserById(userId: number): Promise<AppUser | null> {
  const result = await pool.query("SELECT * FROM app_user WHERE id = $1", [userId]);
  return (result.rows[0] as AppUser) || null;
}

export async function updateUserPasswordHash(userId: number, passwordHash: string): Promise<void> {
  await pool.query("UPDATE app_user SET password_hash = $1 WHERE id = $2", [passwordHash, userId]);
}

export async function getClusters(collectionId: number, limit = 50, offset = 0) {
  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           p.id as photo_id, p.normalized_path, p.filename,
           p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    LEFT JOIN person per ON c.person_id = per.id
    WHERE c.face_count > 0 AND (c.hidden = false OR c.hidden IS NULL)
      AND c.collection_id = $1
    ORDER BY c.face_count DESC, c.id
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
}

export async function getClustersCount(collectionId: number) {
  const query = `
    SELECT COUNT(*) as count
    FROM cluster
    WHERE face_count > 0 AND (hidden = false OR hidden IS NULL)
      AND collection_id = $1
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

/**
 * Get a unified list of people (with aggregated clusters) and unassigned clusters.
 * People are represented by their aggregated face count across all their clusters.
 * Unassigned clusters (no person_id) are shown individually.
 * Sorted by face count descending.
 */
export async function getClustersGroupedByPerson(collectionId: number, limit = 50, offset = 0) {
  const query = `
    WITH person_aggregates AS (
      -- Aggregate clusters by person
      SELECT
        'person' as item_type,
        per.id as id,
        per.id as person_id,
        SUM(c.face_count)::integer as face_count,
        COUNT(c.id)::integer as cluster_count,
        TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
        -- Get the largest cluster ID for drag-drop operations
        (SELECT c2.id FROM cluster c2
         WHERE c2.person_id = per.id
           AND c2.face_count > 0
           AND (c2.hidden = false OR c2.hidden IS NULL)
         ORDER BY c2.face_count DESC
         LIMIT 1) as primary_cluster_id,
        -- Get representative detection (person's or fallback to largest cluster's)
        COALESCE(
          per.representative_detection_id,
          (SELECT c2.representative_detection_id FROM cluster c2
           WHERE c2.person_id = per.id
             AND c2.representative_detection_id IS NOT NULL
             AND (c2.hidden = false OR c2.hidden IS NULL)
           ORDER BY c2.face_count DESC
           LIMIT 1)
        ) as representative_detection_id
      FROM person per
      INNER JOIN cluster c ON c.person_id = per.id
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $1
      GROUP BY per.id, per.first_name, per.last_name, per.representative_detection_id
    ),
    unassigned_clusters AS (
      -- Individual clusters without a person
      SELECT
        'cluster' as item_type,
        c.id as id,
        NULL::bigint as person_id,
        c.face_count::integer as face_count,
        1 as cluster_count,
        NULL as person_name,
        c.id as primary_cluster_id,
        c.representative_detection_id
      FROM cluster c
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $1
        AND c.person_id IS NULL
    ),
    combined AS (
      SELECT * FROM person_aggregates
      UNION ALL
      SELECT * FROM unassigned_clusters
    )
    SELECT
      combined.*,
      pd.face_bbox_x as bbox_x,
      pd.face_bbox_y as bbox_y,
      pd.face_bbox_width as bbox_width,
      pd.face_bbox_height as bbox_height,
      p.id as photo_id,
      p.normalized_width,
      p.normalized_height
    FROM combined
    LEFT JOIN person_detection pd ON combined.representative_detection_id = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    ORDER BY combined.face_count DESC, combined.id
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
}

/**
 * Get count of items for the grouped clusters view.
 * Counts distinct people + unassigned clusters.
 */
export async function getClustersGroupedCount(collectionId: number) {
  const query = `
    SELECT
      (
        -- Count distinct people with visible clusters
        SELECT COUNT(DISTINCT c.person_id)
        FROM cluster c
        WHERE c.face_count > 0
          AND (c.hidden = false OR c.hidden IS NULL)
          AND c.collection_id = $1
          AND c.person_id IS NOT NULL
      ) + (
        -- Count unassigned clusters
        SELECT COUNT(*)
        FROM cluster c
        WHERE c.face_count > 0
          AND (c.hidden = false OR c.hidden IS NULL)
          AND c.collection_id = $1
          AND c.person_id IS NULL
      ) as count
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

export async function getHiddenClusters(collectionId: number, limit = 50, offset = 0) {
  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           p.id as photo_id, p.normalized_path, p.filename,
           p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    LEFT JOIN person per ON c.person_id = per.id
    WHERE c.face_count > 0 AND c.hidden = true
      AND c.collection_id = $1
    ORDER BY c.face_count DESC, c.id
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
}

export async function getHiddenClustersCount(collectionId: number) {
  const query = `
    SELECT COUNT(*) as count
    FROM cluster
    WHERE face_count > 0 AND hidden = true
      AND collection_id = $1
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

export async function getNamedClusters(collectionId: number, limit = 50, offset = 0, sort: "photos" | "name" = "name") {
  const orderBy =
    sort === "photos"
      ? "c.face_count DESC, per.first_name, per.last_name, c.id"
      : "per.first_name, per.last_name, c.id";

  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           p.id as photo_id, p.normalized_path, p.filename,
           p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    INNER JOIN person per ON c.person_id = per.id
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    WHERE c.face_count > 0 AND (c.hidden = false OR c.hidden IS NULL)
      AND c.collection_id = $1
    ORDER BY ${orderBy}
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
}

export async function getNamedClustersCount(collectionId: number) {
  const query = `
    SELECT COUNT(*) as count
    FROM cluster c
    INNER JOIN person per ON c.person_id = per.id
    WHERE c.face_count > 0 AND (c.hidden = false OR c.hidden IS NULL)
      AND c.collection_id = $1
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

/**
 * Get people with aggregated cluster data.
 * Groups clusters by person_id and sums face counts.
 * Uses person.representative_detection_id if set, otherwise falls back to largest cluster's representative.
 */
export async function getPeople(collectionId: number, limit = 50, offset = 0, sort: "photos" | "name" = "name") {
  const orderBy =
    sort === "photos"
      ? "total_face_count DESC, per.first_name, per.last_name, per.id"
      : "per.first_name, per.last_name, per.id";

  const query = `
    WITH person_stats AS (
      SELECT
        c.person_id,
        SUM(c.face_count)::integer as total_face_count,
        COUNT(c.id)::integer as cluster_count,
        -- Fallback: get the representative from the largest cluster
        (SELECT c2.representative_detection_id
         FROM cluster c2
         WHERE c2.person_id = c.person_id
           AND c2.representative_detection_id IS NOT NULL
           AND (c2.hidden = false OR c2.hidden IS NULL)
         ORDER BY c2.face_count DESC
         LIMIT 1) as fallback_representative_detection_id
      FROM cluster c
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $1
        AND c.person_id IS NOT NULL
      GROUP BY c.person_id
    )
    SELECT
      per.id,
      TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
      ps.total_face_count,
      ps.cluster_count,
      -- Use person's representative if set, otherwise use fallback from largest cluster
      COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) as representative_detection_id,
      pd.face_bbox_x as bbox_x,
      pd.face_bbox_y as bbox_y,
      pd.face_bbox_width as bbox_width,
      pd.face_bbox_height as bbox_height,
      p.id as photo_id,
      p.normalized_path,
      p.filename,
      p.normalized_width,
      p.normalized_height
    FROM person per
    INNER JOIN person_stats ps ON per.id = ps.person_id
    LEFT JOIN person_detection pd ON COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    WHERE per.collection_id = $1
    ORDER BY ${orderBy}
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
}

/**
 * Get count of unique people (not clusters).
 */
export async function getPeopleCount(collectionId: number) {
  const query = `
    SELECT COUNT(DISTINCT c.person_id) as count
    FROM cluster c
    WHERE c.face_count > 0
      AND (c.hidden = false OR c.hidden IS NULL)
      AND c.collection_id = $1
      AND c.person_id IS NOT NULL
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

/**
 * Get a person by ID with aggregated cluster data.
 * Uses person.representative_detection_id if set, otherwise falls back to largest cluster's representative.
 */
export async function getPersonById(collectionId: number, personId: string) {
  const query = `
    WITH person_stats AS (
      SELECT
        c.person_id,
        SUM(c.face_count)::integer as total_face_count,
        COUNT(c.id)::integer as cluster_count,
        -- Fallback: get the representative from the largest cluster
        (SELECT c2.representative_detection_id
         FROM cluster c2
         WHERE c2.person_id = c.person_id
           AND c2.representative_detection_id IS NOT NULL
           AND (c2.hidden = false OR c2.hidden IS NULL)
         ORDER BY c2.face_count DESC
         LIMIT 1) as fallback_representative_detection_id
      FROM cluster c
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $2
        AND c.person_id = $1
      GROUP BY c.person_id
    )
    SELECT
      per.id,
      per.first_name,
      per.last_name,
      TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
      COALESCE(ps.total_face_count, 0)::integer as total_face_count,
      COALESCE(ps.cluster_count, 0)::integer as cluster_count,
      COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) as representative_detection_id,
      pd.face_bbox_x as bbox_x,
      pd.face_bbox_y as bbox_y,
      pd.face_bbox_width as bbox_width,
      pd.face_bbox_height as bbox_height,
      p.id as photo_id,
      p.normalized_width,
      p.normalized_height
    FROM person per
    LEFT JOIN person_stats ps ON per.id = ps.person_id
    LEFT JOIN person_detection pd ON COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    WHERE per.id = $1 AND per.collection_id = $2
  `;

  const result = await pool.query(query, [personId, collectionId]);
  return result.rows[0] || null;
}

/**
 * Get all clusters belonging to a person.
 */
export async function getClustersByPerson(collectionId: number, personId: string) {
  const query = `
    SELECT c.id, c.face_count::integer as face_count, c.representative_detection_id, c.hidden, c.verified,
           pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           p.id as photo_id, p.normalized_path, p.filename,
           p.normalized_width, p.normalized_height
    FROM cluster c
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    WHERE c.person_id = $1 AND c.collection_id = $2 AND c.face_count > 0
    ORDER BY c.face_count DESC, c.id
  `;

  const result = await pool.query(query, [personId, collectionId]);
  return result.rows;
}

/**
 * Rename a person directly.
 */
export async function setPersonName(collectionId: number, personId: string, firstName: string, lastName?: string) {
  const query = `
    UPDATE person
    SET first_name = $1, last_name = $2, updated_at = NOW()
    WHERE id = $3 AND collection_id = $4
    RETURNING id
  `;

  const result = await pool.query(query, [firstName, lastName || null, personId, collectionId]);
  return {
    success: result.rows.length > 0,
    message: result.rows.length > 0 ? "Person renamed" : "Person not found",
  };
}

/**
 * Hide or unhide all clusters belonging to a person.
 */
export async function setPersonHidden(collectionId: number, personId: string, hidden: boolean) {
  const query = `
    UPDATE cluster
    SET hidden = $1, updated_at = NOW()
    WHERE person_id = $2 AND collection_id = $3
    RETURNING id
  `;

  const result = await pool.query(query, [hidden, personId, collectionId]);
  const count = result.rowCount || 0;
  return {
    success: count > 0,
    message: hidden ? `Hidden ${count} cluster(s)` : `Unhidden ${count} cluster(s)`,
    count,
  };
}

/**
 * Unlink a cluster from its person (set person_id to NULL).
 */
export async function unlinkClusterFromPerson(collectionId: number, clusterId: string) {
  const query = `
    UPDATE cluster
    SET person_id = NULL, updated_at = NOW()
    WHERE id = $1 AND collection_id = $2
    RETURNING id
  `;

  const result = await pool.query(query, [clusterId, collectionId]);
  return {
    success: result.rows.length > 0,
    message: result.rows.length > 0 ? "Cluster unlinked from person" : "Cluster not found",
  };
}

/**
 * Set a person's representative detection from a cluster.
 * Uses the cluster's representative_detection_id.
 */
export async function setPersonRepresentative(collectionId: number, personId: string, clusterId: string) {
  const client = await pool.connect();
  try {
    // Get the cluster's representative detection and verify it belongs to this person
    const clusterQuery = `
      SELECT c.representative_detection_id, c.person_id
      FROM cluster c
      WHERE c.id = $1 AND c.collection_id = $2
    `;
    const clusterResult = await client.query(clusterQuery, [clusterId, collectionId]);

    if (clusterResult.rows.length === 0) {
      return { success: false, message: "Cluster not found" };
    }

    const cluster = clusterResult.rows[0];
    if (cluster.person_id?.toString() !== personId) {
      return { success: false, message: "Cluster does not belong to this person" };
    }

    if (!cluster.representative_detection_id) {
      return { success: false, message: "Cluster has no representative detection" };
    }

    // Update the person's representative
    const updateQuery = `
      UPDATE person
      SET representative_detection_id = $1, updated_at = NOW()
      WHERE id = $2 AND collection_id = $3
      RETURNING id
    `;
    const updateResult = await client.query(updateQuery, [cluster.representative_detection_id, personId, collectionId]);

    if (updateResult.rows.length === 0) {
      return { success: false, message: "Person not found" };
    }

    return { success: true, message: "Person representative updated" };
  } finally {
    client.release();
  }
}

export async function setClusterHidden(collectionId: number, clusterId: string, hidden: boolean) {
  const query = `
    UPDATE cluster
    SET hidden = $1, updated_at = NOW()
    WHERE id = $2 AND collection_id = $3
    RETURNING id
  `;

  const result = await pool.query(query, [hidden, clusterId, collectionId]);
  return {
    success: result.rows.length > 0,
    message: hidden ? "Cluster hidden" : "Cluster unhidden",
  };
}

export async function setClusterPersonName(
  collectionId: number,
  clusterId: string,
  firstName: string,
  lastName?: string,
) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Check if cluster already has a person_id
    const clusterResult = await client.query("SELECT person_id FROM cluster WHERE id = $1 AND collection_id = $2", [
      clusterId,
      collectionId,
    ]);
    if (clusterResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Cluster not found" };
    }

    const existingPersonId = clusterResult.rows[0].person_id;

    if (existingPersonId) {
      // Update existing person's name
      await client.query(
        "UPDATE person SET first_name = $1, last_name = $2, updated_at = NOW() WHERE id = $3 AND collection_id = $4",
        [firstName, lastName || null, existingPersonId, collectionId],
      );
    } else {
      // Create new person and link to cluster
      const personResult = await client.query(
        "INSERT INTO person (collection_id, first_name, last_name) VALUES ($1, $2, $3) RETURNING id",
        [collectionId, firstName, lastName || null],
      );
      const newPersonId = personResult.rows[0].id;
      await client.query("UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3", [
        newPersonId,
        clusterId,
        collectionId,
      ]);
    }

    await client.query("COMMIT");
    return { success: true, message: "Name updated" };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to set cluster person name:", error);
    return { success: false, message: "Failed to update name" };
  } finally {
    client.release();
  }
}

export async function deleteCluster(collectionId: number, clusterId: string) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Move all detections back to unassigned pool
    const updateDetectionsQuery = `
      UPDATE person_detection
      SET cluster_id = NULL,
          cluster_status = 'unassigned',
          cluster_confidence = 0,
          unassigned_since = NOW()
      WHERE cluster_id = $1 AND collection_id = $2
      RETURNING id
    `;
    const detectionsResult = await client.query(updateDetectionsQuery, [clusterId, collectionId]);
    const detectionsRemoved = detectionsResult.rowCount || 0;

    // Delete the cluster
    const deleteQuery = `
      DELETE FROM cluster
      WHERE id = $1 AND collection_id = $2
      RETURNING id
    `;
    const deleteResult = await client.query(deleteQuery, [clusterId, collectionId]);

    await client.query("COMMIT");

    if (deleteResult.rows.length > 0) {
      return {
        success: true,
        message: `Cluster deleted, ${detectionsRemoved} detection${detectionsRemoved !== 1 ? "s" : ""} moved to unassigned`,
        detectionsRemoved,
      };
    }
    return { success: false, message: "Cluster not found", detectionsRemoved: 0 };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to delete cluster:", error);
    return { success: false, message: "Failed to delete cluster", detectionsRemoved: 0 };
  } finally {
    client.release();
  }
}

export async function getClusterDetails(collectionId: number, clusterId: string) {
  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           c.hidden,
           per.first_name, per.last_name,
           per.gender as person_gender, per.gender_confidence as person_gender_confidence,
           per.estimated_birth_year, per.birth_year_stddev,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
           rep.face_bbox_x as rep_bbox_x, rep.face_bbox_y as rep_bbox_y,
           rep.face_bbox_width as rep_bbox_width, rep.face_bbox_height as rep_bbox_height,
           rep.photo_id as rep_photo_id,
           p.normalized_width as rep_normalized_width, p.normalized_height as rep_normalized_height
    FROM cluster c
    LEFT JOIN person per ON c.person_id = per.id
    LEFT JOIN person_detection rep ON c.representative_detection_id = rep.id
    LEFT JOIN photo p ON rep.photo_id = p.id
    WHERE c.id = $1 AND c.collection_id = $2
  `;

  const result = await pool.query(query, [clusterId, collectionId]);
  return result.rows[0] || null;
}

export async function getClusterFaces(collectionId: number, clusterId: string, limit = 24, offset = 0) {
  const query = `
    SELECT pd.id, pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.cluster_confidence, pd.photo_id,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           p.normalized_path, p.filename, p.normalized_width, p.normalized_height
    FROM person_detection pd
    JOIN photo p ON pd.photo_id = p.id
    WHERE pd.cluster_id = $1 AND pd.collection_id = $2
    ORDER BY (pd.id = (SELECT representative_detection_id FROM cluster WHERE id = $1 AND collection_id = $2)) DESC,
             pd.cluster_confidence DESC, pd.id
    LIMIT $3 OFFSET $4
  `;

  const result = await pool.query(query, [clusterId, collectionId, limit, offset]);
  return result.rows;
}

export async function getClusterFacesCount(collectionId: number, clusterId: string) {
  const query = `
    SELECT COUNT(*) as count
    FROM person_detection
    WHERE cluster_id = $1 AND collection_id = $2
  `;

  const result = await pool.query(query, [clusterId, collectionId]);
  return parseInt(result.rows[0].count, 10);
}

// Constraint management functions

export async function addCannotLink(collectionId: number, detectionId1: number, detectionId2: number) {
  // Canonical ordering to prevent duplicates
  const [id1, id2] = detectionId1 < detectionId2 ? [detectionId1, detectionId2] : [detectionId2, detectionId1];

  const query = `
    INSERT INTO cannot_link (detection_id_1, detection_id_2, collection_id, created_by)
    VALUES ($1, $2, $3, 'web')
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id
  `;

  const result = await pool.query(query, [id1, id2, collectionId]);
  return result.rows[0]?.id || null;
}

export async function getCannotLinksForCluster(collectionId: number, clusterId: string) {
  // Get all cannot-link pairs where both detections are in this cluster
  const query = `
    SELECT cl.id, cl.detection_id_1, cl.detection_id_2, cl.created_at
    FROM cannot_link cl
    JOIN person_detection pd1 ON cl.detection_id_1 = pd1.id
    JOIN person_detection pd2 ON cl.detection_id_2 = pd2.id
    WHERE pd1.cluster_id = $1 AND pd2.cluster_id = $1
      AND cl.collection_id = $2
      AND pd1.collection_id = $2
      AND pd2.collection_id = $2
    ORDER BY cl.created_at DESC
  `;

  const result = await pool.query(query, [clusterId, collectionId]);
  return result.rows;
}

export async function removeCannotLink(collectionId: number, cannotLinkId: number) {
  const query = `
    DELETE FROM cannot_link
    WHERE id = $1 AND collection_id = $2
    RETURNING id
  `;

  const result = await pool.query(query, [cannotLinkId, collectionId]);
  return result.rows[0]?.id || null;
}

export async function setClusterRepresentative(collectionId: number, clusterId: string, detectionId: number) {
  // Verify detection belongs to this cluster
  const verifyQuery = `
    SELECT cluster_id FROM person_detection WHERE id = $1 AND collection_id = $2
  `;
  const verifyResult = await pool.query(verifyQuery, [detectionId, collectionId]);
  if (verifyResult.rows.length === 0 || verifyResult.rows[0].cluster_id?.toString() !== clusterId) {
    return { success: false, message: "Detection does not belong to this cluster" };
  }

  // Update the representative detection
  const updateQuery = `
    UPDATE cluster
    SET representative_detection_id = $1,
        updated_at = NOW()
    WHERE id = $2 AND collection_id = $3
    RETURNING id
  `;

  const result = await pool.query(updateQuery, [detectionId, clusterId, collectionId]);
  return {
    success: result.rows.length > 0,
    message: result.rows.length > 0 ? "Representative photo updated" : "Failed to update",
  };
}

export async function dissociateFacesFromCluster(
  collectionId: number,
  clusterId: string,
  detectionIds: number[],
  similarityThreshold = 0.85,
) {
  if (detectionIds.length === 0) {
    return { success: false, message: "No detections selected", removedCount: 0 };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const clusterCheck = await client.query("SELECT id FROM cluster WHERE id = $1 AND collection_id = $2", [
      clusterId,
      collectionId,
    ]);
    if (clusterCheck.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Cluster not found", removedCount: 0 };
    }

    // Step 1: Find all detections in cluster similar to selected detections above threshold
    // Using cosine distance: 1 - cosine_similarity, so threshold becomes 1 - similarityThreshold
    const distanceThreshold = 1 - similarityThreshold;

    const similarDetectionsQuery = `
      WITH selected_embeddings AS (
        SELECT fe.person_detection_id, fe.embedding
        FROM face_embedding fe
        JOIN person_detection pd ON fe.person_detection_id = pd.id
        WHERE fe.person_detection_id = ANY($1) AND pd.cluster_id = $2
      ),
      cluster_detections AS (
        SELECT fe.person_detection_id, fe.embedding
        FROM face_embedding fe
        JOIN person_detection pd ON fe.person_detection_id = pd.id
        WHERE pd.cluster_id = $2
      )
      SELECT DISTINCT cd.person_detection_id
      FROM cluster_detections cd
      WHERE cd.person_detection_id = ANY($1)
         OR EXISTS (
           SELECT 1 FROM selected_embeddings se
           WHERE (cd.embedding <=> se.embedding) < $3
         )
    `;

    const similarResult = await client.query(similarDetectionsQuery, [detectionIds, clusterId, distanceThreshold]);

    // Parse as integers since PostgreSQL bigint can come back as strings
    const allDetectionsToRemove = similarResult.rows.map((r) => parseInt(r.person_detection_id, 10));

    if (allDetectionsToRemove.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "No detections found to remove", removedCount: 0 };
    }

    // Step 2: Get a sample detection that remains in the cluster (for cannot-link)
    const remainingDetectionQuery = `
      SELECT id FROM person_detection
      WHERE cluster_id = $1 AND id != ALL($2)
      LIMIT 1
    `;
    const remainingResult = await client.query(remainingDetectionQuery, [clusterId, allDetectionsToRemove]);

    // Step 3: Remove detections from cluster and add to unassigned pool
    // Using 'unassigned' status allows them to join other clusters while
    // respecting cannot-link constraints that prevent rejoining original cluster
    const removeQuery = `
      UPDATE person_detection
      SET cluster_id = NULL,
          cluster_status = 'unassigned',
          cluster_confidence = 0,
          unassigned_since = NOW()
      WHERE id = ANY($1) AND cluster_id = $2 AND collection_id = $3
      RETURNING id
    `;
    const removeResult = await client.query(removeQuery, [allDetectionsToRemove, clusterId, collectionId]);
    const removedCount = removeResult.rowCount || 0;

    // Step 4: Update cluster face count
    await client.query(
      `UPDATE cluster SET face_count = GREATEST(0, face_count - $1), updated_at = NOW() WHERE id = $2 AND collection_id = $3`,
      [removedCount, clusterId, collectionId],
    );

    // Step 5: Create cannot-link constraints between removed detections and a remaining cluster detection
    // This prevents the removed detections from rejoining this cluster
    if (remainingResult.rows.length > 0) {
      const remainingDetectionId = parseInt(remainingResult.rows[0].id, 10);

      for (const removedDetectionId of allDetectionsToRemove) {
        // Ensure canonical ordering: detection_id_1 < detection_id_2
        const [id1, id2] =
          removedDetectionId < remainingDetectionId
            ? [removedDetectionId, remainingDetectionId]
            : [remainingDetectionId, removedDetectionId];

        await client.query(
          `INSERT INTO cannot_link (detection_id_1, detection_id_2, collection_id, created_by)
           VALUES ($1, $2, $3, 'web')
           ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING`,
          [id1, id2, collectionId],
        );
      }
    }

    await client.query("COMMIT");

    return {
      success: true,
      message: `Removed ${removedCount} detection${removedCount !== 1 ? "s" : ""} from cluster`,
      removedCount,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to dissociate detections:", error);
    return { success: false, message: "Failed to remove detections", removedCount: 0 };
  } finally {
    client.release();
  }
}

export async function searchClusters(collectionId: number, query: string, excludeClusterId?: string, limit = 20) {
  // Search by cluster ID or person name
  const searchQuery = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           p.id as photo_id, p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
    LEFT JOIN photo p ON pd.photo_id = p.id
    LEFT JOIN person per ON c.person_id = per.id
    WHERE c.face_count > 0
      AND (c.hidden = false OR c.hidden IS NULL)
      AND c.collection_id = $4
      AND ($1::text IS NULL OR c.id::text = $1
           OR per.first_name ILIKE '%' || $1 || '%'
           OR per.last_name ILIKE '%' || $1 || '%'
           OR CONCAT(per.first_name, ' ', COALESCE(per.last_name, '')) ILIKE '%' || $1 || '%')
      AND ($2::text IS NULL OR c.id::text != $2)
    ORDER BY
      CASE WHEN c.id::text = $1 THEN 0 ELSE 1 END,
      CASE WHEN per.first_name ILIKE $1 || '%' OR per.last_name ILIKE $1 || '%' THEN 0 ELSE 1 END,
      c.face_count DESC
    LIMIT $3
  `;

  const result = await pool.query(searchQuery, [query || null, excludeClusterId || null, limit, collectionId]);
  return result.rows;
}

/**
 * Preview what will happen when linking two clusters.
 * Returns info about both clusters' person associations.
 */
export async function previewClusterLink(collectionId: number, sourceClusterId: string, targetClusterId: string) {
  const query = `
    SELECT
      c.id as cluster_id,
      c.person_id,
      TRIM(CONCAT(p.first_name, ' ', COALESCE(p.last_name, ''))) as person_name,
      p.first_name,
      p.last_name,
      (SELECT COUNT(*) FROM cluster c2 WHERE c2.person_id = c.person_id) as person_cluster_count
    FROM cluster c
    LEFT JOIN person p ON c.person_id = p.id
    WHERE c.id = ANY($1) AND c.collection_id = $2
  `;

  const result = await pool.query(query, [[sourceClusterId, targetClusterId], collectionId]);

  const sourceInfo = result.rows.find((r) => r.cluster_id.toString() === sourceClusterId);
  const targetInfo = result.rows.find((r) => r.cluster_id.toString() === targetClusterId);

  if (!sourceInfo || !targetInfo) {
    return { found: false };
  }

  // Check if both have different person records
  const willMergePersons =
    sourceInfo.person_id && targetInfo.person_id && sourceInfo.person_id !== targetInfo.person_id;

  return {
    found: true,
    source: {
      clusterId: sourceClusterId,
      personId: sourceInfo.person_id,
      personName: sourceInfo.person_name || null,
      personClusterCount: parseInt(sourceInfo.person_cluster_count) || 0,
    },
    target: {
      clusterId: targetClusterId,
      personId: targetInfo.person_id,
      personName: targetInfo.person_name || null,
      personClusterCount: parseInt(targetInfo.person_cluster_count) || 0,
    },
    willMergePersons,
    // If merging persons, source person will be deleted and all their clusters moved to target person
    sourcePersonWillBeDeleted: willMergePersons,
  };
}

/**
 * Link two clusters to the same person.
 *
 * This does NOT merge clusters (move faces, delete cluster).
 * Instead, it assigns both clusters to the same person_id,
 * creating a new Person if neither cluster has one.
 *
 * If both clusters belong to different persons, ALL clusters from the
 * source person are moved to the target person, and the source person
 * record is deleted.
 *
 * This preserves cluster integrity while establishing identity linkage.
 */
export async function linkClustersToSamePerson(collectionId: number, sourceClusterId: string, targetClusterId: string) {
  if (sourceClusterId === targetClusterId) {
    return { success: false, message: "Cannot link a cluster with itself" };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Get source cluster info with person details
    const sourceQuery = `
      SELECT c.id, c.face_count, c.person_id, c.representative_detection_id,
             TRIM(CONCAT(p.first_name, ' ', COALESCE(p.last_name, ''))) as person_name
      FROM cluster c
      LEFT JOIN person p ON c.person_id = p.id
      WHERE c.id = $1 AND c.collection_id = $2
    `;
    const sourceResult = await client.query(sourceQuery, [sourceClusterId, collectionId]);
    if (sourceResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Source cluster not found" };
    }
    const sourceCluster = sourceResult.rows[0];

    // Get target cluster info with person details
    const targetQuery = `
      SELECT c.id, c.face_count, c.person_id, c.representative_detection_id,
             TRIM(CONCAT(p.first_name, ' ', COALESCE(p.last_name, ''))) as person_name
      FROM cluster c
      LEFT JOIN person p ON c.person_id = p.id
      WHERE c.id = $1 AND c.collection_id = $2
    `;
    const targetResult = await client.query(targetQuery, [targetClusterId, collectionId]);
    if (targetResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Target cluster not found" };
    }
    const targetCluster = targetResult.rows[0];

    let personId: number;
    let mergedPersons = false;
    let deletedPersonId: number | null = null;
    let deletedPersonName: string | null = null;

    // Check if both clusters have different person records
    if (sourceCluster.person_id && targetCluster.person_id && sourceCluster.person_id !== targetCluster.person_id) {
      // Both have different persons - merge source person into target person
      personId = targetCluster.person_id;
      deletedPersonId = sourceCluster.person_id;
      deletedPersonName = sourceCluster.person_name;
      mergedPersons = true;

      // Move ALL clusters from source person to target person
      await client.query(
        `UPDATE cluster SET person_id = $1, updated_at = NOW()
         WHERE person_id = $2 AND collection_id = $3`,
        [personId, sourceCluster.person_id, collectionId],
      );

      // Delete the now-orphaned source person record
      await client.query(`DELETE FROM person WHERE id = $1 AND collection_id = $2`, [
        sourceCluster.person_id,
        collectionId,
      ]);
    } else if (targetCluster.person_id) {
      // Target already has a person - use that
      personId = targetCluster.person_id;

      // Update source cluster to use target's person
      await client.query(`UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`, [
        personId,
        sourceClusterId,
        collectionId,
      ]);
    } else if (sourceCluster.person_id) {
      // Source has a person - use that
      personId = sourceCluster.person_id;

      // Update target cluster to use source's person
      await client.query(`UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`, [
        personId,
        targetClusterId,
        collectionId,
      ]);
    } else {
      // Neither has a person - create a new one with placeholder name
      // Use the target cluster's representative as the person's representative (prefer target, fallback to source)
      const representativeDetectionId =
        targetCluster.representative_detection_id || sourceCluster.representative_detection_id || null;

      const createPersonResult = await client.query(
        `INSERT INTO person (collection_id, first_name, representative_detection_id, created_at)
         VALUES ($1, 'Unknown', $2, NOW()) RETURNING id`,
        [collectionId, representativeDetectionId],
      );
      personId = createPersonResult.rows[0].id;

      // Update both clusters to use the new person
      await client.query(`UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`, [
        personId,
        sourceClusterId,
        collectionId,
      ]);
      await client.query(`UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`, [
        personId,
        targetClusterId,
        collectionId,
      ]);
    }

    await client.query("COMMIT");

    let message = `Linked clusters as same person`;
    if (mergedPersons) {
      message = `Merged "${deletedPersonName}" into "${targetCluster.person_name}"`;
    }

    return {
      success: true,
      message,
      personId,
      sourceClusterId,
      targetClusterId,
      mergedPersons,
      deletedPersonId,
      deletedPersonName,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to link clusters:", error);
    return { success: false, message: "Failed to link clusters" };
  } finally {
    client.release();
  }
}

export async function addFacesToCluster(collectionId: number, clusterId: string, detectionIds: number[]) {
  if (detectionIds.length === 0) {
    return { success: false, message: "No detections selected", addedCount: 0 };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Verify cluster exists
    const clusterQuery = `SELECT id, face_count FROM cluster WHERE id = $1 AND collection_id = $2`;
    const clusterResult = await client.query(clusterQuery, [clusterId, collectionId]);
    if (clusterResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Cluster not found", addedCount: 0 };
    }

    // Add detections to the cluster (only unclustered detections)
    const addDetectionsQuery = `
      UPDATE person_detection
      SET cluster_id = $1,
          cluster_status = 'manual',
          cluster_confidence = 1.0,
          unassigned_since = NULL
      WHERE id = ANY($2) AND cluster_id IS NULL AND collection_id = $3
      RETURNING id
    `;
    const addResult = await client.query(addDetectionsQuery, [clusterId, detectionIds, collectionId]);
    const addedCount = addResult.rowCount || 0;

    if (addedCount === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "No unclustered detections to add", addedCount: 0 };
    }

    // Update cluster face count
    await client.query(
      `UPDATE cluster SET face_count = face_count + $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`,
      [addedCount, clusterId, collectionId],
    );

    // Recompute cluster centroid
    const centroidQuery = `
      UPDATE cluster
      SET centroid = (
        SELECT AVG(fe.embedding)::vector(512)
        FROM person_detection pd
        JOIN face_embedding fe ON pd.id = fe.person_detection_id
        WHERE pd.cluster_id = $1
      )
      WHERE id = $1 AND collection_id = $2
    `;
    await client.query(centroidQuery, [clusterId, collectionId]);

    await client.query("COMMIT");

    return {
      success: true,
      message: `Added ${addedCount} detection${addedCount !== 1 ? "s" : ""} to cluster`,
      addedCount,
      clusterId,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to add detections to cluster:", error);
    return { success: false, message: "Failed to add detections to cluster", addedCount: 0 };
  } finally {
    client.release();
  }
}

export async function getFaceDetails(collectionId: number, detectionId: number) {
  const query = `
    SELECT pd.id, pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.face_confidence as confidence, pd.cluster_id, pd.cluster_status,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           pd.photo_id, p.normalized_path, p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM person_detection pd
    JOIN photo p ON pd.photo_id = p.id
    LEFT JOIN cluster c ON pd.cluster_id = c.id
    LEFT JOIN person per ON c.person_id = per.id
    WHERE pd.id = $1 AND pd.collection_id = $2
  `;

  const result = await pool.query(query, [detectionId, collectionId]);
  return result.rows[0] || null;
}

export async function createClusterFromFaces(collectionId: number, detectionIds: number[]) {
  if (detectionIds.length === 0) {
    return { success: false, message: "No detections selected", clusterId: null };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Step 1: Create a new cluster
    const createClusterQuery = `
      INSERT INTO cluster (collection_id, face_count, created_at, updated_at)
      VALUES ($1, $2, NOW(), NOW())
      RETURNING id
    `;
    const clusterResult = await client.query(createClusterQuery, [collectionId, detectionIds.length]);
    const clusterId = clusterResult.rows[0].id;

    // Step 2: Assign all detections to the new cluster
    const assignDetectionsQuery = `
      UPDATE person_detection
      SET cluster_id = $1,
          cluster_status = 'manual',
          cluster_confidence = 1.0
      WHERE id = ANY($2) AND cluster_id IS NULL AND collection_id = $3
      RETURNING id
    `;
    const assignResult = await client.query(assignDetectionsQuery, [clusterId, detectionIds, collectionId]);
    const assignedCount = assignResult.rowCount || 0;

    // Step 3: Update the cluster face count to match actual assigned
    await client.query(`UPDATE cluster SET face_count = $1 WHERE id = $2 AND collection_id = $3`, [
      assignedCount,
      clusterId,
      collectionId,
    ]);

    // Step 4: Set the first detection as the representative
    if (assignedCount > 0) {
      const firstDetectionId = detectionIds[0];
      await client.query(
        `UPDATE cluster SET representative_detection_id = $1, medoid_detection_id = $1 WHERE id = $2 AND collection_id = $3`,
        [firstDetectionId, clusterId, collectionId],
      );
    }

    // Step 5: Compute and set centroid from detection embeddings
    const centroidQuery = `
      WITH cluster_embeddings AS (
        SELECT embedding
        FROM face_embedding
        WHERE person_detection_id = ANY($1)
      )
      UPDATE cluster
      SET centroid = (
        SELECT AVG(embedding)::vector(512)
        FROM cluster_embeddings
      )
      WHERE id = $2 AND collection_id = $3
    `;
    await client.query(centroidQuery, [detectionIds, clusterId, collectionId]);

    await client.query("COMMIT");

    return {
      success: true,
      message: `Created cluster ${clusterId} with ${assignedCount} detection${assignedCount !== 1 ? "s" : ""}`,
      clusterId,
      assignedCount,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to create cluster from detections:", error);
    return { success: false, message: "Failed to create cluster", clusterId: null };
  } finally {
    client.release();
  }
}

export async function getSimilarFaces(
  collectionId: number,
  detectionId: number,
  limit = 12,
  similarityThreshold = 0.7,
) {
  // Convert similarity threshold to distance threshold (cosine distance = 1 - similarity)
  const distanceThreshold = 1 - similarityThreshold;

  // Overfetch to ensure enough results after threshold filtering.
  // Vector indexes (IVFFlat/HNSW) can only optimize ORDER BY + LIMIT, not WHERE distance < X.
  // By moving the distance filter to the outer query, we fetch candidates using the index first.
  const overfetchLimit = limit * 5;

  const query = `
    WITH target_embedding AS (
      SELECT embedding FROM face_embedding WHERE person_detection_id = $1
    ),
    target_detection AS (
      SELECT photo_id FROM person_detection WHERE id = $1 AND collection_id = $5
    ),
    candidates AS (
      SELECT pd.id, pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
             pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
             pd.face_confidence as confidence, pd.cluster_id, pd.cluster_status, pd.cluster_confidence,
             pd.age_estimate, pd.gender, pd.gender_confidence,
             p.id as photo_id, p.normalized_path, p.normalized_width, p.normalized_height,
             c.face_count as cluster_face_count,
             TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
             fe.embedding <=> te.embedding as distance
      FROM face_embedding fe
      CROSS JOIN target_embedding te
      JOIN person_detection pd ON fe.person_detection_id = pd.id
      JOIN photo p ON pd.photo_id = p.id
      LEFT JOIN cluster c ON pd.cluster_id = c.id
      LEFT JOIN person per ON c.person_id = per.id
      WHERE fe.person_detection_id != $1
        AND pd.photo_id != (SELECT photo_id FROM target_detection)
        AND pd.collection_id = $5
        AND p.collection_id = $5
      ORDER BY fe.embedding <=> te.embedding ASC
      LIMIT $3
    )
    SELECT id, bbox_x, bbox_y, bbox_width, bbox_height, confidence,
           cluster_id, cluster_status, cluster_confidence,
           age_estimate, gender, gender_confidence,
           photo_id, normalized_path, normalized_width, normalized_height,
           cluster_face_count, person_name,
           1 - distance as similarity
    FROM candidates
    WHERE distance < $2
    LIMIT $4
  `;

  const result = await pool.query(query, [detectionId, distanceThreshold, overfetchLimit, limit, collectionId]);
  return result.rows;
}

export async function dissociateFaceWithConfidenceCutoff(collectionId: number, clusterId: string, detectionId: number) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const clusterCheck = await client.query("SELECT id FROM cluster WHERE id = $1 AND collection_id = $2", [
      clusterId,
      collectionId,
    ]);
    if (clusterCheck.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Cluster not found", constrainedCount: 0, cutoffCount: 0 };
    }

    // Step 1: Get the confidence of the selected detection
    const confidenceQuery = `
      SELECT cluster_confidence FROM person_detection WHERE id = $1 AND cluster_id = $2 AND collection_id = $3
    `;
    const confidenceResult = await client.query(confidenceQuery, [detectionId, clusterId, collectionId]);

    if (confidenceResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Detection not found in cluster", constrainedCount: 0, cutoffCount: 0 };
    }

    const cutoffConfidence = parseFloat(confidenceResult.rows[0].cluster_confidence);

    // Step 2: Find all detections with lower confidence (these will be unconstrained)
    const lowerConfidenceQuery = `
      SELECT id FROM person_detection
      WHERE cluster_id = $1 AND cluster_confidence < $2 AND id != $3 AND collection_id = $4
    `;
    const lowerResult = await client.query(lowerConfidenceQuery, [
      clusterId,
      cutoffConfidence,
      detectionId,
      collectionId,
    ]);
    const lowerConfidenceDetectionIds = lowerResult.rows.map((r) => parseInt(r.id, 10));

    // Step 3: Get a remaining detection for the cannot-link constraint
    const allDetectionsToRemove = [detectionId, ...lowerConfidenceDetectionIds];
    const remainingDetectionQuery = `
      SELECT id FROM person_detection
      WHERE cluster_id = $1 AND id != ALL($2) AND collection_id = $3
      LIMIT 1
    `;
    const remainingResult = await client.query(remainingDetectionQuery, [
      clusterId,
      allDetectionsToRemove,
      collectionId,
    ]);

    // Step 4: Remove the selected detection (constrained - cannot rejoin)
    await client.query(
      `UPDATE person_detection
       SET cluster_id = NULL,
           cluster_status = 'unassigned',
           cluster_confidence = 0,
           unassigned_since = NOW()
       WHERE id = $1 AND collection_id = $2`,
      [detectionId, collectionId],
    );

    // Step 5: Create cannot-link for the selected detection only
    if (remainingResult.rows.length > 0) {
      const remainingDetectionId = parseInt(remainingResult.rows[0].id, 10);
      const [id1, id2] =
        detectionId < remainingDetectionId ? [detectionId, remainingDetectionId] : [remainingDetectionId, detectionId];

      await client.query(
        `INSERT INTO cannot_link (detection_id_1, detection_id_2, collection_id, created_by)
         VALUES ($1, $2, $3, 'web')
         ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING`,
        [id1, id2, collectionId],
      );
    }

    // Step 6: Remove lower confidence detections (unconstrained - can rejoin other clusters)
    let cutoffCount = 0;
    if (lowerConfidenceDetectionIds.length > 0) {
      const cutoffResult = await client.query(
        `UPDATE person_detection
         SET cluster_id = NULL,
             cluster_status = 'unassigned',
             cluster_confidence = 0,
             unassigned_since = NOW()
         WHERE id = ANY($1) AND collection_id = $2
         RETURNING id`,
        [lowerConfidenceDetectionIds, collectionId],
      );
      cutoffCount = cutoffResult.rowCount || 0;
    }

    // Step 7: Update cluster face count
    const totalRemoved = 1 + cutoffCount;
    await client.query(
      `UPDATE cluster SET face_count = GREATEST(0, face_count - $1), updated_at = NOW() WHERE id = $2 AND collection_id = $3`,
      [totalRemoved, clusterId, collectionId],
    );

    await client.query("COMMIT");

    return {
      success: true,
      message: `Removed 1 detection (constrained) and ${cutoffCount} lower-confidence detection${cutoffCount !== 1 ? "s" : ""}`,
      constrainedCount: 1,
      cutoffCount,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to dissociate detection with cutoff:", error);
    return { success: false, message: "Failed to remove detections", constrainedCount: 0, cutoffCount: 0 };
  } finally {
    client.release();
  }
}

export async function removeFaceFromClusterWithConstraint(collectionId: number, detectionId: number) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Get the detection's current cluster
    const detectionQuery = `SELECT cluster_id FROM person_detection WHERE id = $1 AND collection_id = $2`;
    const detectionResult = await client.query(detectionQuery, [detectionId, collectionId]);

    if (detectionResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Detection not found" };
    }

    const clusterId = detectionResult.rows[0].cluster_id;
    if (!clusterId) {
      await client.query("ROLLBACK");
      return { success: false, message: "Detection is not assigned to a cluster" };
    }

    // Get a remaining detection in the cluster for the cannot-link constraint
    const remainingDetectionQuery = `
      SELECT id FROM person_detection
      WHERE cluster_id = $1 AND id != $2 AND collection_id = $3
      LIMIT 1
    `;
    const remainingResult = await client.query(remainingDetectionQuery, [clusterId, detectionId, collectionId]);

    // Remove detection from cluster
    await client.query(
      `UPDATE person_detection
       SET cluster_id = NULL,
           cluster_status = 'unassigned',
           cluster_confidence = 0,
           unassigned_since = NOW()
       WHERE id = $1 AND collection_id = $2`,
      [detectionId, collectionId],
    );

    // Update cluster face count
    await client.query(
      `UPDATE cluster SET face_count = GREATEST(0, face_count - 1), updated_at = NOW() WHERE id = $1 AND collection_id = $2`,
      [clusterId, collectionId],
    );

    // Create cannot-link constraint if there's a remaining detection
    if (remainingResult.rows.length > 0) {
      const remainingDetectionId = parseInt(remainingResult.rows[0].id, 10);
      const [id1, id2] =
        detectionId < remainingDetectionId ? [detectionId, remainingDetectionId] : [remainingDetectionId, detectionId];

      await client.query(
        `INSERT INTO cannot_link (detection_id_1, detection_id_2, collection_id, created_by)
         VALUES ($1, $2, $3, 'web')
         ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING`,
        [id1, id2, collectionId],
      );
    }

    await client.query("COMMIT");

    return { success: true, message: "Detection removed from cluster" };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to remove detection from cluster:", error);
    return { success: false, message: "Failed to remove detection from cluster" };
  } finally {
    client.release();
  }
}

/**
 * Get all albums in a collection with their representative photos.
 * If representative_photo_id is not set, uses the first photo in the album.
 */
export async function getAlbums(collectionId: number, limit = 50, offset = 0) {
  const query = `
    SELECT
      a.id,
      a.name,
      a.representative_photo_id,
      a.created_at,
      a.updated_at,
      COALESCE(a.representative_photo_id, (
        SELECT pa.photo_id
        FROM photo_album pa
        WHERE pa.album_id = a.id
        ORDER BY pa.added_at
        LIMIT 1
      )) as display_photo_id,
      (SELECT COUNT(*) FROM photo_album pa WHERE pa.album_id = a.id)::int as photo_count
    FROM album a
    WHERE a.collection_id = $1
    ORDER BY a.name, a.id
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
}

export async function getAlbumsCount(collectionId: number) {
  const query = `
    SELECT COUNT(*) as count
    FROM album
    WHERE collection_id = $1
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

export async function getAlbumById(collectionId: number, albumId: string) {
  const query = `
    SELECT
      a.id,
      a.name,
      a.representative_photo_id,
      a.created_at,
      a.updated_at,
      COALESCE(a.representative_photo_id, (
        SELECT pa.photo_id
        FROM photo_album pa
        WHERE pa.album_id = a.id
        ORDER BY pa.added_at
        LIMIT 1
      )) as display_photo_id,
      (SELECT COUNT(*) FROM photo_album pa WHERE pa.album_id = a.id)::int as photo_count
    FROM album a
    WHERE a.id = $1 AND a.collection_id = $2
  `;

  const result = await pool.query(query, [albumId, collectionId]);
  return result.rows[0] || null;
}

/**
 * Get photos in an album with metadata for display.
 */
export async function getAlbumPhotos(collectionId: number, albumId: string, limit = 100, offset = 0) {
  const query = `
    SELECT p.id, p.filename, p.normalized_path,
           p.normalized_width, p.normalized_height,
           m.captured_at, m.latitude, m.longitude,
           la.description, la.emotional_tone,
           pa.added_at
    FROM photo_album pa
    JOIN photo p ON pa.photo_id = p.id
    LEFT JOIN metadata m ON p.id = m.photo_id
    LEFT JOIN llm_analysis la ON p.id = la.photo_id
    WHERE pa.album_id = $1 AND p.collection_id = $2
    ORDER BY pa.added_at, p.id
    LIMIT $3 OFFSET $4
  `;

  const result = await pool.query(query, [albumId, collectionId, limit, offset]);
  return result.rows;
}

export async function getAlbumPhotosCount(collectionId: number, albumId: string) {
  const query = `
    SELECT COUNT(*) as count
    FROM photo_album pa
    JOIN photo p ON pa.photo_id = p.id
    WHERE pa.album_id = $1 AND p.collection_id = $2
  `;

  const result = await pool.query(query, [albumId, collectionId]);
  return parseInt(result.rows[0].count, 10);
}
