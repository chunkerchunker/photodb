import path from "node:path";
import dotenv from "dotenv";
import type { PoolClient } from "pg";
import { Pool } from "pg";
import { toSubsequenceLikePattern } from "~/lib/utils";

// Load environment variables from .env file
dotenv.config({ path: path.join(process.cwd(), "..", ".env") });

// Create a connection pool
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || "postgresql://localhost/photodb",
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 10000, // 10s timeout for acquiring connection from pool
});

// Log unexpected connection errors (e.g., server restarts, network issues)
pool.on("error", (err) => {
  console.error("Unexpected pool error:", err);
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
    SELECT p.id, p.orig_path, p.med_path,
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
    ORDER BY m.captured_at, p.orig_path
    LIMIT $4 OFFSET $5
  `;

  const result = await pool.query(query, [year, month, collectionId, limit, offset]);

  // Process photos to add computed fields
  const photos = result.rows.map((photo) => ({
    ...photo,
    filename_only: path.basename(photo.orig_path),
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
    SELECT p.id, p.orig_path, p.full_path, p.med_path,
           p.created_at as photo_created_at, p.updated_at as photo_updated_at,
           p.width, p.height, p.med_width, p.med_height,
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
             NULLIF(TRIM(CONCAT(COALESCE(p.preferred_name, p.first_name), ' ', COALESCE(p.last_name, ''))), ''),
             NULLIF(TRIM(CONCAT(COALESCE(cp.preferred_name, cp.first_name), ' ', COALESCE(cp.last_name, ''))), '')
           ) as person_name,
           COALESCE(p.auto_created, cp.auto_created, false) as auto_created,
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
             TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
             COALESCE(per.auto_created, false) as auto_created,
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
  photo.orig_path_only = path.basename(photo.orig_path);

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
  photo.image_width = photo.med_width || null;
  photo.image_height = photo.med_height || null;

  return photo;
}

export async function getPhotoById(collectionId: number, photoId: number) {
  const query = `
    SELECT id, orig_path, full_path, med_path
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
  is_admin: boolean;
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

/**
 * Update user's first and last name.
 */
export async function updateUserProfile(userId: number, firstName: string, lastName: string | null): Promise<void> {
  await pool.query("UPDATE app_user SET first_name = $1, last_name = $2 WHERE id = $3", [firstName, lastName, userId]);
}

// ============================================================================
// Admin Functions
// ============================================================================

export type UserListCollection = {
  id: number;
  name: string;
};

export type UserListItem = {
  id: number;
  username: string;
  first_name: string;
  last_name: string;
  is_admin: boolean;
  default_collection_id: number | null;
  collections: UserListCollection[];
  created_at: string;
};

/**
 * Get all users for admin user list.
 * Returns users with their collection memberships.
 */
export async function getAllUsers(): Promise<UserListItem[]> {
  const query = `
    SELECT
      u.id,
      u.username,
      u.first_name,
      u.last_name,
      u.is_admin,
      u.default_collection_id,
      u.created_at,
      COALESCE(
        json_agg(json_build_object('id', c.id, 'name', c.name) ORDER BY c.name)
        FILTER (WHERE c.id IS NOT NULL),
        '[]'
      ) as collections
    FROM app_user u
    LEFT JOIN collection_member cm ON u.id = cm.user_id
    LEFT JOIN collection c ON cm.collection_id = c.id
    GROUP BY u.id
    ORDER BY u.created_at DESC
  `;

  const result = await pool.query(query);
  return result.rows;
}

export async function getClusters(collectionId: number, limit = 50, offset = 0) {
  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           pd.face_path,
           p.id as photo_id, p.med_path, p.orig_path,
           p.med_width, p.med_height,
           TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name
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
export async function getClustersGroupedByPerson(collectionId: number, limit = 50, offset = 0, search?: string) {
  const query = `
    WITH person_aggregates AS (
      -- Aggregate clusters by person
      SELECT
        'person' as item_type,
        per.id as id,
        per.id as person_id,
        SUM(c.face_count)::integer as face_count,
        COUNT(c.id)::integer as cluster_count,
        TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
        COALESCE(per.auto_created, false) as auto_created,
        -- Get the largest cluster ID for drag-drop operations
        (array_agg(c.id ORDER BY c.face_count DESC))[1] as primary_cluster_id,
        -- Get representative detection (person's or fallback to largest cluster's)
        COALESCE(
          per.representative_detection_id,
          (array_agg(c.representative_detection_id ORDER BY c.face_count DESC)
           FILTER (WHERE c.representative_detection_id IS NOT NULL))[1]
        ) as representative_detection_id
      FROM person per
      INNER JOIN cluster c ON c.person_id = per.id
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $1
        AND ($4::text IS NULL OR
             per.first_name ILIKE '%' || $4 || '%'
             OR per.preferred_name ILIKE '%' || $4 || '%'
             OR per.last_name ILIKE '%' || $4 || '%'
             OR CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, '')) ILIKE '%' || $4 || '%'
             OR per.id::text = $4)
      GROUP BY per.id
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
        false as auto_created,
        c.id as primary_cluster_id,
        c.representative_detection_id
      FROM cluster c
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $1
        AND c.person_id IS NULL
        AND ($4::text IS NULL OR c.id::text = $4)
    ),
    combined AS (
      SELECT * FROM person_aggregates
      UNION ALL
      SELECT * FROM unassigned_clusters
    )
    SELECT
      combined.*,
      pd.face_path,
      pd.id as detection_id
    FROM combined
    LEFT JOIN person_detection pd ON combined.representative_detection_id = pd.id
    ORDER BY combined.face_count DESC, combined.id
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset, search || null]);
  return result.rows;
}

/**
 * Get count of items for the grouped clusters view.
 * Counts distinct people + unassigned clusters.
 */
export async function getClustersGroupedCount(collectionId: number, search?: string) {
  const query = `
    SELECT
      (
        -- Count distinct people with visible clusters
        SELECT COUNT(DISTINCT per.id)
        FROM person per
        INNER JOIN cluster c ON c.person_id = per.id
        WHERE c.face_count > 0
          AND (c.hidden = false OR c.hidden IS NULL)
          AND c.collection_id = $1
          AND ($2::text IS NULL OR
               per.first_name ILIKE '%' || $2 || '%'
               OR per.preferred_name ILIKE '%' || $2 || '%'
               OR per.last_name ILIKE '%' || $2 || '%'
               OR CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, '')) ILIKE '%' || $2 || '%'
               OR per.id::text = $2)
      ) + (
        -- Count unassigned clusters
        SELECT COUNT(*)
        FROM cluster c
        WHERE c.face_count > 0
          AND (c.hidden = false OR c.hidden IS NULL)
          AND c.collection_id = $1
          AND c.person_id IS NULL
          AND ($2::text IS NULL OR c.id::text = $2)
      ) as count
  `;

  const result = await pool.query(query, [collectionId, search || null]);
  return parseInt(result.rows[0].count, 10);
}

export async function getHiddenClusters(collectionId: number, limit = 50, offset = 0) {
  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           pd.face_path, pd.id as detection_id,
           TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
           COALESCE(per.auto_created, false) as auto_created
    FROM cluster c
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
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
      ? "c.face_count DESC, COALESCE(per.preferred_name, per.first_name), per.last_name, c.id"
      : "COALESCE(per.preferred_name, per.first_name), per.last_name, c.id";

  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           pd.face_path, pd.id as detection_id,
           TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    INNER JOIN person per ON c.person_id = per.id
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
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
 * @param includeWithoutImages - If true, includes people without linked clusters (no images)
 */
export async function getPeople(
  collectionId: number,
  limit = 50,
  offset = 0,
  sort: "photos" | "name" = "name",
  includeWithoutImages = false,
) {
  const orderBy =
    sort === "photos"
      ? "COALESCE(ps.total_face_count, 0) DESC, COALESCE(per.preferred_name, per.first_name), per.last_name, per.id"
      : "COALESCE(per.auto_created, false), COALESCE(per.preferred_name, per.first_name), per.last_name, per.id";

  // Filter clause for people without images (no linked clusters)
  const withoutImagesFilter = includeWithoutImages ? "" : "AND ps.person_id IS NOT NULL";

  const query = `
    WITH person_stats AS (
      SELECT
        c.person_id,
        SUM(c.face_count)::integer as total_face_count,
        COUNT(c.id)::integer as cluster_count,
        -- Fallback: get the representative from the largest cluster
        (array_agg(c.representative_detection_id ORDER BY c.face_count DESC)
         FILTER (WHERE c.representative_detection_id IS NOT NULL))[1]
         as fallback_representative_detection_id
      FROM cluster c
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $1
        AND c.person_id IS NOT NULL
      GROUP BY c.person_id
    )
    SELECT
      per.id,
      TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
      COALESCE(per.auto_created, false) as auto_created,
      COALESCE(ps.total_face_count, 0) as total_face_count,
      COALESCE(ps.cluster_count, 0) as cluster_count,
      -- Use person's representative if set, otherwise use fallback from largest cluster
      COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) as representative_detection_id,
      pd.face_path,
      pd.id as detection_id
    FROM person per
    LEFT JOIN person_stats ps ON per.id = ps.person_id
    LEFT JOIN person_detection pd ON COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) = pd.id
    WHERE per.collection_id = $1
      AND (per.hidden = false OR per.hidden IS NULL)
      ${withoutImagesFilter}
    ORDER BY ${orderBy}
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
}

/**
 * Get count of unique people (not clusters).
 * @param includeWithoutImages - If true, includes people without linked clusters (no images)
 */
export async function getPeopleCount(collectionId: number, includeWithoutImages = false) {
  if (includeWithoutImages) {
    // Count all non-hidden people
    const query = `
      SELECT COUNT(*) as count
      FROM person per
      WHERE per.collection_id = $1
        AND (per.hidden = false OR per.hidden IS NULL)
    `;
    const result = await pool.query(query, [collectionId]);
    return parseInt(result.rows[0].count, 10);
  }

  // Count only people with linked clusters (original behavior)
  const query = `
    SELECT COUNT(DISTINCT c.person_id) as count
    FROM cluster c
    INNER JOIN person per ON c.person_id = per.id
    WHERE c.face_count > 0
      AND (c.hidden = false OR c.hidden IS NULL)
      AND c.collection_id = $1
      AND c.person_id IS NOT NULL
      AND (per.hidden = false OR per.hidden IS NULL)
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

/**
 * Get count of people without linked clusters (no images).
 */
export async function getPeopleWithoutImagesCount(collectionId: number) {
  const query = `
    SELECT COUNT(*) as count
    FROM person per
    WHERE per.collection_id = $1
      AND (per.hidden = false OR per.hidden IS NULL)
      AND NOT EXISTS (
        SELECT 1 FROM cluster c
        WHERE c.person_id = per.id
          AND c.face_count > 0
          AND (c.hidden = false OR c.hidden IS NULL)
      )
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

/**
 * Get count of hidden people.
 */
export async function getHiddenPeopleCount(collectionId: number) {
  const query = `
    SELECT COUNT(*) as count
    FROM person per
    WHERE per.collection_id = $1
      AND per.hidden = true
      AND EXISTS (
        SELECT 1 FROM cluster c
        WHERE c.person_id = per.id
          AND c.face_count > 0
      )
  `;

  const result = await pool.query(query, [collectionId]);
  return parseInt(result.rows[0].count, 10);
}

/**
 * Get hidden people with aggregated cluster data.
 */
export async function getHiddenPeople(collectionId: number, limit = 50, offset = 0, sort: "photos" | "name" = "name") {
  const orderBy =
    sort === "photos"
      ? "total_face_count DESC, COALESCE(per.preferred_name, per.first_name), per.last_name, per.id"
      : "COALESCE(per.preferred_name, per.first_name), per.last_name, per.id";

  const query = `
    WITH person_stats AS (
      SELECT
        c.person_id,
        SUM(c.face_count)::integer as total_face_count,
        COUNT(c.id)::integer as cluster_count,
        -- Fallback: get the representative from the largest cluster
        (array_agg(c.representative_detection_id ORDER BY c.face_count DESC)
         FILTER (WHERE c.representative_detection_id IS NOT NULL))[1]
         as fallback_representative_detection_id
      FROM cluster c
      WHERE c.face_count > 0
        AND c.collection_id = $1
        AND c.person_id IS NOT NULL
      GROUP BY c.person_id
    )
    SELECT
      per.id,
      TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
      ps.total_face_count,
      ps.cluster_count,
      -- Use person's representative if set, otherwise use fallback from largest cluster
      COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) as representative_detection_id,
      pd.face_path,
      pd.id as detection_id
    FROM person per
    INNER JOIN person_stats ps ON per.id = ps.person_id
    LEFT JOIN person_detection pd ON COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) = pd.id
    WHERE per.collection_id = $1
      AND per.hidden = true
    ORDER BY ${orderBy}
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [collectionId, limit, offset]);
  return result.rows;
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
        (array_agg(c.representative_detection_id ORDER BY c.face_count DESC)
         FILTER (WHERE c.representative_detection_id IS NOT NULL))[1]
         as fallback_representative_detection_id
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
      per.middle_name,
      per.maiden_name,
      per.preferred_name,
      per.suffix,
      per.alternate_names,
      TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
      COALESCE(per.auto_created, false) as auto_created,
      COALESCE(ps.total_face_count, 0)::integer as total_face_count,
      COALESCE(ps.cluster_count, 0)::integer as cluster_count,
      COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) as representative_detection_id,
      pd.face_path,
      pd.id as detection_id
    FROM person per
    LEFT JOIN person_stats ps ON per.id = ps.person_id
    LEFT JOIN person_detection pd ON COALESCE(per.representative_detection_id, ps.fallback_representative_detection_id) = pd.id
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
           c.age_estimate, c.age_estimate_stddev,
           pd.face_path, pd.id as detection_id
    FROM cluster c
    LEFT JOIN person_detection pd ON c.representative_detection_id = pd.id
    WHERE c.person_id = $1 AND c.collection_id = $2 AND c.face_count > 0
    ORDER BY c.face_count DESC, c.id
  `;

  const result = await pool.query(query, [personId, collectionId]);
  return result.rows;
}

/**
 * Rename a person directly.
 */
export async function setPersonName(
  collectionId: number,
  personId: string,
  firstName: string,
  lastName?: string,
  middleName?: string,
  maidenName?: string,
  preferredName?: string,
  suffix?: string,
  alternateNames?: string[],
) {
  const query = `
    UPDATE person
    SET first_name = $1, last_name = $2, middle_name = $3, maiden_name = $4,
        preferred_name = $5, suffix = $6, alternate_names = $7,
        auto_created = false, updated_at = NOW()
    WHERE id = $8 AND collection_id = $9
    RETURNING id
  `;

  const result = await pool.query(query, [
    firstName,
    lastName || null,
    middleName || null,
    maidenName || null,
    preferredName || null,
    suffix || null,
    alternateNames || [],
    personId,
    collectionId,
  ]);
  return {
    success: result.rows.length > 0,
    message: result.rows.length > 0 ? "Person renamed" : "Person not found",
  };
}

/**
 * Hide or unhide a person and all their clusters.
 */
export async function setPersonHidden(collectionId: number, personId: string, hidden: boolean) {
  // Update the person's hidden flag
  const personQuery = `
    UPDATE person
    SET hidden = $1, updated_at = NOW()
    WHERE id = $2 AND collection_id = $3
    RETURNING id
  `;
  await pool.query(personQuery, [hidden, personId, collectionId]);

  // Also update all clusters belonging to this person
  const clusterQuery = `
    UPDATE cluster
    SET hidden = $1, updated_at = NOW()
    WHERE person_id = $2 AND collection_id = $3
    RETURNING id
  `;

  const result = await pool.query(clusterQuery, [hidden, personId, collectionId]);
  const count = result.rowCount || 0;
  return {
    success: true,
    message: hidden ? `Hidden person and ${count} cluster(s)` : `Unhidden person and ${count} cluster(s)`,
    count,
  };
}

/**
 * Unlink a cluster from its person (set person_id to NULL).
 */
export async function unlinkClusterFromPerson(collectionId: number, clusterId: string) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Read current person_id before unlinking
    const current = await client.query(`SELECT person_id FROM cluster WHERE id = $1 AND collection_id = $2`, [
      clusterId,
      collectionId,
    ]);
    if (current.rows.length === 0 || !current.rows[0].person_id) {
      await client.query("ROLLBACK");
      return { success: false, message: current.rows.length === 0 ? "Cluster not found" : "Cluster has no person" };
    }
    const personId = current.rows[0].person_id;

    // Unlink cluster from person
    await client.query(`UPDATE cluster SET person_id = NULL, updated_at = NOW() WHERE id = $1 AND collection_id = $2`, [
      clusterId,
      collectionId,
    ]);

    // Add cannot-link to prevent re-association
    await client.query(
      `INSERT INTO cluster_person_cannot_link (cluster_id, person_id, collection_id)
       VALUES ($1, $2, $3)
       ON CONFLICT (cluster_id, person_id) DO NOTHING`,
      [clusterId, personId, collectionId],
    );

    // If person was auto_created and now has no remaining clusters, delete it
    let deletedPerson = false;
    const remaining = await client.query(`SELECT COUNT(*) FROM cluster WHERE person_id = $1 AND collection_id = $2`, [
      personId,
      collectionId,
    ]);
    if (parseInt(remaining.rows[0].count, 10) === 0) {
      const personRow = await client.query(`SELECT auto_created FROM person WHERE id = $1 AND collection_id = $2`, [
        personId,
        collectionId,
      ]);
      if (personRow.rows.length > 0 && personRow.rows[0].auto_created) {
        await client.query(`DELETE FROM person WHERE id = $1 AND collection_id = $2`, [personId, collectionId]);
        deletedPerson = true;
      }
    }

    await client.query("COMMIT");
    return {
      success: true,
      message: deletedPerson
        ? "Cluster unlinked and empty auto-created person deleted"
        : "Cluster unlinked from person",
      deletedPerson,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

/**
 * Delete a person aggregation by clearing person_id on all associated clusters
 * and removing the person's representative detection.
 */
export async function deletePersonAggregation(collectionId: number, personId: string) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Clear person_id from all clusters belonging to this person
    const unlinkResult = await client.query(
      `UPDATE cluster
       SET person_id = NULL, updated_at = NOW()
       WHERE person_id = $1 AND collection_id = $2`,
      [personId, collectionId],
    );

    // If person was auto_created, delete it entirely since it now has no clusters
    let deletedPerson = false;
    const personRow = await client.query(`SELECT auto_created FROM person WHERE id = $1 AND collection_id = $2`, [
      personId,
      collectionId,
    ]);
    if (personRow.rows.length > 0 && personRow.rows[0].auto_created) {
      await client.query(`DELETE FROM person WHERE id = $1 AND collection_id = $2`, [personId, collectionId]);
      deletedPerson = true;
    } else {
      // Clear the person's representative detection (keep the person row)
      await client.query(
        `UPDATE person
         SET representative_detection_id = NULL, updated_at = NOW()
         WHERE id = $1 AND collection_id = $2`,
        [personId, collectionId],
      );
    }

    await client.query("COMMIT");
    return {
      success: true,
      message: deletedPerson
        ? `Unlinked ${unlinkResult.rowCount} cluster(s) and deleted auto-created person`
        : `Unlinked ${unlinkResult.rowCount} cluster(s) from person`,
      unlinkedCount: unlinkResult.rowCount,
      deletedPerson,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

/**
 * Delete a person row entirely (only when they have no clusters).
 */
export async function deletePersonRow(collectionId: number, personId: string) {
  const result = await pool.query(`DELETE FROM person WHERE id = $1 AND collection_id = $2`, [personId, collectionId]);
  return {
    success: (result.rowCount ?? 0) > 0,
    message: result.rowCount ? "Person deleted" : "Person not found",
  };
}

/**
 * Set a person's representative detection directly from a detection ID.
 */
export async function setPersonRepresentativeDetection(collectionId: number, personId: string, detectionId: number) {
  const result = await pool.query(
    `UPDATE person
     SET representative_detection_id = $1, updated_at = NOW()
     WHERE id = $2 AND collection_id = $3`,
    [detectionId, personId, collectionId],
  );
  return {
    success: (result.rowCount ?? 0) > 0,
    message: result.rowCount ? "Person representative updated" : "Person not found",
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
  middleName?: string,
  maidenName?: string,
  preferredName?: string,
  suffix?: string,
  alternateNames?: string[],
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
        `UPDATE person SET first_name = $1, last_name = $2, middle_name = $3, maiden_name = $4,
                preferred_name = $5, suffix = $6, alternate_names = $7,
                auto_created = false, updated_at = NOW()
         WHERE id = $8 AND collection_id = $9`,
        [
          firstName,
          lastName || null,
          middleName || null,
          maidenName || null,
          preferredName || null,
          suffix || null,
          alternateNames || [],
          existingPersonId,
          collectionId,
        ],
      );
    } else {
      // Create new person and link to cluster
      const personResult = await client.query(
        `INSERT INTO person (collection_id, first_name, last_name, middle_name, maiden_name, preferred_name, suffix, alternate_names)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING id`,
        [
          collectionId,
          firstName,
          lastName || null,
          middleName || null,
          maidenName || null,
          preferredName || null,
          suffix || null,
          alternateNames || [],
        ],
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
           c.hidden, c.person_id,
           per.first_name, per.last_name,
           per.gender as person_gender, per.gender_confidence as person_gender_confidence,
           per.estimated_birth_year, per.birth_year_stddev,
           TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
           rep.face_path as rep_face_path,
           rep.id as rep_detection_id
    FROM cluster c
    LEFT JOIN person per ON c.person_id = per.id
    LEFT JOIN person_detection rep ON c.representative_detection_id = rep.id
    WHERE c.id = $1 AND c.collection_id = $2
  `;

  const result = await pool.query(query, [clusterId, collectionId]);
  return result.rows[0] || null;
}

export async function getClusterFaces(collectionId: number, clusterId: string, limit = 24, offset = 0) {
  const query = `
    SELECT pd.id, pd.face_path,
           pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.cluster_confidence, pd.photo_id,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           p.med_width, p.med_height
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
  // Search persons (aggregated) and unassigned clusters.
  // Person results use their primary cluster ID as `id` for backward compatibility
  // with callers that pass `id` directly to the merge/link API.
  // Name matching uses subsequence patterns (e.g. "jhn" matches "John").
  const subseqPattern = query ? toSubsequenceLikePattern(query) : null;
  const searchQuery = `
    WITH source_person AS (
      SELECT person_id FROM cluster
      WHERE id::text = $2 AND collection_id = $4
    ),
    person_results AS (
      SELECT
        'person' as item_type,
        (array_agg(c.id ORDER BY c.face_count DESC NULLS LAST)
         FILTER (WHERE c.id IS NOT NULL))[1]::text as id,
        per.id::text as person_id,
        COALESCE(SUM(c.face_count) FILTER (WHERE c.hidden = false OR c.hidden IS NULL), 0)::integer as face_count,
        COALESCE(
          per.representative_detection_id,
          (array_agg(c.representative_detection_id ORDER BY c.face_count DESC)
           FILTER (WHERE c.representative_detection_id IS NOT NULL
             AND (c.hidden = false OR c.hidden IS NULL)))[1]
        ) as representative_detection_id,
        TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
        COUNT(c.id) FILTER (WHERE c.hidden = false OR c.hidden IS NULL)::integer as cluster_count,
        CASE WHEN EXISTS (
          SELECT 1 FROM cluster c3
          WHERE c3.person_id = per.id AND c3.collection_id = $4 AND c3.id::text = $1
        ) THEN 0 ELSE 1 END as id_match_rank,
        CASE WHEN COALESCE(per.preferred_name, per.first_name) ILIKE $1 || '%' OR per.last_name ILIKE $1 || '%' THEN 0 ELSE 1 END as name_match_rank
      FROM person per
      LEFT JOIN cluster c ON c.person_id = per.id AND c.collection_id = $4
      WHERE per.collection_id = $4
        AND ($5::text IS NULL
             OR per.first_name ILIKE $5
             OR per.preferred_name ILIKE $5
             OR per.last_name ILIKE $5
             OR CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, '')) ILIKE $5
             OR EXISTS (SELECT 1 FROM cluster c3
                        WHERE c3.person_id = per.id AND c3.collection_id = $4 AND c3.id::text = $1))
        AND ($2::text IS NULL OR per.id IS DISTINCT FROM (SELECT person_id FROM source_person))
      GROUP BY per.id
    ),
    cluster_results AS (
      SELECT
        'cluster' as item_type,
        c.id::text as id,
        NULL::text as person_id,
        c.face_count::integer as face_count,
        c.representative_detection_id,
        NULL::text as person_name,
        1::integer as cluster_count,
        CASE WHEN c.id::text = $1 THEN 0 ELSE 1 END as id_match_rank,
        1::integer as name_match_rank
      FROM cluster c
      WHERE c.face_count > 0
        AND (c.hidden = false OR c.hidden IS NULL)
        AND c.collection_id = $4
        AND c.person_id IS NULL
        AND ($1::text IS NULL OR c.id::text = $1)
        AND ($2::text IS NULL OR c.id::text != $2)
    )
    SELECT item_type, combined.id, combined.person_id, face_count, person_name, cluster_count,
           pd.face_path, pd.id as detection_id
    FROM (
      SELECT * FROM person_results
      UNION ALL
      SELECT * FROM cluster_results
    ) combined
    LEFT JOIN person_detection pd ON combined.representative_detection_id = pd.id
    ORDER BY
      combined.id_match_rank,
      combined.name_match_rank,
      combined.face_count DESC
    LIMIT $3
  `;

  const result = await pool.query(searchQuery, [
    query || null,
    excludeClusterId || null,
    limit,
    collectionId,
    subseqPattern,
  ]);
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
      TRIM(CONCAT(COALESCE(p.preferred_name, p.first_name), ' ', COALESCE(p.last_name, ''))) as person_name,
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
      personClusterCount: parseInt(sourceInfo.person_cluster_count, 10) || 0,
    },
    target: {
      clusterId: targetClusterId,
      personId: targetInfo.person_id,
      personName: targetInfo.person_name || null,
      personClusterCount: parseInt(targetInfo.person_cluster_count, 10) || 0,
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
             TRIM(CONCAT(COALESCE(p.preferred_name, p.first_name), ' ', COALESCE(p.last_name, ''))) as person_name
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
             TRIM(CONCAT(COALESCE(p.preferred_name, p.first_name), ' ', COALESCE(p.last_name, ''))) as person_name
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

      // Migrate cannot-link constraints from deleted person to kept person
      await client.query(
        `UPDATE cluster_person_cannot_link SET person_id = $1
         WHERE person_id = $2
         AND NOT EXISTS (
           SELECT 1 FROM cluster_person_cannot_link c2
           WHERE c2.cluster_id = cluster_person_cannot_link.cluster_id AND c2.person_id = $1
         )`,
        [personId, sourceCluster.person_id],
      );
      await client.query(`DELETE FROM cluster_person_cannot_link WHERE person_id = $1`, [sourceCluster.person_id]);

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
        `INSERT INTO person (collection_id, representative_detection_id, auto_created, created_at)
         VALUES ($1, $2, true, NOW()) RETURNING id`,
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

    // Remove any cannot-links between these clusters and their new person
    await client.query(
      `DELETE FROM cluster_person_cannot_link
       WHERE cluster_id = ANY($1::int[]) AND person_id = $2`,
      [[sourceClusterId, targetClusterId], personId],
    );

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

/**
 * Assign a cluster directly to an existing person.
 * Used when linking to a person that may have no clusters.
 * If the source cluster already belongs to a different person,
 * that person's clusters are all merged into the target person.
 */
export async function assignClusterToPerson(collectionId: number, sourceClusterId: string, targetPersonId: string) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    const sourceResult = await client.query(
      `SELECT c.id, c.person_id,
              TRIM(CONCAT(COALESCE(p.preferred_name, p.first_name), ' ', COALESCE(p.last_name, ''))) as person_name
       FROM cluster c
       LEFT JOIN person p ON c.person_id = p.id
       WHERE c.id = $1 AND c.collection_id = $2`,
      [sourceClusterId, collectionId],
    );
    if (sourceResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Source cluster not found" };
    }
    const sourceCluster = sourceResult.rows[0];

    const targetResult = await client.query(
      `SELECT id, TRIM(CONCAT(COALESCE(preferred_name, first_name), ' ', COALESCE(last_name, ''))) as person_name
       FROM person WHERE id = $1 AND collection_id = $2`,
      [targetPersonId, collectionId],
    );
    if (targetResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Target person not found" };
    }
    const targetPerson = targetResult.rows[0];

    if (sourceCluster.person_id === targetPerson.id) {
      await client.query("ROLLBACK");
      return { success: true, message: "Already linked" };
    }

    if (sourceCluster.person_id && sourceCluster.person_id !== targetPerson.id) {
      // Source has a different person - merge into target
      const sourcePersonId = sourceCluster.person_id;
      await client.query(
        `UPDATE cluster SET person_id = $1, updated_at = NOW()
         WHERE person_id = $2 AND collection_id = $3`,
        [targetPerson.id, sourcePersonId, collectionId],
      );

      // Migrate cannot-link constraints from deleted person to kept person
      await client.query(
        `UPDATE cluster_person_cannot_link SET person_id = $1
         WHERE person_id = $2
         AND NOT EXISTS (
           SELECT 1 FROM cluster_person_cannot_link c2
           WHERE c2.cluster_id = cluster_person_cannot_link.cluster_id AND c2.person_id = $1
         )`,
        [targetPerson.id, sourcePersonId],
      );
      await client.query(`DELETE FROM cluster_person_cannot_link WHERE person_id = $1`, [sourcePersonId]);

      await client.query(`DELETE FROM person WHERE id = $1 AND collection_id = $2`, [sourcePersonId, collectionId]);
    } else {
      await client.query(`UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`, [
        targetPerson.id,
        sourceClusterId,
        collectionId,
      ]);
    }

    // Remove any cannot-link between this cluster and the target person
    await client.query(`DELETE FROM cluster_person_cannot_link WHERE cluster_id = $1 AND person_id = $2`, [
      sourceClusterId,
      targetPerson.id,
    ]);

    await client.query("COMMIT");
    return { success: true, message: `Linked to "${targetPerson.person_name}"`, personId: targetPerson.id };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to assign cluster to person:", error);
    return { success: false, message: "Failed to assign cluster to person" };
  } finally {
    client.release();
  }
}

/**
 * Reassign a single cluster to a different person (or to a cluster's person).
 * Unlike assignClusterToPerson/linkClustersToSamePerson which merge ALL clusters
 * from the old person, this only moves the one cluster. The old person is deleted
 * only if it becomes empty and was auto_created.
 */
export async function reassignCluster(
  collectionId: number,
  clusterId: string,
  target: { personId?: string; clusterId?: string },
) {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Get source cluster
    const sourceResult = await client.query(`SELECT id, person_id FROM cluster WHERE id = $1 AND collection_id = $2`, [
      clusterId,
      collectionId,
    ]);
    if (sourceResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Source cluster not found" };
    }
    const oldPersonId = sourceResult.rows[0].person_id;

    // Resolve target person ID
    let targetPersonId: number;
    let targetPersonName: string;

    if (target.personId) {
      const personResult = await client.query(
        `SELECT id, TRIM(CONCAT(COALESCE(preferred_name, first_name), ' ', COALESCE(last_name, ''))) as person_name
         FROM person WHERE id = $1 AND collection_id = $2`,
        [target.personId, collectionId],
      );
      if (personResult.rows.length === 0) {
        await client.query("ROLLBACK");
        return { success: false, message: "Target person not found" };
      }
      targetPersonId = personResult.rows[0].id;
      targetPersonName = personResult.rows[0].person_name;
    } else if (target.clusterId) {
      const targetClusterResult = await client.query(
        `SELECT c.id, c.person_id,
                TRIM(CONCAT(COALESCE(p.preferred_name, p.first_name), ' ', COALESCE(p.last_name, ''))) as person_name
         FROM cluster c
         LEFT JOIN person p ON c.person_id = p.id
         WHERE c.id = $1 AND c.collection_id = $2`,
        [target.clusterId, collectionId],
      );
      if (targetClusterResult.rows.length === 0) {
        await client.query("ROLLBACK");
        return { success: false, message: "Target cluster not found" };
      }
      const targetCluster = targetClusterResult.rows[0];
      if (targetCluster.person_id) {
        targetPersonId = targetCluster.person_id;
        targetPersonName = targetCluster.person_name;
      } else {
        // Target cluster has no person — create one and assign the target cluster to it
        const newPerson = await client.query(
          `INSERT INTO person (collection_id, auto_created, representative_detection_id)
           SELECT $1, true, representative_detection_id FROM cluster WHERE id = $2 AND collection_id = $1
           RETURNING id`,
          [collectionId, target.clusterId],
        );
        targetPersonId = newPerson.rows[0].id;
        targetPersonName = `Cluster ${target.clusterId}`;
        await client.query(
          `UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`,
          [targetPersonId, target.clusterId, collectionId],
        );
      }
    } else {
      await client.query("ROLLBACK");
      return { success: false, message: "No target specified" };
    }

    if (oldPersonId === targetPersonId) {
      await client.query("ROLLBACK");
      return { success: true, message: "Already assigned to this person" };
    }

    // Reassign only this cluster
    await client.query(`UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2 AND collection_id = $3`, [
      targetPersonId,
      clusterId,
      collectionId,
    ]);

    // Remove any cannot-link between this cluster and the target person
    await client.query(`DELETE FROM cluster_person_cannot_link WHERE cluster_id = $1 AND person_id = $2`, [
      clusterId,
      targetPersonId,
    ]);

    // Clean up old person if it's now empty and was auto_created
    if (oldPersonId) {
      const remaining = await client.query(`SELECT COUNT(*) FROM cluster WHERE person_id = $1 AND collection_id = $2`, [
        oldPersonId,
        collectionId,
      ]);
      if (parseInt(remaining.rows[0].count, 10) === 0) {
        const oldPerson = await client.query(`SELECT auto_created FROM person WHERE id = $1 AND collection_id = $2`, [
          oldPersonId,
          collectionId,
        ]);
        if (oldPerson.rows.length > 0 && oldPerson.rows[0].auto_created) {
          await client.query(`DELETE FROM cluster_person_cannot_link WHERE person_id = $1`, [oldPersonId]);
          await client.query(`DELETE FROM person WHERE id = $1 AND collection_id = $2`, [oldPersonId, collectionId]);
        }
      }
    }

    await client.query("COMMIT");
    return { success: true, message: `Reassigned to "${targetPersonName}"` };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to reassign cluster:", error);
    return { success: false, message: "Failed to reassign cluster" };
  } finally {
    client.release();
  }
}

/**
 * Preview what will happen when merging two persons.
 */
export async function previewPersonMerge(collectionId: number, sourcePersonId: string, targetPersonId: string) {
  const query = `
    SELECT
      per.id as person_id,
      TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
      (SELECT COUNT(*) FROM cluster c WHERE c.person_id = per.id
        AND (c.hidden = false OR c.hidden IS NULL)) as cluster_count,
      (SELECT COALESCE(SUM(c.face_count), 0) FROM cluster c WHERE c.person_id = per.id
        AND (c.hidden = false OR c.hidden IS NULL)) as face_count
    FROM person per
    WHERE per.id = ANY($1) AND per.collection_id = $2
  `;

  const result = await pool.query(query, [[sourcePersonId, targetPersonId], collectionId]);

  const sourceInfo = result.rows.find((r) => r.person_id.toString() === sourcePersonId);
  const targetInfo = result.rows.find((r) => r.person_id.toString() === targetPersonId);

  if (!sourceInfo || !targetInfo) {
    return { found: false };
  }

  return {
    found: true,
    willMergePersons: true,
    source: {
      personName: sourceInfo.person_name || null,
      personClusterCount: parseInt(sourceInfo.cluster_count, 10) || 0,
      personFaceCount: parseInt(sourceInfo.face_count, 10) || 0,
    },
    target: {
      personName: targetInfo.person_name || null,
      personClusterCount: parseInt(targetInfo.cluster_count, 10) || 0,
      personFaceCount: parseInt(targetInfo.face_count, 10) || 0,
    },
  };
}

/**
 * Merge two persons by moving all clusters from source to target and deleting the source person.
 */
export async function mergePersons(collectionId: number, sourcePersonId: string, targetPersonId: string) {
  if (sourcePersonId === targetPersonId) {
    return { success: false, message: "Cannot merge a person with itself" };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Verify both persons exist
    const personsResult = await client.query(
      `SELECT id, TRIM(CONCAT(COALESCE(preferred_name, first_name), ' ', COALESCE(last_name, ''))) as person_name
       FROM person WHERE id = ANY($1) AND collection_id = $2`,
      [[sourcePersonId, targetPersonId], collectionId],
    );

    const sourcePerson = personsResult.rows.find((r) => r.id.toString() === sourcePersonId);
    const targetPerson = personsResult.rows.find((r) => r.id.toString() === targetPersonId);

    if (!sourcePerson) {
      await client.query("ROLLBACK");
      return { success: false, message: "Source person not found" };
    }
    if (!targetPerson) {
      await client.query("ROLLBACK");
      return { success: false, message: "Target person not found" };
    }

    // Move all clusters from source person to target person
    await client.query(
      `UPDATE cluster SET person_id = $1, updated_at = NOW()
       WHERE person_id = $2 AND collection_id = $3`,
      [targetPerson.id, sourcePerson.id, collectionId],
    );

    // Migrate cannot-link constraints from source to target
    await client.query(
      `UPDATE cluster_person_cannot_link SET person_id = $1
       WHERE person_id = $2
       AND NOT EXISTS (
         SELECT 1 FROM cluster_person_cannot_link c2
         WHERE c2.cluster_id = cluster_person_cannot_link.cluster_id AND c2.person_id = $1
       )`,
      [targetPerson.id, sourcePerson.id],
    );
    await client.query(`DELETE FROM cluster_person_cannot_link WHERE person_id = $1`, [sourcePerson.id]);

    // Delete source person
    await client.query(`DELETE FROM person WHERE id = $1 AND collection_id = $2`, [sourcePerson.id, collectionId]);

    await client.query("COMMIT");
    return {
      success: true,
      message: `Merged "${sourcePerson.person_name}" into "${targetPerson.person_name}"`,
      personId: targetPerson.id,
      deletedPersonId: sourcePerson.id,
      deletedPersonName: sourcePerson.person_name,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to merge persons:", error);
    return { success: false, message: "Failed to merge persons" };
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
    SELECT pd.id, pd.face_path,
           pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.face_confidence as confidence, pd.cluster_id, pd.cluster_status,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           pd.photo_id, p.med_width, p.med_height,
           TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name
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
      SELECT pd.id, pd.face_path,
             pd.face_bbox_x, pd.face_bbox_y, pd.face_bbox_width, pd.face_bbox_height,
             pd.face_confidence as confidence, pd.cluster_id, pd.cluster_status, pd.cluster_confidence,
             pd.age_estimate, pd.gender, pd.gender_confidence,
             p.id as photo_id, p.med_width, p.med_height,
             c.face_count as cluster_face_count,
             TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
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
    SELECT id, face_path,
           face_bbox_x, face_bbox_y, face_bbox_width, face_bbox_height,
           confidence,
           cluster_id, cluster_status, cluster_confidence,
           age_estimate, gender, gender_confidence,
           photo_id, med_width, med_height,
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
    SELECT p.id, p.orig_path, p.med_path,
           p.med_width, p.med_height,
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

// ============================================================================
// Collection Management
// ============================================================================

export type UserCollection = {
  id: number;
  name: string;
  photo_count: number;
  person_id: number | null;
  person_name: string | null;
  avatar_detection_id: number | null;
  avatar_face_path: string | null;
};

/**
 * Get all collections a user is a member of, with linked person info for avatar.
 * Avatar is derived from the person's largest cluster's representative detection.
 */
export async function getUserCollections(userId: number): Promise<UserCollection[]> {
  const query = `
    SELECT
      c.id,
      c.name,
      COALESCE(pc.photo_count, 0)::int as photo_count,
      cm.person_id,
      CASE WHEN per.id IS NOT NULL
        THEN TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, '')))
        ELSE NULL
      END as person_name,
      pd.id as avatar_detection_id,
      pd.face_path as avatar_face_path
    FROM collection_member cm
    JOIN collection c ON cm.collection_id = c.id
    LEFT JOIN person per ON cm.person_id = per.id
    LEFT JOIN (
      SELECT collection_id, COUNT(*)::int as photo_count
      FROM photo
      GROUP BY collection_id
    ) pc ON c.id = pc.collection_id
    LEFT JOIN LATERAL (
      SELECT cl.representative_detection_id
      FROM cluster cl
      WHERE cl.person_id = per.id
        AND cl.representative_detection_id IS NOT NULL
        AND (cl.hidden = false OR cl.hidden IS NULL)
      ORDER BY cl.face_count DESC
      LIMIT 1
    ) rep ON per.id IS NOT NULL
    LEFT JOIN person_detection pd ON pd.id = rep.representative_detection_id
    WHERE cm.user_id = $1
    ORDER BY c.name
  `;

  const result = await pool.query(query, [userId]);
  return result.rows;
}

export type CollectionMemberInfo = {
  collection_id: number;
  collection_name: string;
  person_id: number | null;
  person_name: string | null;
  avatar_detection_id: number | null;
  avatar_face_path: string | null;
};

/**
 * Get collection member info for a specific user and collection.
 * Returns linked person info for avatar display.
 * Avatar is derived from the person's largest cluster's representative detection.
 */
export async function getCollectionMemberInfo(
  userId: number,
  collectionId: number,
): Promise<CollectionMemberInfo | null> {
  const query = `
    SELECT
      c.id as collection_id,
      c.name as collection_name,
      cm.person_id,
      CASE WHEN per.id IS NOT NULL
        THEN TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, '')))
        ELSE NULL
      END as person_name,
      pd.id as avatar_detection_id,
      pd.face_path as avatar_face_path
    FROM collection_member cm
    JOIN collection c ON cm.collection_id = c.id
    LEFT JOIN person per ON cm.person_id = per.id
    LEFT JOIN LATERAL (
      SELECT cl.representative_detection_id
      FROM cluster cl
      WHERE cl.person_id = per.id
        AND cl.representative_detection_id IS NOT NULL
        AND (cl.hidden = false OR cl.hidden IS NULL)
      ORDER BY cl.face_count DESC
      LIMIT 1
    ) rep ON per.id IS NOT NULL
    LEFT JOIN person_detection pd ON pd.id = rep.representative_detection_id
    WHERE cm.user_id = $1 AND cm.collection_id = $2
  `;

  const result = await pool.query(query, [userId, collectionId]);
  return result.rows[0] || null;
}

/**
 * Update user's default collection.
 */
export async function updateUserDefaultCollection(userId: number, collectionId: number): Promise<void> {
  // Verify user is member of this collection
  const memberCheck = await pool.query("SELECT 1 FROM collection_member WHERE user_id = $1 AND collection_id = $2", [
    userId,
    collectionId,
  ]);

  if (memberCheck.rows.length === 0) {
    throw new Error("User is not a member of this collection");
  }

  await pool.query("UPDATE app_user SET default_collection_id = $1 WHERE id = $2", [collectionId, userId]);
}

/**
 * Set the person linked to a user's collection membership.
 * Used for avatar display in that collection.
 */
export async function setCollectionMemberPerson(
  userId: number,
  collectionId: number,
  personId: number | null,
): Promise<void> {
  const result = await pool.query(
    "UPDATE collection_member SET person_id = $1 WHERE user_id = $2 AND collection_id = $3 RETURNING user_id",
    [personId, userId, collectionId],
  );

  if (result.rows.length === 0) {
    throw new Error("User is not a member of this collection");
  }
}

export type PersonForSelection = {
  id: number;
  first_name: string;
  last_name: string | null;
  person_name: string;
  detection_id: number | null;
  linked_user_id: number | null;
  linked_user_name: string | null;
};

/**
 * Get all persons in a collection with info about which user they're linked to.
 * Used for person selection modal.
 */
export async function getPersonsForCollection(collectionId: number): Promise<PersonForSelection[]> {
  const query = `
    SELECT
      per.id,
      per.first_name,
      per.last_name,
      TRIM(CONCAT(COALESCE(per.preferred_name, per.first_name), ' ', COALESCE(per.last_name, ''))) as person_name,
      COALESCE(
        per.representative_detection_id,
        (SELECT c.representative_detection_id
         FROM cluster c
         WHERE c.person_id = per.id
           AND c.representative_detection_id IS NOT NULL
           AND (c.hidden = false OR c.hidden IS NULL)
         ORDER BY c.face_count DESC
         LIMIT 1)
      ) as detection_id,
      cm.user_id as linked_user_id,
      CASE WHEN cm.user_id IS NOT NULL
        THEN TRIM(CONCAT(u.first_name, ' ', COALESCE(u.last_name, '')))
        ELSE NULL
      END as linked_user_name
    FROM person per
    LEFT JOIN collection_member cm ON cm.person_id = per.id AND cm.collection_id = per.collection_id
    LEFT JOIN app_user u ON cm.user_id = u.id
    WHERE per.collection_id = $1
      AND (per.hidden = false OR per.hidden IS NULL)
      AND COALESCE(per.auto_created, false) = false
    ORDER BY COALESCE(per.preferred_name, per.first_name), per.last_name, per.id
  `;

  const result = await pool.query(query, [collectionId]);
  return result.rows;
}

/**
 * Get face_path for a detection (used by face image API).
 */
export async function getDetectionFacePath(collectionId: number, detectionId: number): Promise<string | null> {
  const query = `
    SELECT pd.face_path
    FROM person_detection pd
    WHERE pd.id = $1 AND pd.collection_id = $2
  `;
  const result = await pool.query(query, [detectionId, collectionId]);
  return result.rows[0]?.face_path || null;
}

// --- Genealogy queries ---

export interface FamilyMember {
  person_id: number;
  display_name: string;
  relation: string;
  generation_offset: number;
  is_placeholder: boolean;
  detection_id: number | null;
}

export interface PersonParentRow {
  person_id: number;
  parent_id: number;
  parent_role: string;
  is_biological: boolean;
}

export interface PersonPartnershipRow {
  id: number;
  person1_id: number;
  person2_id: number;
  partnership_type: string;
  start_year: number | null;
  end_year: number | null;
  is_current: boolean;
}

export async function getFamilyTree(
  collectionId: number,
  personId: string,
  maxGenerations = 3,
): Promise<FamilyMember[]> {
  const result = await pool.query(
    `SELECT ft.person_id, ft.display_name, ft.relation, ft.generation_offset, ft.is_placeholder,
            COALESCE(p.representative_detection_id, (
              SELECT pd.id FROM person_detection pd
              JOIN cluster c ON c.id = pd.cluster_id
              WHERE c.person_id = ft.person_id AND c.collection_id = $1
              ORDER BY pd.face_confidence DESC NULLS LAST LIMIT 1
            )) AS detection_id
     FROM get_family_tree($2::int, $3::int, true) ft
     LEFT JOIN person p ON p.id = ft.person_id
     WHERE p.collection_id = $1 OR ft.is_placeholder = true`,
    [collectionId, personId, maxGenerations],
  );
  return result.rows;
}

export async function getPersonParents(collectionId: number, personId: string): Promise<PersonParentRow[]> {
  const result = await pool.query(
    `SELECT pp.person_id, pp.parent_id, pp.parent_role, pp.is_biological
     FROM person_parent pp
     JOIN person p ON p.id = pp.parent_id
     WHERE pp.person_id = $2 AND (p.collection_id = $1 OR p.is_placeholder = true)
     ORDER BY pp.parent_role, pp.parent_id`,
    [collectionId, personId],
  );
  return result.rows;
}

export async function getFamilyParentLinks(personIds: number[]): Promise<PersonParentRow[]> {
  if (personIds.length === 0) return [];
  const result = await pool.query(
    `SELECT DISTINCT pp.person_id, pp.parent_id, pp.parent_role, pp.is_biological
     FROM person_parent pp
     WHERE pp.person_id = ANY($1) AND pp.parent_id = ANY($1)`,
    [personIds],
  );
  return result.rows;
}

export async function getPersonPartnerships(_collectionId: number, personId: string): Promise<PersonPartnershipRow[]> {
  const result = await pool.query(
    `SELECT pp.id, pp.person1_id, pp.person2_id, pp.partnership_type, pp.start_year, pp.end_year, pp.is_current
     FROM person_partnership pp
     WHERE pp.person1_id = $1::int OR pp.person2_id = $1::int
     ORDER BY pp.is_current DESC, pp.start_year DESC NULLS LAST`,
    [personId],
  );
  return result.rows;
}

/** Fetch all partnerships involving any of the given person IDs. */
export async function getFamilyPartnerships(personIds: number[]): Promise<PersonPartnershipRow[]> {
  if (personIds.length === 0) return [];
  const result = await pool.query(
    `SELECT DISTINCT ON (pp.id) pp.id, pp.person1_id, pp.person2_id, pp.partnership_type, pp.start_year, pp.end_year, pp.is_current
     FROM person_partnership pp
     WHERE pp.person1_id = ANY($1::int[]) OR pp.person2_id = ANY($1::int[])
     ORDER BY pp.id`,
    [personIds],
  );
  return result.rows;
}

const VALID_PARENT_ROLES = new Set(["mother", "father", "parent"]);
function sanitizeParentRole(role: string): string {
  return VALID_PARENT_ROLES.has(role) ? role : "parent";
}

async function withTransaction<T>(fn: (client: PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try {
    await client.query("BEGIN");
    const result = await fn(client);
    await client.query("COMMIT");
    return result;
  } catch (e) {
    await client.query("ROLLBACK");
    throw e;
  } finally {
    client.release();
  }
}

export async function addPersonParent(
  _collectionId: number,
  personId: string,
  parentId: string,
  parentRole: string = "parent",
): Promise<{ success: boolean }> {
  const role = sanitizeParentRole(parentRole);
  return withTransaction(async (client) => {
    // Check existing parent count before inserting
    const existing = await client.query(
      `SELECT parent_id FROM person_parent WHERE person_id = $1::int`,
      [personId],
    );
    const existingParentIds = existing.rows.map((r: { parent_id: number }) => r.parent_id);

    await client.query(
      `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
       VALUES ($1::int, $2::int, $3, 'user')
       ON CONFLICT (person_id, parent_id) DO UPDATE SET parent_role = $3`,
      [personId, parentId, role],
    );

    // If this is the 2nd parent, auto-create a partnership between the two parents
    if (existingParentIds.length === 1 && !existingParentIds.includes(Number(parentId))) {
      const otherParentId = existingParentIds[0];
      const p1 = Math.min(Number(parentId), otherParentId);
      const p2 = Math.max(Number(parentId), otherParentId);
      await client.query(
        `INSERT INTO person_partnership (person1_id, person2_id, partnership_type, is_current)
         VALUES ($1, $2, 'partner', true)
         ON CONFLICT (person1_id, person2_id, COALESCE(start_year, 0)) DO NOTHING`,
        [p1, p2],
      );
    }

    // Also link this parent to personId's existing siblings (bulk insert)
    await client.query(
      `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
       SELECT DISTINCT p2.person_id, $2::int, $3, 'user'
       FROM person_parent p1
       JOIN person_parent p2 ON p1.parent_id = p2.parent_id
       WHERE p1.person_id = $1::int AND p2.person_id != $1::int
       ON CONFLICT (person_id, parent_id) DO NOTHING`,
      [personId, parentId, role],
    );

    await client.query(`SELECT refresh_genealogy_closures()`);
    return { success: true };
  });
}

export async function removePersonParent(personId: string, parentId: string): Promise<{ success: boolean }> {
  return withTransaction(async (client) => {
    await client.query(
      `DELETE FROM person_parent WHERE person_id = $1::int AND parent_id = $2::int`,
      [personId, parentId],
    );
    await client.query(`SELECT refresh_genealogy_closures()`);
    return { success: true };
  });
}

export async function addPersonPartnership(
  personId: string,
  partnerId: string,
  partnershipType: string = "partner",
): Promise<{ success: boolean }> {
  const p1 = Math.min(Number(personId), Number(partnerId));
  const p2 = Math.max(Number(personId), Number(partnerId));
  return withTransaction(async (client) => {
    await client.query(
      `INSERT INTO person_partnership (person1_id, person2_id, partnership_type, is_current)
       VALUES ($1, $2, $3, true)
       ON CONFLICT (person1_id, person2_id, COALESCE(start_year, 0)) DO UPDATE SET partnership_type = $3`,
      [p1, p2, partnershipType],
    );

    // Also link partner as parent to center person's existing children (bulk insert)
    await client.query(
      `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
       SELECT person_id, $2::int, 'parent', 'user'
       FROM person_parent WHERE parent_id = $1::int
       ON CONFLICT (person_id, parent_id) DO NOTHING`,
      [personId, partnerId],
    );

    await client.query(`SELECT refresh_genealogy_closures()`);
    return { success: true };
  });
}

export async function removePersonPartnership(personId: string, partnerId: string): Promise<{ success: boolean }> {
  const p1 = Math.min(Number(personId), Number(partnerId));
  const p2 = Math.max(Number(personId), Number(partnerId));
  return withTransaction(async (client) => {
    // Remove partner's parent links to center person's children (bulk delete)
    await client.query(
      `DELETE FROM person_parent
       WHERE parent_id = $2::int
         AND person_id IN (SELECT person_id FROM person_parent WHERE parent_id = $1::int)`,
      [personId, partnerId],
    );

    await client.query(
      `DELETE FROM person_partnership WHERE person1_id = $1 AND person2_id = $2`,
      [p1, p2],
    );
    await client.query(`SELECT refresh_genealogy_closures()`);
    return { success: true };
  });
}

export async function addPersonChild(
  _collectionId: number,
  parentId: string,
  childId: string,
  parentRole: string = "parent",
): Promise<{ success: boolean }> {
  const role = sanitizeParentRole(parentRole);
  return withTransaction(async (client) => {
    // Add parent→child link
    await client.query(
      `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
       VALUES ($1::int, $2::int, $3, 'user')
       ON CONFLICT (person_id, parent_id) DO UPDATE SET parent_role = $3`,
      [childId, parentId, role],
    );

    // Also link child to parent's current partner(s) as parent (bulk insert)
    await client.query(
      `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
       SELECT $1::int,
              CASE WHEN person1_id = $2::int THEN person2_id ELSE person1_id END,
              $3, 'user'
       FROM person_partnership
       WHERE person1_id = $2::int OR person2_id = $2::int
       ON CONFLICT (person_id, parent_id) DO NOTHING`,
      [childId, parentId, role],
    );

    await client.query(`SELECT refresh_genealogy_closures()`);
    return { success: true };
  });
}

export async function createPlaceholderPerson(
  collectionId: number,
  name: string | null,
  gender: string = "U",
  description: string | null = null,
): Promise<{ id: number }> {
  const result = await pool.query(
    `INSERT INTO person (collection_id, first_name, is_placeholder, placeholder_description, gender, auto_created)
     VALUES ($1, $2, true, $3, $4, false)
     RETURNING id`,
    [collectionId, name, description, gender],
  );
  return { id: result.rows[0].id };
}

export async function addPersonSibling(
  _collectionId: number,
  personId: string,
  siblingId: string,
): Promise<{ success: boolean }> {
  return withTransaction(async (client) => {
    // Check if personId has parents
    const existingParents = await client.query(
      `SELECT count(*) as cnt FROM person_parent WHERE person_id = $1::int`,
      [personId],
    );
    const personHasParents = Number(existingParents.rows[0].cnt) > 0;

    if (personHasParents) {
      // Link siblingId to all of personId's existing parents (bulk insert)
      await client.query(
        `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
         SELECT $2::int, parent_id, parent_role, 'user'
         FROM person_parent WHERE person_id = $1::int
         ON CONFLICT (person_id, parent_id) DO NOTHING`,
        [personId, siblingId],
      );
    } else {
      // Check if siblingId has parents — copy them to personId
      const siblingHasParents = await client.query(
        `SELECT count(*) as cnt FROM person_parent WHERE person_id = $1::int`,
        [siblingId],
      );
      if (Number(siblingHasParents.rows[0].cnt) > 0) {
        await client.query(
          `INSERT INTO person_parent (person_id, parent_id, parent_role, source)
           SELECT $2::int, parent_id, parent_role, 'user'
           FROM person_parent WHERE person_id = $1::int
           ON CONFLICT (person_id, parent_id) DO NOTHING`,
          [siblingId, personId],
        );
      }
      // If neither has parents, the sibling link has no effect — the user
      // needs to add parents first for the relationship to be visible.
    }

    await client.query(`SELECT refresh_genealogy_closures()`);
    return { success: true };
  });
}

export async function removePersonSibling(
  personId: string,
  siblingId: string,
): Promise<{ success: boolean }> {
  return withTransaction(async (client) => {
    // Remove sibling's links to shared parents (bulk delete)
    await client.query(
      `DELETE FROM person_parent
       WHERE person_id = $2::int
         AND parent_id IN (
           SELECT p1.parent_id FROM person_parent p1
           JOIN person_parent p2 ON p1.parent_id = p2.parent_id
           WHERE p1.person_id = $1::int AND p2.person_id = $2::int
         )`,
      [personId, siblingId],
    );
    await client.query(`SELECT refresh_genealogy_closures()`);
    return { success: true };
  });
}
