import fs from "node:fs";
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

// Initialize database on first connection
let dbInitialized = false;

async function initDatabase() {
  if (dbInitialized) return;

  try {
    // Check if tables exist first
    const result = await pool.query(`
      SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'photo'
      );
    `);

    if (!result.rows[0].exists) {
      // Try to find and run schema.sql
      const schemaPath = path.join(process.cwd(), "..", "schema.sql");
      if (fs.existsSync(schemaPath)) {
        const schema = fs.readFileSync(schemaPath, "utf-8");
        await pool.query(schema);
        console.log("Database schema initialized");
      }
    }

    dbInitialized = true;
  } catch (error) {
    console.error("Failed to initialize database:", error);
  }
}

// Query functions that match the Python PhotoQueries class

export async function getYearsWithPhotos() {
  await initDatabase();

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
        EXTRACT(YEAR FROM m.captured_at)::int as year,
        COUNT(*)::int as photo_count
      FROM metadata m
      WHERE m.captured_at IS NOT NULL
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
        WHERE EXTRACT(YEAR FROM m.captured_at) = y.year
        LIMIT 4
      ) ranked
    ) s ON true
    ORDER BY y.year DESC
  `;

  const result = await pool.query(query);

  // Add backward compatibility field
  return result.rows.map((row) => ({
    ...row,
    sample_photo_id: row.sample_photo_ids?.[0] || null,
  }));
}

export async function getMonthsInYear(year: number) {
  await initDatabase();

  // Single query with LATERAL join for sample photos
  // Uses daily-varying deterministic selection like getYearsWithPhotos
  const query = `
    SELECT
      mo.month,
      mo.photo_count,
      COALESCE(s.sample_ids, ARRAY[]::int[]) as sample_photo_ids
    FROM (
      SELECT
        EXTRACT(MONTH FROM m.captured_at)::int as month,
        COUNT(*)::int as photo_count
      FROM metadata m
      WHERE EXTRACT(YEAR FROM m.captured_at) = $1
        AND m.captured_at IS NOT NULL
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
        WHERE EXTRACT(YEAR FROM m.captured_at) = $1
          AND EXTRACT(MONTH FROM m.captured_at) = mo.month
        LIMIT 4
      ) ranked
    ) s ON true
    ORDER BY mo.month
  `;

  const result = await pool.query(query, [year]);

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

export async function getPhotosByMonth(year: number, month: number, limit = 100, offset = 0) {
  await initDatabase();

  const query = `
    SELECT p.id, p.filename, p.normalized_path,
           m.captured_at, m.latitude, m.longitude,
           la.description, la.emotional_tone,
           la.objects, la.people_count
    FROM photo p
    JOIN metadata m ON p.id = m.photo_id
    LEFT JOIN llm_analysis la ON p.id = la.photo_id
    WHERE EXTRACT(YEAR FROM m.captured_at) = $1
      AND EXTRACT(MONTH FROM m.captured_at) = $2
    ORDER BY m.captured_at, p.filename
    LIMIT $3 OFFSET $4
  `;

  const result = await pool.query(query, [year, month, limit, offset]);

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

export async function getPhotoCountByMonth(year: number, month: number) {
  await initDatabase();

  const query = `
    SELECT COUNT(*) as count
    FROM metadata m
    WHERE EXTRACT(YEAR FROM m.captured_at) = $1
      AND EXTRACT(MONTH FROM m.captured_at) = $2
  `;

  const result = await pool.query(query, [year, month]);
  return parseInt(result.rows[0].count, 10);
}

export async function getPhotoDetails(photoId: number) {
  await initDatabase();

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
    WHERE p.id = $1
  `;

  const result = await pool.query(query, [photoId]);
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

  // Set image dimensions from metadata table
  photo.image_width = photo.normalized_width || null;
  photo.image_height = photo.normalized_height || null;

  return photo;
}

export async function getPhotoById(photoId: number) {
  await initDatabase();

  const query = `
    SELECT id, filename, normalized_path
    FROM photo
    WHERE id = $1
  `;

  const result = await pool.query(query, [photoId]);
  return result.rows[0] || null;
}

export async function getClusters(limit = 50, offset = 0) {
  await initDatabase();

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
    ORDER BY c.face_count DESC, c.id
    LIMIT $1 OFFSET $2
  `;

  const result = await pool.query(query, [limit, offset]);
  return result.rows;
}

export async function getClustersCount() {
  await initDatabase();

  const query = `
    SELECT COUNT(*) as count
    FROM cluster
    WHERE face_count > 0 AND (hidden = false OR hidden IS NULL)
  `;

  const result = await pool.query(query);
  return parseInt(result.rows[0].count, 10);
}

export async function getHiddenClusters(limit = 50, offset = 0) {
  await initDatabase();

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
    ORDER BY c.face_count DESC, c.id
    LIMIT $1 OFFSET $2
  `;

  const result = await pool.query(query, [limit, offset]);
  return result.rows;
}

export async function getHiddenClustersCount() {
  await initDatabase();

  const query = `
    SELECT COUNT(*) as count
    FROM cluster
    WHERE face_count > 0 AND hidden = true
  `;

  const result = await pool.query(query);
  return parseInt(result.rows[0].count, 10);
}

export async function setClusterHidden(clusterId: string, hidden: boolean) {
  await initDatabase();

  const query = `
    UPDATE cluster
    SET hidden = $1, updated_at = NOW()
    WHERE id = $2
    RETURNING id
  `;

  const result = await pool.query(query, [hidden, clusterId]);
  return {
    success: result.rows.length > 0,
    message: hidden ? "Cluster hidden" : "Cluster unhidden",
  };
}

export async function setClusterPersonName(clusterId: string, firstName: string, lastName?: string) {
  await initDatabase();

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Check if cluster already has a person_id
    const clusterResult = await client.query("SELECT person_id FROM cluster WHERE id = $1", [clusterId]);
    if (clusterResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Cluster not found" };
    }

    const existingPersonId = clusterResult.rows[0].person_id;

    if (existingPersonId) {
      // Update existing person's name
      await client.query("UPDATE person SET first_name = $1, last_name = $2, updated_at = NOW() WHERE id = $3", [
        firstName,
        lastName || null,
        existingPersonId,
      ]);
    } else {
      // Create new person and link to cluster
      const personResult = await client.query(
        "INSERT INTO person (first_name, last_name) VALUES ($1, $2) RETURNING id",
        [firstName, lastName || null],
      );
      const newPersonId = personResult.rows[0].id;
      await client.query("UPDATE cluster SET person_id = $1, updated_at = NOW() WHERE id = $2", [
        newPersonId,
        clusterId,
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

export async function deleteCluster(clusterId: string) {
  await initDatabase();

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
      WHERE cluster_id = $1
      RETURNING id
    `;
    const detectionsResult = await client.query(updateDetectionsQuery, [clusterId]);
    const detectionsRemoved = detectionsResult.rowCount || 0;

    // Delete the cluster
    const deleteQuery = `
      DELETE FROM cluster
      WHERE id = $1
      RETURNING id
    `;
    const deleteResult = await client.query(deleteQuery, [clusterId]);

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

export async function getClusterDetails(clusterId: string) {
  await initDatabase();

  const query = `
    SELECT c.id, c.face_count, c.representative_detection_id,
           c.hidden,
           per.first_name, per.last_name,
           per.gender as person_gender, per.gender_confidence as person_gender_confidence,
           per.estimated_birth_year, per.birth_year_stddev,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    LEFT JOIN person per ON c.person_id = per.id
    WHERE c.id = $1
  `;

  const result = await pool.query(query, [clusterId]);
  return result.rows[0] || null;
}

export async function getClusterFaces(clusterId: string, limit = 24, offset = 0) {
  await initDatabase();

  const query = `
    SELECT pd.id, pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.cluster_confidence, pd.photo_id,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           p.normalized_path, p.filename, p.normalized_width, p.normalized_height
    FROM person_detection pd
    JOIN photo p ON pd.photo_id = p.id
    WHERE pd.cluster_id = $1
    ORDER BY pd.cluster_confidence DESC, pd.id
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [clusterId, limit, offset]);
  return result.rows;
}

export async function getClusterFacesCount(clusterId: string) {
  await initDatabase();

  const query = `
    SELECT COUNT(*) as count
    FROM person_detection
    WHERE cluster_id = $1
  `;

  const result = await pool.query(query, [clusterId]);
  return parseInt(result.rows[0].count, 10);
}

// Constraint management functions

export async function addCannotLink(detectionId1: number, detectionId2: number) {
  await initDatabase();

  // Canonical ordering to prevent duplicates
  const [id1, id2] = detectionId1 < detectionId2 ? [detectionId1, detectionId2] : [detectionId2, detectionId1];

  const query = `
    INSERT INTO cannot_link (detection_id_1, detection_id_2, created_by)
    VALUES ($1, $2, 'web')
    ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING
    RETURNING id
  `;

  const result = await pool.query(query, [id1, id2]);
  return result.rows[0]?.id || null;
}

export async function getCannotLinksForCluster(clusterId: string) {
  await initDatabase();

  // Get all cannot-link pairs where both detections are in this cluster
  const query = `
    SELECT cl.id, cl.detection_id_1, cl.detection_id_2, cl.created_at
    FROM cannot_link cl
    JOIN person_detection pd1 ON cl.detection_id_1 = pd1.id
    JOIN person_detection pd2 ON cl.detection_id_2 = pd2.id
    WHERE pd1.cluster_id = $1 AND pd2.cluster_id = $1
    ORDER BY cl.created_at DESC
  `;

  const result = await pool.query(query, [clusterId]);
  return result.rows;
}

export async function removeCannotLink(cannotLinkId: number) {
  await initDatabase();

  const query = `
    DELETE FROM cannot_link
    WHERE id = $1
    RETURNING id
  `;

  const result = await pool.query(query, [cannotLinkId]);
  return result.rows[0]?.id || null;
}

export async function setClusterRepresentative(clusterId: string, detectionId: number) {
  await initDatabase();

  // Verify detection belongs to this cluster
  const verifyQuery = `
    SELECT cluster_id FROM person_detection WHERE id = $1
  `;
  const verifyResult = await pool.query(verifyQuery, [detectionId]);
  if (verifyResult.rows.length === 0 || verifyResult.rows[0].cluster_id?.toString() !== clusterId) {
    return { success: false, message: "Detection does not belong to this cluster" };
  }

  // Update the representative detection
  const updateQuery = `
    UPDATE cluster
    SET representative_detection_id = $1,
        updated_at = NOW()
    WHERE id = $2
    RETURNING id
  `;

  const result = await pool.query(updateQuery, [detectionId, clusterId]);
  return {
    success: result.rows.length > 0,
    message: result.rows.length > 0 ? "Representative photo updated" : "Failed to update",
  };
}

export async function dissociateFacesFromCluster(
  clusterId: string,
  detectionIds: number[],
  similarityThreshold = 0.85,
) {
  await initDatabase();

  if (detectionIds.length === 0) {
    return { success: false, message: "No detections selected", removedCount: 0 };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

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
      WHERE id = ANY($1) AND cluster_id = $2
      RETURNING id
    `;
    const removeResult = await client.query(removeQuery, [allDetectionsToRemove, clusterId]);
    const removedCount = removeResult.rowCount || 0;

    // Step 4: Update cluster face count
    await client.query(
      `UPDATE cluster SET face_count = GREATEST(0, face_count - $1), updated_at = NOW() WHERE id = $2`,
      [removedCount, clusterId],
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
          `INSERT INTO cannot_link (detection_id_1, detection_id_2, created_by)
           VALUES ($1, $2, 'web')
           ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING`,
          [id1, id2],
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

export async function searchClusters(query: string, excludeClusterId?: string, limit = 20) {
  await initDatabase();

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

  const result = await pool.query(searchQuery, [query || null, excludeClusterId || null, limit]);
  return result.rows;
}

export async function mergeClusters(sourceClusterId: string, targetClusterId: string) {
  await initDatabase();

  if (sourceClusterId === targetClusterId) {
    return { success: false, message: "Cannot merge a cluster with itself" };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Get source cluster info
    const sourceQuery = `SELECT id, face_count, person_id FROM cluster WHERE id = $1`;
    const sourceResult = await client.query(sourceQuery, [sourceClusterId]);
    if (sourceResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Source cluster not found" };
    }

    // Get target cluster info
    const targetQuery = `SELECT id, face_count, person_id FROM cluster WHERE id = $1`;
    const targetResult = await client.query(targetQuery, [targetClusterId]);
    if (targetResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Target cluster not found" };
    }

    // Move all detections from source to target cluster
    const moveDetectionsQuery = `
      UPDATE person_detection
      SET cluster_id = $1,
          cluster_status = 'manual'
      WHERE cluster_id = $2
      RETURNING id
    `;
    const moveResult = await client.query(moveDetectionsQuery, [targetClusterId, sourceClusterId]);
    const movedCount = moveResult.rowCount || 0;

    // Update target cluster face count
    await client.query(`UPDATE cluster SET face_count = face_count + $1, updated_at = NOW() WHERE id = $2`, [
      movedCount,
      targetClusterId,
    ]);

    // Recompute target cluster centroid
    const centroidQuery = `
      UPDATE cluster
      SET centroid = (
        SELECT AVG(fe.embedding)::vector(512)
        FROM person_detection pd
        JOIN face_embedding fe ON pd.id = fe.person_detection_id
        WHERE pd.cluster_id = $1
      )
      WHERE id = $1
    `;
    await client.query(centroidQuery, [targetClusterId]);

    // Delete the source cluster
    await client.query(`DELETE FROM cluster WHERE id = $1`, [sourceClusterId]);

    await client.query("COMMIT");

    return {
      success: true,
      message: `Merged ${movedCount} detections into cluster ${targetClusterId}`,
      movedCount,
      targetClusterId,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to merge clusters:", error);
    return { success: false, message: "Failed to merge clusters" };
  } finally {
    client.release();
  }
}

export async function addFacesToCluster(clusterId: string, detectionIds: number[]) {
  await initDatabase();

  if (detectionIds.length === 0) {
    return { success: false, message: "No detections selected", addedCount: 0 };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Verify cluster exists
    const clusterQuery = `SELECT id, face_count FROM cluster WHERE id = $1`;
    const clusterResult = await client.query(clusterQuery, [clusterId]);
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
      WHERE id = ANY($2) AND cluster_id IS NULL
      RETURNING id
    `;
    const addResult = await client.query(addDetectionsQuery, [clusterId, detectionIds]);
    const addedCount = addResult.rowCount || 0;

    if (addedCount === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "No unclustered detections to add", addedCount: 0 };
    }

    // Update cluster face count
    await client.query(`UPDATE cluster SET face_count = face_count + $1, updated_at = NOW() WHERE id = $2`, [
      addedCount,
      clusterId,
    ]);

    // Recompute cluster centroid
    const centroidQuery = `
      UPDATE cluster
      SET centroid = (
        SELECT AVG(fe.embedding)::vector(512)
        FROM person_detection pd
        JOIN face_embedding fe ON pd.id = fe.person_detection_id
        WHERE pd.cluster_id = $1
      )
      WHERE id = $1
    `;
    await client.query(centroidQuery, [clusterId]);

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

export async function getFaceDetails(detectionId: number) {
  await initDatabase();

  const query = `
    SELECT pd.id, pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.face_confidence as confidence, pd.cluster_id, pd.cluster_status,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           pd.photo_id, p.normalized_path, p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM person_detection pd
    JOIN photo p ON pd.photo_id = p.id
    LEFT JOIN person per ON pd.person_id = per.id
    WHERE pd.id = $1
  `;

  const result = await pool.query(query, [detectionId]);
  return result.rows[0] || null;
}

export async function createClusterFromFaces(detectionIds: number[]) {
  await initDatabase();

  if (detectionIds.length === 0) {
    return { success: false, message: "No detections selected", clusterId: null };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Step 1: Create a new cluster
    const createClusterQuery = `
      INSERT INTO cluster (face_count, created_at, updated_at)
      VALUES ($1, NOW(), NOW())
      RETURNING id
    `;
    const clusterResult = await client.query(createClusterQuery, [detectionIds.length]);
    const clusterId = clusterResult.rows[0].id;

    // Step 2: Assign all detections to the new cluster
    const assignDetectionsQuery = `
      UPDATE person_detection
      SET cluster_id = $1,
          cluster_status = 'manual',
          cluster_confidence = 1.0
      WHERE id = ANY($2) AND cluster_id IS NULL
      RETURNING id
    `;
    const assignResult = await client.query(assignDetectionsQuery, [clusterId, detectionIds]);
    const assignedCount = assignResult.rowCount || 0;

    // Step 3: Update the cluster face count to match actual assigned
    await client.query(`UPDATE cluster SET face_count = $1 WHERE id = $2`, [assignedCount, clusterId]);

    // Step 4: Set the first detection as the representative
    if (assignedCount > 0) {
      const firstDetectionId = detectionIds[0];
      await client.query(
        `UPDATE cluster SET representative_detection_id = $1, medoid_detection_id = $1 WHERE id = $2`,
        [firstDetectionId, clusterId],
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
      WHERE id = $2
    `;
    await client.query(centroidQuery, [detectionIds, clusterId]);

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

export async function getSimilarFaces(detectionId: number, limit = 12, similarityThreshold = 0.7) {
  await initDatabase();

  // Convert similarity threshold to distance threshold (cosine distance = 1 - similarity)
  const distanceThreshold = 1 - similarityThreshold;

  const query = `
    WITH target_embedding AS (
      SELECT embedding FROM face_embedding WHERE person_detection_id = $1
    ),
    target_detection AS (
      SELECT photo_id FROM person_detection WHERE id = $1
    )
    SELECT pd.id, pd.face_bbox_x as bbox_x, pd.face_bbox_y as bbox_y,
           pd.face_bbox_width as bbox_width, pd.face_bbox_height as bbox_height,
           pd.face_confidence as confidence, pd.cluster_id, pd.cluster_status, pd.cluster_confidence,
           pd.age_estimate, pd.gender, pd.gender_confidence,
           p.id as photo_id, p.normalized_path, p.normalized_width, p.normalized_height,
           c.face_count as cluster_face_count,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
           1 - (fe.embedding <=> te.embedding) as similarity
    FROM face_embedding fe
    CROSS JOIN target_embedding te
    JOIN person_detection pd ON fe.person_detection_id = pd.id
    JOIN photo p ON pd.photo_id = p.id
    LEFT JOIN cluster c ON pd.cluster_id = c.id
    LEFT JOIN person per ON c.person_id = per.id
    WHERE fe.person_detection_id != $1
      AND pd.photo_id != (SELECT photo_id FROM target_detection)
      AND (fe.embedding <=> te.embedding) < $2
    ORDER BY (fe.embedding <=> te.embedding) ASC
    LIMIT $3
  `;

  const result = await pool.query(query, [detectionId, distanceThreshold, limit]);
  return result.rows;
}

export async function dissociateFaceWithConfidenceCutoff(clusterId: string, detectionId: number) {
  await initDatabase();

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Step 1: Get the confidence of the selected detection
    const confidenceQuery = `
      SELECT cluster_confidence FROM person_detection WHERE id = $1 AND cluster_id = $2
    `;
    const confidenceResult = await client.query(confidenceQuery, [detectionId, clusterId]);

    if (confidenceResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Detection not found in cluster", constrainedCount: 0, cutoffCount: 0 };
    }

    const cutoffConfidence = parseFloat(confidenceResult.rows[0].cluster_confidence);

    // Step 2: Find all detections with lower confidence (these will be unconstrained)
    const lowerConfidenceQuery = `
      SELECT id FROM person_detection
      WHERE cluster_id = $1 AND cluster_confidence < $2 AND id != $3
    `;
    const lowerResult = await client.query(lowerConfidenceQuery, [clusterId, cutoffConfidence, detectionId]);
    const lowerConfidenceDetectionIds = lowerResult.rows.map((r) => parseInt(r.id, 10));

    // Step 3: Get a remaining detection for the cannot-link constraint
    const allDetectionsToRemove = [detectionId, ...lowerConfidenceDetectionIds];
    const remainingDetectionQuery = `
      SELECT id FROM person_detection
      WHERE cluster_id = $1 AND id != ALL($2)
      LIMIT 1
    `;
    const remainingResult = await client.query(remainingDetectionQuery, [clusterId, allDetectionsToRemove]);

    // Step 4: Remove the selected detection (constrained - cannot rejoin)
    await client.query(
      `UPDATE person_detection
       SET cluster_id = NULL,
           cluster_status = 'unassigned',
           cluster_confidence = 0,
           unassigned_since = NOW()
       WHERE id = $1`,
      [detectionId],
    );

    // Step 5: Create cannot-link for the selected detection only
    if (remainingResult.rows.length > 0) {
      const remainingDetectionId = parseInt(remainingResult.rows[0].id, 10);
      const [id1, id2] =
        detectionId < remainingDetectionId ? [detectionId, remainingDetectionId] : [remainingDetectionId, detectionId];

      await client.query(
        `INSERT INTO cannot_link (detection_id_1, detection_id_2, created_by)
         VALUES ($1, $2, 'web')
         ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING`,
        [id1, id2],
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
         WHERE id = ANY($1)
         RETURNING id`,
        [lowerConfidenceDetectionIds],
      );
      cutoffCount = cutoffResult.rowCount || 0;
    }

    // Step 7: Update cluster face count
    const totalRemoved = 1 + cutoffCount;
    await client.query(
      `UPDATE cluster SET face_count = GREATEST(0, face_count - $1), updated_at = NOW() WHERE id = $2`,
      [totalRemoved, clusterId],
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

export async function removeFaceFromClusterWithConstraint(detectionId: number) {
  await initDatabase();

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Get the detection's current cluster
    const detectionQuery = `SELECT cluster_id FROM person_detection WHERE id = $1`;
    const detectionResult = await client.query(detectionQuery, [detectionId]);

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
      WHERE cluster_id = $1 AND id != $2
      LIMIT 1
    `;
    const remainingResult = await client.query(remainingDetectionQuery, [clusterId, detectionId]);

    // Remove detection from cluster
    await client.query(
      `UPDATE person_detection
       SET cluster_id = NULL,
           cluster_status = 'unassigned',
           cluster_confidence = 0,
           unassigned_since = NOW()
       WHERE id = $1`,
      [detectionId],
    );

    // Update cluster face count
    await client.query(
      `UPDATE cluster SET face_count = GREATEST(0, face_count - 1), updated_at = NOW() WHERE id = $1`,
      [clusterId],
    );

    // Create cannot-link constraint if there's a remaining detection
    if (remainingResult.rows.length > 0) {
      const remainingDetectionId = parseInt(remainingResult.rows[0].id, 10);
      const [id1, id2] =
        detectionId < remainingDetectionId ? [detectionId, remainingDetectionId] : [remainingDetectionId, detectionId];

      await client.query(
        `INSERT INTO cannot_link (detection_id_1, detection_id_2, created_by)
         VALUES ($1, $2, 'web')
         ON CONFLICT (detection_id_1, detection_id_2) DO NOTHING`,
        [id1, id2],
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
