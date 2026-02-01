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

  const query = `
    SELECT EXTRACT(YEAR FROM m.captured_at)::int as year,
           COUNT(*)::int as photo_count,
           MIN(p.id) as sample_photo_id
    FROM metadata m
    JOIN photo p ON p.id = m.photo_id
    WHERE m.captured_at IS NOT NULL
    GROUP BY year
    ORDER BY year DESC
  `;

  const result = await pool.query(query);
  return result.rows;
}

export async function getMonthsInYear(year: number) {
  await initDatabase();

  const query = `
    SELECT EXTRACT(MONTH FROM m.captured_at)::int as month,
           COUNT(*)::int as photo_count
    FROM metadata m
    WHERE EXTRACT(YEAR FROM m.captured_at) = $1
      AND m.captured_at IS NOT NULL
    GROUP BY month
    ORDER BY month
  `;

  const result = await pool.query(query, [year]);
  const months = result.rows;

  // Get sample photo IDs for each month
  for (const monthData of months) {
    const sampleQuery = `
      SELECT p.id
      FROM photo p
      JOIN metadata m ON p.id = m.photo_id
      WHERE EXTRACT(YEAR FROM m.captured_at) = $1
        AND EXTRACT(MONTH FROM m.captured_at) = $2
      ORDER BY m.captured_at
      LIMIT 4
    `;
    const sampleResult = await pool.query(sampleQuery, [year, monthData.month]);
    monthData.sample_photo_ids = sampleResult.rows.map((row) => row.id);
  }

  // Add month names
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

  return months.map((m) => ({
    ...m,
    month_name: monthNames[m.month],
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

  // Get faces for this photo with match candidates
  const facesQuery = `
    SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           f.confidence, f.person_id,
           TRIM(CONCAT(p.first_name, ' ', COALESCE(p.last_name, ''))) as person_name,
           f.cluster_id, f.cluster_status, f.cluster_confidence
    FROM face f
    LEFT JOIN person p ON f.person_id = p.id
    WHERE f.photo_id = $1
    ORDER BY f.confidence DESC
  `;

  const facesResult = await pool.query(facesQuery, [photoId]);

  // Get face match candidates for each face
  for (const face of facesResult.rows) {
    const candidatesQuery = `
      SELECT fmc.cluster_id, fmc.similarity, fmc.status,
             c.person_id,
             TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
             c.face_count
      FROM face_match_candidate fmc
      LEFT JOIN "cluster" c ON fmc.cluster_id = c.id
      LEFT JOIN person per ON c.person_id = per.id
      WHERE fmc.face_id = $1 AND fmc.status = 'pending'
      ORDER BY fmc.similarity DESC
      LIMIT 3
    `;

    const candidatesResult = await pool.query(candidatesQuery, [face.id]);
    face.match_candidates = candidatesResult.rows;
  }

  photo.faces = facesResult.rows;
  photo.face_count = facesResult.rows.length;

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
    SELECT c.id, c.face_count, c.representative_face_id,
           f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           p.id as photo_id, p.normalized_path, p.filename,
           p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    LEFT JOIN face f ON c.representative_face_id = f.id
    LEFT JOIN photo p ON f.photo_id = p.id
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
    SELECT c.id, c.face_count, c.representative_face_id,
           f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           p.id as photo_id, p.normalized_path, p.filename,
           p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    LEFT JOIN face f ON c.representative_face_id = f.id
    LEFT JOIN photo p ON f.photo_id = p.id
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
      await client.query(
        "UPDATE person SET first_name = $1, last_name = $2, updated_at = NOW() WHERE id = $3",
        [firstName, lastName || null, existingPersonId],
      );
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

    // Move all faces back to unassigned pool
    const updateFacesQuery = `
      UPDATE face
      SET cluster_id = NULL,
          cluster_status = 'unassigned',
          cluster_confidence = 0,
          unassigned_since = NOW()
      WHERE cluster_id = $1
      RETURNING id
    `;
    const facesResult = await client.query(updateFacesQuery, [clusterId]);
    const facesRemoved = facesResult.rowCount || 0;

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
        message: `Cluster deleted, ${facesRemoved} face${facesRemoved !== 1 ? "s" : ""} moved to unassigned`,
        facesRemoved,
      };
    }
    return { success: false, message: "Cluster not found", facesRemoved: 0 };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to delete cluster:", error);
    return { success: false, message: "Failed to delete cluster", facesRemoved: 0 };
  } finally {
    client.release();
  }
}

export async function getClusterDetails(clusterId: string) {
  await initDatabase();

  const query = `
    SELECT c.id, c.face_count, c.representative_face_id,
           c.hidden,
           per.first_name, per.last_name,
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
    SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           f.cluster_confidence, f.photo_id,
           p.normalized_path, p.filename, p.normalized_width, p.normalized_height
    FROM face f
    JOIN photo p ON f.photo_id = p.id
    WHERE f.cluster_id = $1
    ORDER BY f.cluster_confidence DESC, f.id
    LIMIT $2 OFFSET $3
  `;

  const result = await pool.query(query, [clusterId, limit, offset]);
  return result.rows;
}

export async function getClusterFacesCount(clusterId: string) {
  await initDatabase();

  const query = `
    SELECT COUNT(*) as count
    FROM face
    WHERE cluster_id = $1
  `;

  const result = await pool.query(query, [clusterId]);
  return parseInt(result.rows[0].count, 10);
}

// Constraint management functions

export async function addCannotLink(faceId1: number, faceId2: number) {
  await initDatabase();

  // Canonical ordering to prevent duplicates
  const [id1, id2] = faceId1 < faceId2 ? [faceId1, faceId2] : [faceId2, faceId1];

  const query = `
    INSERT INTO cannot_link (face_id_1, face_id_2, created_by)
    VALUES ($1, $2, 'web')
    ON CONFLICT (face_id_1, face_id_2) DO NOTHING
    RETURNING id
  `;

  const result = await pool.query(query, [id1, id2]);
  return result.rows[0]?.id || null;
}

export async function getCannotLinksForCluster(clusterId: string) {
  await initDatabase();

  // Get all cannot-link pairs where both faces are in this cluster
  const query = `
    SELECT cl.id, cl.face_id_1, cl.face_id_2, cl.created_at
    FROM cannot_link cl
    JOIN face f1 ON cl.face_id_1 = f1.id
    JOIN face f2 ON cl.face_id_2 = f2.id
    WHERE f1.cluster_id = $1 AND f2.cluster_id = $1
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

export async function setClusterRepresentative(clusterId: string, faceId: number) {
  await initDatabase();

  // Verify face belongs to this cluster
  const verifyQuery = `
    SELECT cluster_id FROM face WHERE id = $1
  `;
  const verifyResult = await pool.query(verifyQuery, [faceId]);
  if (verifyResult.rows.length === 0 || verifyResult.rows[0].cluster_id?.toString() !== clusterId) {
    return { success: false, message: "Face does not belong to this cluster" };
  }

  // Update the representative face
  const updateQuery = `
    UPDATE cluster
    SET representative_face_id = $1,
        updated_at = NOW()
    WHERE id = $2
    RETURNING id
  `;

  const result = await pool.query(updateQuery, [faceId, clusterId]);
  return {
    success: result.rows.length > 0,
    message: result.rows.length > 0 ? "Representative photo updated" : "Failed to update",
  };
}

export async function dissociateFacesFromCluster(clusterId: string, faceIds: number[], similarityThreshold = 0.85) {
  await initDatabase();

  if (faceIds.length === 0) {
    return { success: false, message: "No faces selected", removedCount: 0 };
  }

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Step 1: Find all faces in cluster similar to selected faces above threshold
    // Using cosine distance: 1 - cosine_similarity, so threshold becomes 1 - similarityThreshold
    const distanceThreshold = 1 - similarityThreshold;

    const similarFacesQuery = `
      WITH selected_embeddings AS (
        SELECT fe.face_id, fe.embedding
        FROM face_embedding fe
        JOIN face f ON fe.face_id = f.id
        WHERE fe.face_id = ANY($1) AND f.cluster_id = $2
      ),
      cluster_faces AS (
        SELECT fe.face_id, fe.embedding
        FROM face_embedding fe
        JOIN face f ON fe.face_id = f.id
        WHERE f.cluster_id = $2
      )
      SELECT DISTINCT cf.face_id
      FROM cluster_faces cf
      WHERE cf.face_id = ANY($1)
         OR EXISTS (
           SELECT 1 FROM selected_embeddings se
           WHERE (cf.embedding <=> se.embedding) < $3
         )
    `;

    const similarResult = await client.query(similarFacesQuery, [faceIds, clusterId, distanceThreshold]);

    // Parse as integers since PostgreSQL bigint can come back as strings
    const allFacesToRemove = similarResult.rows.map((r) => parseInt(r.face_id, 10));

    if (allFacesToRemove.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "No faces found to remove", removedCount: 0 };
    }

    // Step 2: Get a sample face that remains in the cluster (for cannot-link)
    const remainingFaceQuery = `
      SELECT id FROM face
      WHERE cluster_id = $1 AND id != ALL($2)
      LIMIT 1
    `;
    const remainingResult = await client.query(remainingFaceQuery, [clusterId, allFacesToRemove]);

    // Step 3: Remove faces from cluster and add to unassigned pool
    // Using 'unassigned' status allows them to join other clusters while
    // respecting cannot-link constraints that prevent rejoining original cluster
    const removeQuery = `
      UPDATE face
      SET cluster_id = NULL,
          cluster_status = 'unassigned',
          cluster_confidence = 0,
          unassigned_since = NOW()
      WHERE id = ANY($1) AND cluster_id = $2
      RETURNING id
    `;
    const removeResult = await client.query(removeQuery, [allFacesToRemove, clusterId]);
    const removedCount = removeResult.rowCount || 0;

    // Step 4: Update cluster face count
    await client.query(
      `UPDATE cluster SET face_count = GREATEST(0, face_count - $1), updated_at = NOW() WHERE id = $2`,
      [removedCount, clusterId],
    );

    // Step 5: Create cannot-link constraints between removed faces and a remaining cluster face
    // This prevents the removed faces from rejoining this cluster
    if (remainingResult.rows.length > 0) {
      const remainingFaceId = parseInt(remainingResult.rows[0].id, 10);

      for (const removedFaceId of allFacesToRemove) {
        // Ensure canonical ordering: face_id_1 < face_id_2
        const [id1, id2] =
          removedFaceId < remainingFaceId ? [removedFaceId, remainingFaceId] : [remainingFaceId, removedFaceId];

        await client.query(
          `INSERT INTO cannot_link (face_id_1, face_id_2, created_by)
           VALUES ($1, $2, 'web')
           ON CONFLICT (face_id_1, face_id_2) DO NOTHING`,
          [id1, id2],
        );
      }
    }

    await client.query("COMMIT");

    return {
      success: true,
      message: `Removed ${removedCount} face${removedCount !== 1 ? "s" : ""} from cluster`,
      removedCount,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to dissociate faces:", error);
    return { success: false, message: "Failed to remove faces", removedCount: 0 };
  } finally {
    client.release();
  }
}

export async function searchClusters(query: string, excludeClusterId?: string, limit = 20) {
  await initDatabase();

  // Search by cluster ID or person name
  const searchQuery = `
    SELECT c.id, c.face_count, c.representative_face_id,
           f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           p.id as photo_id, p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM cluster c
    LEFT JOIN face f ON c.representative_face_id = f.id
    LEFT JOIN photo p ON f.photo_id = p.id
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

    // Move all faces from source to target cluster
    const moveFacesQuery = `
      UPDATE face
      SET cluster_id = $1,
          cluster_status = 'manual'
      WHERE cluster_id = $2
      RETURNING id
    `;
    const moveResult = await client.query(moveFacesQuery, [targetClusterId, sourceClusterId]);
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
        FROM face f
        JOIN face_embedding fe ON f.id = fe.face_id
        WHERE f.cluster_id = $1
      )
      WHERE id = $1
    `;
    await client.query(centroidQuery, [targetClusterId]);

    // Delete the source cluster
    await client.query(`DELETE FROM cluster WHERE id = $1`, [sourceClusterId]);

    await client.query("COMMIT");

    return {
      success: true,
      message: `Merged ${movedCount} faces into cluster ${targetClusterId}`,
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

export async function addFacesToCluster(clusterId: string, faceIds: number[]) {
  await initDatabase();

  if (faceIds.length === 0) {
    return { success: false, message: "No faces selected", addedCount: 0 };
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

    // Add faces to the cluster (only unclustered faces)
    const addFacesQuery = `
      UPDATE face
      SET cluster_id = $1,
          cluster_status = 'manual',
          cluster_confidence = 1.0,
          unassigned_since = NULL
      WHERE id = ANY($2) AND cluster_id IS NULL
      RETURNING id
    `;
    const addResult = await client.query(addFacesQuery, [clusterId, faceIds]);
    const addedCount = addResult.rowCount || 0;

    if (addedCount === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "No unclustered faces to add", addedCount: 0 };
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
        FROM face f
        JOIN face_embedding fe ON f.id = fe.face_id
        WHERE f.cluster_id = $1
      )
      WHERE id = $1
    `;
    await client.query(centroidQuery, [clusterId]);

    await client.query("COMMIT");

    return {
      success: true,
      message: `Added ${addedCount} face${addedCount !== 1 ? "s" : ""} to cluster`,
      addedCount,
      clusterId,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to add faces to cluster:", error);
    return { success: false, message: "Failed to add faces to cluster", addedCount: 0 };
  } finally {
    client.release();
  }
}

export async function getFaceDetails(faceId: number) {
  await initDatabase();

  const query = `
    SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           f.confidence, f.cluster_id, f.cluster_status,
           f.photo_id, p.normalized_path, p.normalized_width, p.normalized_height,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name
    FROM face f
    JOIN photo p ON f.photo_id = p.id
    LEFT JOIN person per ON f.person_id = per.id
    WHERE f.id = $1
  `;

  const result = await pool.query(query, [faceId]);
  return result.rows[0] || null;
}

export async function createClusterFromFaces(faceIds: number[]) {
  await initDatabase();

  if (faceIds.length === 0) {
    return { success: false, message: "No faces selected", clusterId: null };
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
    const clusterResult = await client.query(createClusterQuery, [faceIds.length]);
    const clusterId = clusterResult.rows[0].id;

    // Step 2: Assign all faces to the new cluster
    const assignFacesQuery = `
      UPDATE face
      SET cluster_id = $1,
          cluster_status = 'manual',
          cluster_confidence = 1.0
      WHERE id = ANY($2) AND cluster_id IS NULL
      RETURNING id
    `;
    const assignResult = await client.query(assignFacesQuery, [clusterId, faceIds]);
    const assignedCount = assignResult.rowCount || 0;

    // Step 3: Update the cluster face count to match actual assigned
    await client.query(`UPDATE cluster SET face_count = $1 WHERE id = $2`, [assignedCount, clusterId]);

    // Step 4: Set the first face as the representative
    if (assignedCount > 0) {
      const firstFaceId = faceIds[0];
      await client.query(`UPDATE cluster SET representative_face_id = $1, medoid_face_id = $1 WHERE id = $2`, [
        firstFaceId,
        clusterId,
      ]);
    }

    // Step 5: Compute and set centroid from face embeddings
    const centroidQuery = `
      WITH cluster_embeddings AS (
        SELECT embedding
        FROM face_embedding
        WHERE face_id = ANY($1)
      )
      UPDATE cluster
      SET centroid = (
        SELECT AVG(embedding)::vector(512)
        FROM cluster_embeddings
      )
      WHERE id = $2
    `;
    await client.query(centroidQuery, [faceIds, clusterId]);

    await client.query("COMMIT");

    return {
      success: true,
      message: `Created cluster ${clusterId} with ${assignedCount} face${assignedCount !== 1 ? "s" : ""}`,
      clusterId,
      assignedCount,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to create cluster from faces:", error);
    return { success: false, message: "Failed to create cluster", clusterId: null };
  } finally {
    client.release();
  }
}

export async function getSimilarFaces(faceId: number, limit = 12, similarityThreshold = 0.7) {
  await initDatabase();

  // Convert similarity threshold to distance threshold (cosine distance = 1 - similarity)
  const distanceThreshold = 1 - similarityThreshold;

  const query = `
    WITH target_embedding AS (
      SELECT embedding FROM face_embedding WHERE face_id = $1
    ),
    target_face AS (
      SELECT photo_id FROM face WHERE id = $1
    )
    SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           f.confidence, f.cluster_id, f.cluster_status, f.cluster_confidence,
           p.id as photo_id, p.normalized_path, p.normalized_width, p.normalized_height,
           c.face_count as cluster_face_count,
           TRIM(CONCAT(per.first_name, ' ', COALESCE(per.last_name, ''))) as person_name,
           1 - (fe.embedding <=> te.embedding) as similarity
    FROM face_embedding fe
    CROSS JOIN target_embedding te
    JOIN face f ON fe.face_id = f.id
    JOIN photo p ON f.photo_id = p.id
    LEFT JOIN cluster c ON f.cluster_id = c.id
    LEFT JOIN person per ON c.person_id = per.id
    WHERE fe.face_id != $1
      AND f.photo_id != (SELECT photo_id FROM target_face)
      AND (fe.embedding <=> te.embedding) < $2
    ORDER BY (fe.embedding <=> te.embedding) ASC
    LIMIT $3
  `;

  const result = await pool.query(query, [faceId, distanceThreshold, limit]);
  return result.rows;
}

export async function dissociateFaceWithConfidenceCutoff(clusterId: string, faceId: number) {
  await initDatabase();

  const client = await pool.connect();
  try {
    await client.query("BEGIN");

    // Step 1: Get the confidence of the selected face
    const confidenceQuery = `
      SELECT cluster_confidence FROM face WHERE id = $1 AND cluster_id = $2
    `;
    const confidenceResult = await client.query(confidenceQuery, [faceId, clusterId]);

    if (confidenceResult.rows.length === 0) {
      await client.query("ROLLBACK");
      return { success: false, message: "Face not found in cluster", constrainedCount: 0, cutoffCount: 0 };
    }

    const cutoffConfidence = parseFloat(confidenceResult.rows[0].cluster_confidence);

    // Step 2: Find all faces with lower confidence (these will be unconstrained)
    const lowerConfidenceQuery = `
      SELECT id FROM face
      WHERE cluster_id = $1 AND cluster_confidence < $2 AND id != $3
    `;
    const lowerResult = await client.query(lowerConfidenceQuery, [clusterId, cutoffConfidence, faceId]);
    const lowerConfidenceFaceIds = lowerResult.rows.map((r) => parseInt(r.id, 10));

    // Step 3: Get a remaining face for the cannot-link constraint
    const allFacesToRemove = [faceId, ...lowerConfidenceFaceIds];
    const remainingFaceQuery = `
      SELECT id FROM face
      WHERE cluster_id = $1 AND id != ALL($2)
      LIMIT 1
    `;
    const remainingResult = await client.query(remainingFaceQuery, [clusterId, allFacesToRemove]);

    // Step 4: Remove the selected face (constrained - cannot rejoin)
    await client.query(
      `UPDATE face
       SET cluster_id = NULL,
           cluster_status = 'unassigned',
           cluster_confidence = 0,
           unassigned_since = NOW()
       WHERE id = $1`,
      [faceId],
    );

    // Step 5: Create cannot-link for the selected face only
    if (remainingResult.rows.length > 0) {
      const remainingFaceId = parseInt(remainingResult.rows[0].id, 10);
      const [id1, id2] = faceId < remainingFaceId ? [faceId, remainingFaceId] : [remainingFaceId, faceId];

      await client.query(
        `INSERT INTO cannot_link (face_id_1, face_id_2, created_by)
         VALUES ($1, $2, 'web')
         ON CONFLICT (face_id_1, face_id_2) DO NOTHING`,
        [id1, id2],
      );
    }

    // Step 6: Remove lower confidence faces (unconstrained - can rejoin other clusters)
    let cutoffCount = 0;
    if (lowerConfidenceFaceIds.length > 0) {
      const cutoffResult = await client.query(
        `UPDATE face
         SET cluster_id = NULL,
             cluster_status = 'unassigned',
             cluster_confidence = 0,
             unassigned_since = NOW()
         WHERE id = ANY($1)
         RETURNING id`,
        [lowerConfidenceFaceIds],
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
      message: `Removed 1 face (constrained) and ${cutoffCount} lower-confidence face${cutoffCount !== 1 ? "s" : ""}`,
      constrainedCount: 1,
      cutoffCount,
    };
  } catch (error) {
    await client.query("ROLLBACK");
    console.error("Failed to dissociate face with cutoff:", error);
    return { success: false, message: "Failed to remove faces", constrainedCount: 0, cutoffCount: 0 };
  } finally {
    client.release();
  }
}
