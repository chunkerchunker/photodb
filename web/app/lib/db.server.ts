import { Pool } from "pg";
import path from "path";
import fs from "fs";
import dotenv from "dotenv";

// Mark this file as server-only
export {};

// Load environment variables from .env file
dotenv.config({ path: path.join(process.cwd(), "..", ".env") });

// Create a connection pool
const pool = new Pool({
	connectionString:
		process.env.DATABASE_URL || "postgresql://localhost/photodb",
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

export async function getPhotosByMonth(
	year: number,
	month: number,
	limit = 100,
	offset = 0,
) {
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
				? photo.description.substring(0, 47) + "..."
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
	return parseInt(result.rows[0].count);
}

export async function getPhotoDetails(photoId: number) {
	await initDatabase();

	const query = `
    SELECT p.id, p.filename, p.normalized_path, 
           p.created_at as photo_created_at, p.updated_at as photo_updated_at,
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

	// Get faces for this photo
	const facesQuery = `
    SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
           f.confidence, f.person_id, p.name as person_name,
           f.cluster_id, f.cluster_status, f.cluster_confidence
    FROM face f
    LEFT JOIN person p ON f.person_id = p.id
    WHERE f.photo_id = $1
    ORDER BY f.confidence DESC
  `;

	const facesResult = await pool.query(facesQuery, [photoId]);
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

	// Try to get image dimensions if we have faces
	if (photo.faces.length > 0 && photo.normalized_path) {
		try {
			const sharp = await import("sharp");
			const imagePath = photo.normalized_path;
			if (fs.existsSync(imagePath)) {
				const metadata = await sharp.default(imagePath).metadata();
				photo.image_width = metadata.width || null;
				photo.image_height = metadata.height || null;
			}
		} catch (error) {
			console.warn("Could not get image dimensions:", error);
			photo.image_width = null;
			photo.image_height = null;
		}
	}

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
