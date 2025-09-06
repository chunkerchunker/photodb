from typing import List, Dict, Any, Optional
import json
from ..database.connection import Connection


class PhotoQueries:
    def __init__(self, connection_string: Optional[str] = None):
        self.db = Connection(connection_string)

    def get_years_with_photos(self) -> List[Dict[str, Any]]:
        query = """
            SELECT EXTRACT(YEAR FROM m.captured_at)::int as year,
                   COUNT(*)::int as photo_count,
                   MIN(p.id) as sample_photo_id
            FROM metadata m
            JOIN photo p ON p.id = m.photo_id
            WHERE m.captured_at IS NOT NULL
            GROUP BY year
            ORDER BY year DESC
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_months_in_year(self, year: int) -> List[Dict[str, Any]]:
        # First get month statistics
        query = """
            SELECT EXTRACT(MONTH FROM m.captured_at)::int as month,
                   COUNT(*)::int as photo_count
            FROM metadata m
            WHERE EXTRACT(YEAR FROM m.captured_at) = %s
              AND m.captured_at IS NOT NULL
            GROUP BY month
            ORDER BY month
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (year,))
                columns = [col[0] for col in cursor.description]
                months = [dict(zip(columns, row)) for row in cursor.fetchall()]

                # For each month, get sample photo IDs
                for month_data in months:
                    sample_query = """
                        SELECT p.id
                        FROM photo p
                        JOIN metadata m ON p.id = m.photo_id
                        WHERE EXTRACT(YEAR FROM m.captured_at) = %s
                          AND EXTRACT(MONTH FROM m.captured_at) = %s
                        ORDER BY m.captured_at
                        LIMIT 4
                    """
                    cursor.execute(sample_query, (year, month_data["month"]))
                    month_data["sample_photo_ids"] = [row[0] for row in cursor.fetchall()]

                return months

    def get_photos_by_month(
        self, year: int, month: int, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT p.id, p.filename, p.normalized_path,
                   m.captured_at, m.latitude, m.longitude,
                   la.description, la.emotional_tone,
                   la.objects, la.people_count
            FROM photo p
            JOIN metadata m ON p.id = m.photo_id
            LEFT JOIN llm_analysis la ON p.id = la.photo_id
            WHERE EXTRACT(YEAR FROM m.captured_at) = %s
              AND EXTRACT(MONTH FROM m.captured_at) = %s
            ORDER BY m.captured_at, p.filename
            LIMIT %s OFFSET %s
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (year, month, limit, offset))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_photo_count_by_month(self, year: int, month: int) -> int:
        query = """
            SELECT COUNT(*) as count
            FROM metadata m
            WHERE EXTRACT(YEAR FROM m.captured_at) = %s
              AND EXTRACT(MONTH FROM m.captured_at) = %s
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (year, month))
                return cursor.fetchone()[0]

    def get_photo_details(self, photo_id: int) -> Optional[Dict[str, Any]]:
        query = """
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
            WHERE p.id = %s
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (photo_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                columns = [col[0] for col in cursor.description]
                result = dict(zip(columns, row))

                if result.get("metadata_extra"):
                    result["metadata_extra"] = (
                        json.loads(result["metadata_extra"])
                        if isinstance(result["metadata_extra"], str)
                        else result["metadata_extra"]
                    )
                if result.get("analysis"):
                    result["analysis"] = (
                        json.loads(result["analysis"])
                        if isinstance(result["analysis"], str)
                        else result["analysis"]
                    )

                return result

    def get_faces_for_photo(self, photo_id: int) -> List[Dict[str, Any]]:
        """Get all faces detected in a photo with their bounding boxes and clustering info."""
        query = """
            SELECT f.id, f.bbox_x, f.bbox_y, f.bbox_width, f.bbox_height,
                   f.confidence, f.person_id, p.name as person_name,
                   f.cluster_id, f.cluster_status, f.cluster_confidence
            FROM face f
            LEFT JOIN person p ON f.person_id = p.id
            WHERE f.photo_id = %s
            ORDER BY f.confidence DESC
        """
        
        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (photo_id,))
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_photo_by_id(self, photo_id: int) -> Optional[Dict[str, Any]]:
        query = """
            SELECT id, filename, normalized_path
            FROM photo
            WHERE id = %s
        """

        with self.db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (photo_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                columns = [col[0] for col in cursor.description]
                return dict(zip(columns, row))
