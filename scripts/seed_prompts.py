#!/usr/bin/env python3
"""
Seed initial prompts and compute embeddings.

Usage:
    uv run python scripts/seed_prompts.py
    uv run python scripts/seed_prompts.py --recompute-embeddings
"""

import argparse
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Prompt definitions
# =============================================================================

FACE_EMOTION_PROMPTS = [
    ("happy", "a photo of a happy smiling joyful person"),
    ("sad", "a photo of a sad unhappy melancholic person"),
    ("angry", "a photo of an angry frustrated irritated person"),
    ("surprised", "a photo of a surprised shocked astonished person"),
    ("fearful", "a photo of a fearful scared anxious person"),
    ("disgusted", "a photo of a disgusted repulsed person"),
    ("neutral", "a photo of a person with neutral calm expression"),
    ("confused", "a photo of a confused puzzled person"),
    ("excited", "a photo of an excited enthusiastic thrilled person"),
    ("proud", "a photo of a proud confident person"),
    ("embarrassed", "a photo of an embarrassed shy person"),
    ("contempt", "a photo of a person showing contempt or disdain"),
]

FACE_GAZE_PROMPTS = [
    ("looking_at_camera", "a photo of a person looking directly at the camera"),
    ("looking_away", "a photo of a person looking away from the camera"),
    ("looking_down", "a photo of a person looking downward"),
    ("looking_up", "a photo of a person looking upward"),
    ("eyes_closed", "a photo of a person with eyes closed"),
]

SCENE_MOOD_PROMPTS = [
    ("joyful", "a joyful happy celebratory cheerful scene"),
    ("peaceful", "a peaceful calm serene tranquil scene"),
    ("somber", "a somber sad melancholic gloomy scene"),
    ("tense", "a tense dramatic intense suspenseful scene"),
    ("energetic", "an energetic exciting dynamic lively scene"),
    ("romantic", "a romantic loving intimate scene"),
    ("mysterious", "a mysterious intriguing enigmatic scene"),
    ("nostalgic", "a nostalgic wistful sentimental scene"),
    ("neutral", "an ordinary everyday neutral mundane scene"),
]

SCENE_SETTING_PROMPTS = [
    # Indoor
    ("indoor_home", "a photo taken inside a home or apartment"),
    ("indoor_office", "a photo taken in an office or workplace"),
    ("indoor_restaurant", "a photo taken in a restaurant or cafe"),
    ("indoor_store", "a photo taken in a store or shopping mall"),
    ("indoor_school", "a photo taken in a school or classroom"),
    ("indoor_gym", "a photo taken in a gym or fitness center"),
    ("indoor_museum", "a photo taken in a museum or gallery"),
    ("indoor_hospital", "a photo taken in a hospital or medical facility"),
    # Outdoor natural
    ("outdoor_beach", "a photo taken at a beach with sand and ocean"),
    ("outdoor_mountain", "a photo taken in mountains or hills"),
    ("outdoor_forest", "a photo taken in a forest or woods"),
    ("outdoor_park", "a photo taken in a park or garden"),
    ("outdoor_lake", "a photo taken at a lake or river"),
    ("outdoor_desert", "a photo taken in a desert landscape"),
    ("outdoor_field", "a photo taken in an open field or meadow"),
    # Outdoor urban
    ("outdoor_city", "a photo taken in a city with buildings"),
    ("outdoor_street", "a photo taken on a street or road"),
    ("outdoor_parking", "a photo taken in a parking lot"),
    # Transportation
    ("in_car", "a photo taken inside a car or vehicle"),
    ("in_airplane", "a photo taken inside an airplane"),
    ("at_airport", "a photo taken at an airport"),
]

SCENE_ACTIVITY_PROMPTS = [
    ("celebration", "a celebration party birthday or festive event"),
    ("wedding", "a wedding ceremony or reception"),
    ("graduation", "a graduation ceremony"),
    ("travel", "travel vacation or tourism"),
    ("sports", "sports or athletic activity"),
    ("dining", "eating food or dining together"),
    ("working", "working or professional activity"),
    ("relaxing", "relaxing or leisure activity"),
    ("playing", "playing games or recreational activity"),
    ("concert", "a concert or live music performance"),
    ("meeting", "a meeting or gathering"),
    ("studying", "studying or educational activity"),
]

SCENE_TIME_PROMPTS = [
    ("daytime", "a photo taken during daytime with daylight"),
    ("sunset", "a photo taken during sunset or golden hour"),
    ("sunrise", "a photo taken during sunrise or dawn"),
    ("night", "a photo taken at night or evening"),
    ("overcast", "a photo taken on a cloudy overcast day"),
]

SCENE_WEATHER_PROMPTS = [
    ("sunny", "a photo taken on a sunny clear day"),
    ("cloudy", "a photo taken on a cloudy day"),
    ("rainy", "a photo taken in rain or wet weather"),
    ("snowy", "a photo taken in snow or winter weather"),
    ("foggy", "a photo taken in fog or mist"),
]

SCENE_SOCIAL_PROMPTS = [
    ("solo", "a photo of one person alone"),
    ("couple", "a photo of a couple or two people together"),
    ("small_group", "a photo of a small group of 3-5 people"),
    ("large_group", "a photo of a large group or crowd of people"),
    ("family", "a photo of a family with adults and children"),
    ("no_people", "a photo with no people visible"),
]

PROMPT_SETS = {
    "face_emotion": FACE_EMOTION_PROMPTS,
    "face_gaze": FACE_GAZE_PROMPTS,
    "scene_mood": SCENE_MOOD_PROMPTS,
    "scene_setting": SCENE_SETTING_PROMPTS,
    "scene_activity": SCENE_ACTIVITY_PROMPTS,
    "scene_time": SCENE_TIME_PROMPTS,
    "scene_weather": SCENE_WEATHER_PROMPTS,
    "scene_social": SCENE_SOCIAL_PROMPTS,
}


def seed_prompts(repository, recompute: bool = False):
    """Seed prompts into database and compute embeddings."""
    from photodb.database.models import PromptEmbedding
    from photodb.utils.mobileclip_analyzer import MobileCLIPAnalyzer

    try:
        analyzer = MobileCLIPAnalyzer()
    except Exception as e:
        logger.error(f"Failed to initialize MobileCLIPAnalyzer: {e}")
        raise RuntimeError("Cannot seed prompts without MobileCLIP model") from e

    model_name = analyzer.model_name

    total_created = 0
    total_updated = 0
    skipped_categories: list[str] = []

    for category_name, prompts in PROMPT_SETS.items():
        category = repository.get_prompt_category_by_name(category_name)
        if not category:
            logger.warning(f"Category '{category_name}' not found, skipping")
            skipped_categories.append(category_name)
            continue

        logger.info(f"Processing category: {category_name} ({len(prompts)} prompts)")

        # Get existing prompts for this category
        existing = repository.get_prompts_by_category(category.id, with_embeddings=False)
        existing_labels = {p.label for p in existing}

        for label, prompt_text in prompts:
            needs_embedding = recompute or label not in existing_labels

            if needs_embedding:
                # Compute embedding
                try:
                    embedding = analyzer.encode_text(prompt_text)
                    embedding_list = embedding.cpu().squeeze().tolist()
                except Exception as e:
                    logger.error(f"Failed to compute embedding for '{label}': {e}")
                    continue

                prompt = PromptEmbedding.create(
                    category_id=category.id,
                    label=label,
                    prompt_text=prompt_text,
                    model_name=model_name,
                    embedding=embedding_list,
                )
                repository.upsert_prompt_embedding(prompt)

                if label in existing_labels:
                    total_updated += 1
                    logger.debug(f"  Updated: {label}")
                else:
                    total_created += 1
                    logger.debug(f"  Created: {label}")

    logger.info(f"Done: {total_created} created, {total_updated} updated")
    if skipped_categories:
        logger.warning(f"Skipped categories (not in database): {', '.join(skipped_categories)}")


def main():
    parser = argparse.ArgumentParser(description="Seed prompts and compute embeddings")
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="Recompute embeddings for all prompts",
    )
    args = parser.parse_args()

    # Validate DATABASE_URL is set
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        logger.error("Set it with: export DATABASE_URL='postgresql://localhost/photodb'")
        sys.exit(1)

    from photodb.database.connection import ConnectionPool
    from photodb.database.repository import PhotoRepository

    pool = ConnectionPool()
    repository = PhotoRepository(pool)

    try:
        seed_prompts(repository, recompute=args.recompute_embeddings)
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
