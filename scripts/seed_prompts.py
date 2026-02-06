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

FACE_AGE_PROMPTS = [
    ("baby", [
        "a photo of a baby",
        "a photo of an infant",
        "a newborn baby",
        "a baby face",
        "a small child under 1 year old"
    ]),
    ("child", [
        "a photo of a child",
        "a young kid",
        "a school age child",
        "a toddler or preschooler",
        "a little boy or girl"
    ]),
    ("teen", [
        "a photo of a teenager",
        "a teen",
        "an adolescent",
        "a high school student",
        "a young person aged 13 to 19"
    ]),
    ("young_adult", [
        "a photo of a young adult",
        "a person in their 20s",
        "a college student or young professional",
        "a young man or woman",
    ]),
    ("adult", [
        "a photo of an adult",
        "a middle aged person",
        "a person in their 30s or 40s or 50s",
        "a mature adult face",
    ]),
    ("senior", [
        "a photo of a senior citizen",
        "an elderly person",
        "a person over 60 years old",
        "a grandmother or grandfather",
        "an old person with wrinkles and grey hair"
    ]),
]

FACE_EMOTION_PROMPTS = [
    ("happy", [
        "a photo of a happy person",
        "a smiling joyful face",
        "someone looking cheerful and content",
        "a person beaming with happiness",
        "an expression of joy"
    ]),
    ("sad", [
        "a photo of a sad person",
        "an unhappy or melancholic face",
        "someone looking downcast or gloomy",
        "tearful eyes",
        "an expression of sorrow"
    ]),
    ("angry", [
        "a photo of an angry person",
        "a furious or irritated face",
        "someone looking mad or frustrated",
        "clenched jaw or furrowed brow",
        "an expression of anger"
    ]),
    ("surprised", [
        "a photo of a surprised person",
        "a shocked or astonished face",
        "eyes wide open in surprise",
        "jaw dropped in disbelief",
        "an amazed expression"
    ]),
    ("fearful", [
        "a photo of a fearful person",
        "a scared or anxious face",
        "someone looking terrified",
        "wide eyes showing fear",
        "an expression of panic"
    ]),
    ("disgusted", [
        "a photo of a disgusted person",
        "a repulsed or nauseated face",
        "scrunched nose in disgust",
        "a reaction to something gross",
        "an expression of aversion"
    ]),
    ("neutral", [
        "a photo of a person with a neutral expression",
        "a calm face with no strong emotion",
        "a blank stare",
        "a passport photo style face",
        "an expressionless face"
    ]),
    ("excited", [
        "a photo of an excited person",
        "an enthusiastic or thrilled face",
        "someone looking very happy and energetic",
        "an expression of eagerness"
    ]),
]

FACE_GAZE_PROMPTS = [
    ("looking_at_camera", [
        "a photo of a person looking directly at the camera",
        "eyes making eye contact",
        "face turned towards the lens"
    ]),
    ("looking_away", [
        "a photo of a person looking away from the camera",
        "eyes averted",
        "face in profile or turned aside"
    ]),
     ("eyes_closed", [
        "a photo of a person with eyes closed",
        "sleeping or resting eyes",
        "blinking or shut eyes"
    ]),
]

SCENE_EVENT_PROMPTS = [
    ("birth_newborn", [
        "a newborn baby in a hospital bassinette",
        "a mother holding a newborn infant in a hospital bed",
        "a baby swaddled in blankets",
        "birth announcement photo",
        "parents with their new baby"
    ]),
    ("wedding", [
        "a wedding ceremony",
        "a bride and groom kissing",
        "a woman in a white wedding dress",
        "a couple exchanging rings",
        "a wedding reception with cake and dancing",
        "bridesmaids and groomsmen"
    ]),
    ("funeral", [
        "a funeral or memorial service",
        "people wearing black clothing in mourning",
        "a casket or coffin with flowers",
        "a hearse or cemetery scene",
        "a church service for a funeral"
    ]),
    ("graduation", [
        "a graduation ceremony",
        "a student wearing a cap and gown",
        "holding a diploma or degree",
        "tossing graduation caps in the air",
        "class of graduates posing together"
    ]),
    ("birthday", [
        "a birthday party celebration",
        "a birthday cake with lit candles",
        "someone blowing out candles on a cake",
        "balloons and party hats",
        "opening birthday presents"
    ]),
    ("holiday", [
        "a christmas tree with lights and ornaments",
        "family gathering for thanksgiving dinner",
        "halloween costumes and pumpkins",
        "holiday decorations and festive atmosphere",
        "hanukkah menorah or festive lighting"
    ]),
    ("family_portrait", [
        "a formal family portrait",
        "a large group photo of family members",
        "multiple generations posing together",
        "parents and children standing together for a photo"
    ]),
    ("retirement", [
        "a retirement party",
        "an older colleague being celebrated",
        "a cake saying happy retirement",
        "a farewell party at an office"
    ]),
    ("major_travel", [
        "traveling with luggage and suitcases",
        "an airport terminal or airplane cabin",
        "a famous tourist landmark",
        "sightseeing on vacation",
        "a passport and tickets"
    ]),
    ("grandchild_birth", [
        "grandparents holding a new baby",
        "elderly people smiling at an infant",
        "meeting the new grandchild",
        "multi-generational photo with a baby"
    ]),
]

SCENE_MOOD_PROMPTS = [
    ("joyful", ["a joyful scene", "celebration and happiness", "cheerful atmosphere"]),
    ("peaceful", ["a peaceful scene", "calm and serene environment", "tranquil quiet mood"]),
    ("somber", ["a somber scene", "sad or gloomy atmosphere", "melancholic mood"]),
    ("tense", ["a tense scene", "dramatic or suspenseful atmosphere", "intense mood"]),
    ("energetic", ["an energetic scene", "lively and dynamic action", "excitement"]),
    ("romantic", ["a romantic scene", "love and intimacy", "a couple creating a romantic mood"]),
    ("nostalgic", ["a nostalgic scene", "vintage or retro feel", "sentimental atmosphere"]),
]

SCENE_SETTING_PROMPTS = [
    ("indoor", [
        "an indoor scene", "inside a room", "interior of a building", "indoor lighting"
    ]),
    ("outdoor_nature", [
        "an outdoor nature scene", "forest, mountains, or beach", "natural landscape", "trees and sky"
    ]),
    ("outdoor_urban", [
        "an outdoor urban scene", "city streets and buildings", "architecture and roads", "downtown area"
    ]),
    ("transportation", [
        "inside a vehicle", "car, bus, train, or plane", "traveling in a vehicle", "cockpit or dashboard"
    ]),
]

PROMPT_SETS = {
    "face_age": FACE_AGE_PROMPTS,
    "face_emotion": FACE_EMOTION_PROMPTS,
    "face_gaze": FACE_GAZE_PROMPTS,
    "scene_event": SCENE_EVENT_PROMPTS,
    "scene_mood": SCENE_MOOD_PROMPTS,
    "scene_setting": SCENE_SETTING_PROMPTS,
}


# =============================================================================
# OpenAI CLIP ImageNet Zero-Shot Templates
# Source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
# =============================================================================
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a photo of many {}.",
]

SCENE_TEMPLATES = [
    "a photo of a {} scene.",
    "a photo of a {} environment.",
    "a view of the {}.",
    "inside a {}.",
    "a photo of the {} location.",
    "a picture of a {}.",
    "looking at a {}.",
    "a photo of a place used for {}.",
    "a {} setting.",
    "an outdoor {} scene.",
    "an indoor {} scene.",
    "a photo involving {}.",
    "a scene depicting {}.",
    "the atmosphere of a {}.",
    "a vacation photo of {}.",
    "a travel photo of {}.",
    "a photo showing {}.",
    "environment: {}.",
    "location: {}.",
    "scene: {}.",
]

SENTIMENT_TEMPLATES = [
    "a photo of a {} face.",
    "a {} expression.",
    "they look {}.",
    "a face showing the emotion: {}.",
    "a person looking {}.",
    "an expression of {}.",
    "a photo of a person who is {}.",
    "a {} mood.",
    "feeling {}.",
    "a look of {}.",
    "the person is {}.",
    "face depicting {}.",
    "emotion: {}.",
    "mood: {}.",
    "sentiment: {}.",
]

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

    for category_name, prompts_data in PROMPT_SETS.items():
        # Ensure category exists or create it
        category = repository.get_prompt_category_by_name(category_name)
        if not category:
            try:
                category = repository.create_prompt_category(category_name)
                logger.info(f"Created new category: {category_name}")
            except AttributeError:
                 logger.warning(f"Category '{category_name}' not found and creation not supported, skipping")
                 skipped_categories.append(category_name)
                 continue

        logger.info(f"Processing category: {category_name} ({len(prompts_data)} labels)")

        # Determine which template set to use based on category
        if category_name in ["face_emotion", "scene_mood"]:
            templates = SENTIMENT_TEMPLATES
            template_name = "SENTIMENT"
        elif category_name in ["scene_event", "scene_setting"]:
            templates = SCENE_TEMPLATES
            template_name = "SCENE"
        else:
            templates = IMAGENET_TEMPLATES
            template_name = "IMAGENET"
        
        logger.info(f"  Using {template_name} templates ({len(templates)} templates)")

        # Get existing prompts for this category
        existing = repository.get_prompts_by_category(category.id, with_embeddings=False)
        existing_labels = {p.label for p in existing}

        for label, rich_descriptions in prompts_data:
            needs_embedding = recompute or label not in existing_labels

            if needs_embedding:
                # 1. Generate template-based prompts using the label itself
                #    Convert underscore to space for better natural language mapping
                clean_label = label.replace("_", " ")
                template_prompts = [t.format(clean_label) for t in templates]
                
                # 2. Add the rich manually curated descriptions from PROMPT_SETS
                #    Ensure rich_descriptions is a list
                if isinstance(rich_descriptions, str):
                    rich_descriptions = [rich_descriptions]
                
                # 3. Combine ALL prompts for massive ensemble
                all_prompts = template_prompts + rich_descriptions
                
                # Compute embedding on the massive ensemble
                try:
                    embedding = analyzer.encode_text_ensemble(all_prompts)
                    
                    # Store the label as the prompt text, but maybe log the count
                    # We can't store 80+ lines in the prompt_text field easily, so we store the rich descriptions
                    # as the "visible" text, but the embedding represents the full ensemble.
                    prompt_text_stored = " | ".join(rich_descriptions)
                        
                    embedding_list = embedding.cpu().squeeze().tolist()
                except Exception as e:
                    logger.error(f"Failed to compute embedding for '{label}': {e}")
                    continue

                prompt = PromptEmbedding.create(
                    category_id=category.id,
                    label=label,
                    prompt_text=prompt_text_stored[:1000], 
                    model_name=model_name,
                    embedding=embedding_list,
                )
                repository.upsert_prompt_embedding(prompt)

                if label in existing_labels:
                    total_updated += 1
                    logger.debug(f"  Updated: {label} (Ensemble size: {len(all_prompts)})")
                else:
                    total_created += 1
                    logger.debug(f"  Created: {label} (Ensemble size: {len(all_prompts)})")

    logger.info(f"Done: {total_created} created, {total_updated} updated")
    if skipped_categories:
        logger.warning(f"Skipped categories: {', '.join(skipped_categories)}")


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
        pool.close_all()


if __name__ == "__main__":
    main()
