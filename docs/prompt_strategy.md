# MobileCLIP Prompt Strategy

## Overview
This document outlines the prompt engineering strategy used to optimize **MobileCLIP-S2** for zero-shot classification in PhotoDB. Our goal is to maximize the robustness and accuracy of embeddings for Event, Scene, and Person analysis without fine-tuning the model.

## Core Strategy: Massive Prompt Ensembling
MobileCLIP (like the original CLIP) is highly sensitive to phrasing. A single prompt like *"a photo of a wedding"* can produce a noisy embedding. To mitigate this, we use **Prompt Ensembling**: generating dozens of variations for a single class label and averaging their embeddings into a single, stable vector.

We combine two types of prompts for every category:
1.  **Rich Descriptions**: Manually curated, highly descriptive sentences that capture the semantic "core" of the class (e.g., *"a wedding ceremony with a bride and groom in white dress and suit"*).
2.  **Standardized Templates**: A large set of machine-generated sentence structures (e.g., *"a blurry photo of a..."*, *"a sketch of a..."*) that vary domain, quality, and style.

### Template Sets by Domain

We selected specific template sets based on the semantic nature of the category to ensure the model focuses on the correct visual features.

#### 1. Objects & Demographics (Standard)
*   **Target Categories**: `face_age`, `face_gaze`, General Objects.
*   **Source**: [OpenAI CLIP ImageNet Zero-Shot Templates](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb).
*   **Count**: ~80 templates.
*   **Why**: These are the gold standard for object recognition, covering a massive range of image qualities (blur, crop, sketch, toy, jpeg artifacts) to ensure the model recognizes the *subject* regardless of presentation.

#### 2. Scenes & Events
*   **Target Categories**: `scene_event`, `scene_setting`.
*   **Source**: Inspired by **Places365** and **SUN397** zero-shot benchmarks.
*   **Count**: ~20 templates.
*   **Examples**:
    *   *"a photo of a {} scene."*
    *   *"inside a {}."*
    *   *"a view of the {}."*
    *   *"the atmosphere of a {}."*
*   **Why**: These templates direct attention to the *environment* and *context*, preventing the model from fixating on a single object. For example, for "Wedding", we want the *scene* of a wedding, not just an object tagged "wedding".

#### 3. Sentiment & Emotion
*   **Target Categories**: `face_emotion`, `scene_mood`.
*   **Source**: Emotion Recognition in the Wild (EmotiW) and similar affective computing benchmarks.
*   **Count**: ~15 templates.
*   **Examples**:
    *   *"a {} expression."*
    *   *"they look {}."*
    *   *"a face showing the emotion: {}."*
*   **Why**: These focus the model on abstract, facial, or atmospheric cues rather than objects or places.

## Implementation Details
The logic sits in `scripts/seed_prompts.py`:
1.  **Selection**: The script checks the category name (`face_emotion`, `scene_event`, etc.).
2.  **Expansion**: It takes the raw label (e.g., `birth_newborn`) and populates the appropriate template set.
3.  **Ensembling**: It appends the "Rich Descriptions" defined in `PROMPT_SETS`.
4.  **Encoding**: It computes the feature vector for *every* prompt string (templates + rich) and averages them (`mean-pooling`) to create the final stored embedding.

## Future Work & Improvements
The current prompts are a solid baseline ("decent but not great"). Areas for future exploration:

1.  **Negative Prompting**: Subtracting embeddings for common confusions (e.g., subtracting "birthday" from "wedding" if they often confuse cake/gathering features).
2.  **LLM-Generated Descriptions**: Instead of manually writing 3-5 rich descriptions, we could use an LLM (GPT-4/Claude) to generate 50+ visually distinct descriptions for each event to capture edge cases.
3.  **DataComp-Specific Tuning**: MobileCLIP was trained on DataCompDR. Investigating the specific caption distributions in DataComp could reveal "magic words" or patterns that the model heavily overfits to.
4.  **Hierarchical Taxonomy**: Splitting "Holiday" into "Christmas", "Halloween", etc., at the embedding level rather than grouping them might improve distinctiveness.
