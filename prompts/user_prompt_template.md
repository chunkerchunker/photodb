# User Prompt Template

Extract metadata according to the schema below. Use EXIF when present; otherwise infer conservatively from visual cues.

Focus: people, activities, events, location cues, time/season cues, objects, text, and accessibility descriptions.

Return only JSON. If uncertain, use hypotheses with confidence scores.

{EXIF_CONTEXT}