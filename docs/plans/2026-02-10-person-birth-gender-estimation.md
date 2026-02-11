# Person Birth Date & Gender Estimation Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a robust algorithm for estimating the birth date and gender of persons based on face detection age/gender estimates, photo timestamps, and co-occurrence constraints.

**Key Insight:** When multiple people appear in the same photo, that creates a reference point in time. If we have high confidence in one person's birth date, it provides information about other faces in that photo.

---

## Problem Statement

Given:

- Face detections with per-face age/gender estimates (noisy, from MiVOLO)
- Face clusters (potentially imperfect groupings)
- Persons (groups of clusters representing the same individual)
- Photo timestamps (from EXIF metadata)
- Co-occurrence (multiple people appearing in the same photo)

Estimate: **Gender** and **birth date** for each person, with confidence scores.

### Challenges

1. **Cluster inaccuracies**: Wrong faces may be grouped together
2. **Temporal span**: Photos of a person may span many years
3. **Sparse data**: Some people have few photos
4. **Noisy age estimates**: MiVOLO estimates have significant variance
5. **Missing dates**: Some photos lack EXIF timestamps

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    PERSON ESTIMATION PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
│  │  Detection   │────►│  Per-Person  │────►│  Constraint Propagation  │ │
│  │  Age/Gender  │     │  Aggregation │     │  (Co-occurrence Graph)   │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────┘ │
│         │                    │                         │                 │
│         ▼                    ▼                         ▼                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────────┐ │
│  │ per-detection│     │   Initial    │     │     Refined Estimates    │ │
│  │  estimates   │     │  Estimates   │     │   + Confidence Scores    │ │
│  └──────────────┘     └──────────────┘     └──────────────────────────┘ │
│                                                        │                 │
│                              ┌─────────────────────────┘                 │
│                              ▼                                           │
│                       ┌──────────────┐                                   │
│                       │ Ground Truth │ ◄─── User Overrides               │
│                       │   Anchors    │                                   │
│                       └──────────────┘                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```text
person_detection (age_estimate, gender, photo_id)
        │
        ├─► metadata (captured_at) ─► birth_date_estimate = captured_at - age
        │
        ▼
    cluster ─► person
        │
        ├─► Aggregate all birth_date_estimates for person
        │
        ▼
person_cooccurrence (person_id_1, person_id_2, photo_id)
        │
        ├─► Propagate confidence from anchors
        │
        ▼
person (estimated_birth_date, birth_date_confidence, gender, gender_confidence)
```

---

## Efficiency & Scalability

### Concern

With large collections (100k+ photos, 1000+ persons), naive implementation would be too slow:

- **Constraint propagation**: O(persons × iterations × cooccurrences)
- **Full recalculation on every photo**: Wasteful and slow
- **Co-occurrence graph**: Can be O(n²) if everyone appears with everyone

### Tiered Computation Model

Computation is split into three tiers with different triggers and costs:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: Per-Detection Cache (Always, O(1))                                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Trigger: Detection age_estimate is set                                     │
│  Action:  Cache birth_date_estimate = photo_date - age                      │
│  Store:   detection_birth_estimate table                                    │
│  Cost:    O(1) per detection                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: Per-Person Aggregation (On-demand, O(detections))                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Trigger: Person viewed in UI, or person's cluster modified                 │
│  Action:  Aggregate cached detection estimates → person estimate            │
│  Note:    No propagation, just aggregate own detections                     │
│  Cost:    O(n_detections) per person                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 3: Constraint Propagation (Batch only, O(persons × cooccurrences))    │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Trigger: Ground truth set, weekly maintenance, or explicit CLI             │
│  Action:  Run constraint propagation algorithm                              │
│  Note:    Never triggered by normal photo processing                        │
│  Cost:    O(persons × iterations × avg_cooccurrences)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hot Path (Photo Processing)

Photo processing must remain fast. Only O(1) operations allowed:

```python
def on_photo_processed(photo: Photo):
    """
    Lightweight operations during photo processing.
    All O(1) per detection - no aggregation or propagation.
    """
    for detection in photo.detections:
        # Tier 1: Cache birth estimate (O(1))
        if detection.age_estimate and photo.captured_at:
            cache_detection_birth_estimate(detection, photo.captured_at)

        # Mark person as stale for lazy recomputation (O(1))
        if detection.cluster_id:
            cluster = get_cluster(detection.cluster_id)
            if cluster.person_id:
                mark_person_stale(cluster.person_id)

    # Record co-occurrences (O(persons_in_photo²), typically small)
    persons_in_photo = get_persons_in_photo(photo.id)
    if len(persons_in_photo) > 1:
        for p1_id, p2_id in combinations(persons_in_photo, 2):
            record_cooccurrence(p1_id, p2_id, photo.id, photo.captured_at)
```

### Lazy Aggregation (UI/API Requests)

Person estimates are computed on-demand when requested:

```python
def get_person_with_estimates(person_id: int) -> Person:
    """
    Return person with up-to-date estimates.
    Recomputes if stale, otherwise returns cached.
    """
    person = get_person(person_id)

    if person.birth_estimate_stale:
        # Tier 2: Aggregate own detections only
        estimates = get_cached_birth_estimates_for_person(person_id)
        person.estimated_birth_date = weighted_median(estimates)
        person.birth_date_confidence = compute_confidence(estimates)
        person.birth_estimate_stale = False
        person.birth_estimate_computed_at = datetime.now(timezone.utc)
        save_person(person)

    return person
```

### Scoped Propagation

When ground truth is set, propagation is limited to affected network:

```python
def propagate_from_anchor(anchor_person_id: int, max_hops: int = 2):
    """
    Propagate from anchor to persons within N hops only.

    Most benefit comes from direct co-occurrences (hop=1).
    Diminishing returns beyond 2 hops.
    """
    affected = {anchor_person_id}
    frontier = {anchor_person_id}

    for hop in range(max_hops):
        next_frontier = set()
        for person_id in frontier:
            cooccurring = get_cooccurring_person_ids(person_id)
            next_frontier.update(cooccurring - affected)
        affected.update(next_frontier)
        frontier = next_frontier

    # Only recompute these persons
    for person_id in affected:
        recompute_with_constraints(person_id)

    logger.info(f"Propagated from anchor {anchor_person_id} to {len(affected)} persons")
```

### Operation Cost Summary

| Operation | Trigger | Cost | Frequency |
| --------- | ------- | ---- | --------- |
| Cache detection estimate | Detection created | O(1) | Every photo |
| Mark person stale | Cluster change | O(1) | Per cluster change |
| Record co-occurrence | Photo has 2+ persons | O(k²) where k=persons in photo | Per photo |
| Aggregate person estimate | UI view / API request | O(n_detections) | On demand |
| Propagate from anchor | Ground truth set | O(anchor's network size) | Rare, explicit |
| Full propagation | Weekly maintenance | O(all persons × cooccurrences) | Weekly batch |

### Collection Isolation

All operations are scoped to a single collection:

- Co-occurrence graph is per-collection
- Propagation never crosses collection boundaries
- Maintenance jobs process one collection at a time

---

## Algorithm Design

### Phase 1: Gender Estimation

Gender is time-invariant, so we aggregate across all observations using weighted voting.

```python
def estimate_gender(person: Person) -> Tuple[str, float]:
    """
    Aggregate gender predictions across all faces in person's clusters.

    Returns:
        Tuple of (gender, confidence) where gender is 'M', 'F', or 'U'
    """
    score_m = 0.0
    score_f = 0.0

    for detection in person.all_detections():
        if detection.gender == 'M':
            score_m += detection.gender_confidence or 0.5
        elif detection.gender == 'F':
            score_f += detection.gender_confidence or 0.5

    total = score_m + score_f
    if total == 0:
        return ('U', 0.0)

    if score_m > score_f:
        return ('M', score_m / total)
    else:
        return ('F', score_f / total)
```

**Cluster error detection**: If gender distribution is roughly 50/50 with high-confidence predictions on both sides, the cluster likely contains multiple people.

### Phase 2: Initial Birth Date Estimation

Each face detection with a known photo date gives us an estimate:

```text
birth_date_estimate = photo_date - estimated_age
```

```python
def initial_birth_date_estimate(person: Person) -> Tuple[date, float]:
    """
    Compute initial birth date estimate from person's own detections.

    Uses weighted median with outlier rejection for robustness.
    """
    estimates = []

    for detection in person.all_detections():
        photo_date = get_photo_date(detection.photo_id)
        if photo_date is None or detection.age_estimate is None:
            continue

        birth_est = photo_date - timedelta(days=detection.age_estimate * 365.25)

        # Weight by detection quality factors
        weight = compute_detection_weight(detection)
        estimates.append((birth_est, weight))

    if len(estimates) < 1:
        return (None, 0.0)

    # Robust aggregation: weighted median with outlier rejection
    birth_date = weighted_median(estimates)

    # Confidence based on consistency
    residuals = [abs((est - birth_date).days) for est, _ in estimates]
    mad = median_absolute_deviation(residuals) / 365.25  # in years

    confidence = compute_confidence(
        n_samples=len(estimates),
        mad_years=mad,
        temporal_spread=compute_temporal_spread(estimates)
    )

    return (birth_date, confidence)

def compute_detection_weight(detection: PersonDetection) -> float:
    """
    Compute quality weight for a detection based on multiple factors.
    """
    weight = 1.0

    # Face confidence (0-1)
    if detection.face_confidence:
        weight *= detection.face_confidence

    # Face size factor (larger faces = more reliable age estimate)
    face_area = detection.face_bbox_width * detection.face_bbox_height
    size_factor = min(1.0, face_area / 10000)  # Saturate at 100x100
    weight *= (0.5 + 0.5 * size_factor)

    # Frontal face indicator (if available from gaze detection)
    # Could be enhanced with face pose estimation

    return weight
```

### Phase 3: Constraint Propagation via Co-occurrence

The key insight: when people appear together in photos, their ages are correlated in time.

**Build co-occurrence graph:**

```text
Nodes: Persons
Edges: (person_1, person_2, photo_date, photo_id)
       for each photo containing both persons
```

```python
def propagate_constraints(max_iterations: int = 10):
    """
    Iteratively refine birth date estimates using co-occurrence constraints.

    High-confidence persons (anchors) propagate information to others.
    """
    for iteration in range(max_iterations):
        updates = []

        for person in all_persons():
            constraints = []

            # 1. Direct estimates from own detections
            for detection in person.detections:
                photo_date = get_photo_date(detection.photo_id)
                if photo_date and detection.age_estimate:
                    birth_est = photo_date - age_to_timedelta(detection.age_estimate)
                    constraints.append(Constraint(
                        source='direct',
                        birth_date=birth_est,
                        weight=compute_detection_weight(detection)
                    ))

            # 2. Cross-person constraints from co-occurrences
            for cooccurrence in get_cooccurrences(person):
                other_person = cooccurrence.other_person

                # Only use anchors (high confidence persons)
                if other_person.birth_date_confidence < ANCHOR_THRESHOLD:
                    continue

                # Find this person's detection in the shared photo
                my_detection = get_detection_in_photo(person, cooccurrence.photo_id)
                if not my_detection or not my_detection.age_estimate:
                    continue

                # Infer birth date from other person's known birth date
                # and my age in this photo
                inferred_birth = (cooccurrence.photo_date -
                                  age_to_timedelta(my_detection.age_estimate))

                # Weight by anchor's confidence AND my detection quality
                weight = (other_person.birth_date_confidence *
                         compute_detection_weight(my_detection))

                constraints.append(Constraint(
                    source='cooccurrence',
                    birth_date=inferred_birth,
                    weight=weight,
                    via_person=other_person
                ))

            # Solve constraints to get updated estimate
            if constraints:
                new_estimate, new_confidence = solve_constraints(constraints)
                updates.append((person, new_estimate, new_confidence))

        # Apply updates
        for person, birth_date, confidence in updates:
            person.estimated_birth_date = birth_date
            person.birth_date_confidence = confidence

        # Check convergence
        if has_converged(updates):
            break

def solve_constraints(constraints: List[Constraint]) -> Tuple[date, float]:
    """
    Combine multiple constraints into a single estimate.

    Uses weighted median for robustness to outliers.
    """
    estimates = [(c.birth_date, c.weight) for c in constraints]
    birth_date = weighted_median(estimates)

    # Confidence from constraint agreement
    residuals = [abs((c.birth_date - birth_date).days) for c in constraints]
    mad_days = median_absolute_deviation(residuals)

    # More constraints + better agreement = higher confidence
    n = len(constraints)
    consistency = max(0, 1 - mad_days / (365.25 * 5))  # 5 year MAD = 0 conf
    n_factor = min(1.0, n / 10)  # Saturates at 10 samples

    confidence = n_factor * consistency

    return (birth_date, confidence)
```

### Phase 4: User Overrides (Ground Truth)

When users provide definitive information, it becomes a hard constraint:

```python
def apply_ground_truth(person: Person, ground_truth: PersonGroundTruth):
    """
    Apply user-provided ground truth as an anchor.

    Ground truth overrides estimates and propagates to co-occurring persons.
    """
    if ground_truth.birth_date:
        person.ground_truth_birth_date = ground_truth.birth_date
        person.estimated_birth_date = ground_truth.birth_date
        person.birth_date_confidence = 1.0
        person.birth_date_source = 'user'

    if ground_truth.gender:
        person.ground_truth_gender = ground_truth.gender
        person.gender = ground_truth.gender
        person.gender_confidence = 1.0
        person.gender_source = 'user'

    # This person is now an anchor for propagation
    person.is_anchor = True

    # Trigger re-propagation from anchors
    propagate_from_anchors()
```

---

## Confidence Score Computation

```python
def compute_birth_date_confidence(person: Person) -> float:
    """
    Multi-factor confidence score for birth date estimate.

    Returns:
        Confidence score from 0.0 to 1.0
    """
    # Ground truth = maximum confidence
    if person.ground_truth_birth_date:
        return 1.0

    factors = []

    # Factor 1: Number of observations
    n_detections = len([d for d in person.detections if d.age_estimate])
    factors.append(min(1.0, n_detections / 10))  # Saturates at 10

    # Factor 2: Consistency of estimates
    estimates = get_birth_date_estimates(person)
    if len(estimates) > 1:
        mad_years = median_absolute_deviation(estimates) / 365.25
        consistency = max(0, 1 - mad_years / 5)  # 5 year MAD = 0
        factors.append(consistency)
    else:
        factors.append(0.3)  # Single estimate = low confidence

    # Factor 3: Temporal spread of photos
    photo_dates = [get_photo_date(d.photo_id) for d in person.detections]
    photo_dates = [d for d in photo_dates if d]
    if len(photo_dates) > 1:
        span_years = (max(photo_dates) - min(photo_dates)).days / 365.25
        spread_factor = min(1.0, span_years / 10)  # More spread = more confidence
        factors.append(spread_factor)
    else:
        factors.append(0.2)

    # Factor 4: Co-occurrence with anchors
    anchor_connections = count_cooccurrences_with_anchors(person)
    factors.append(min(1.0, anchor_connections / 3))  # 3+ anchors = max

    # Geometric mean of factors
    if not factors:
        return 0.0

    return math.exp(sum(math.log(max(f, 0.01)) for f in factors) / len(factors))
```

---

## Cluster Error Detection

Multimodal birth date distribution indicates potential cluster errors:

```python
def detect_cluster_errors(cluster: Cluster) -> Optional[ClusterError]:
    """
    Detect potential cluster errors via birth date distribution analysis.

    Returns:
        ClusterError if issues detected, None otherwise
    """
    estimates = []
    for detection in cluster.detections:
        photo_date = get_photo_date(detection.photo_id)
        if photo_date and detection.age_estimate:
            birth_est = photo_date - age_to_timedelta(detection.age_estimate)
            estimates.append(birth_est)

    if len(estimates) < 3:
        return None

    # Convert to years for analysis
    estimates_years = [e.year + e.timetuple().tm_yday / 365.25 for e in estimates]

    # Check for bimodality (two distinct groups)
    if is_bimodal(estimates_years, threshold_years=5):
        gap = find_gap(estimates_years)
        return ClusterError(
            cluster_id=cluster.id,
            error_type='likely_mixed_identity',
            message=f'Bimodal birth year distribution detected (gap at {gap})',
            suggested_split_point=gap
        )

    # Check for excessive variance
    std_years = statistics.stdev(estimates_years)
    if std_years > 8:
        return ClusterError(
            cluster_id=cluster.id,
            error_type='high_variance',
            message=f'High birth year variance: {std_years:.1f} years',
            suggested_action='review_faces'
        )

    # Check gender consistency
    gender_counts = Counter(d.gender for d in cluster.detections if d.gender)
    if len(gender_counts) > 1:
        m_pct = gender_counts.get('M', 0) / sum(gender_counts.values())
        if 0.3 < m_pct < 0.7:
            return ClusterError(
                cluster_id=cluster.id,
                error_type='mixed_gender',
                message=f'Mixed gender in cluster: {dict(gender_counts)}',
                suggested_action='review_faces'
            )

    return None
```

---

## Incremental Updates

Following the tiered model, incremental updates are designed for efficiency:

### Tier 1: Detection Processing (Hot Path)

```python
def on_detection_created(detection: PersonDetection, photo: Photo):
    """
    Cache birth estimate when detection is created.
    Called during detection/age_gender stage. O(1).
    """
    if detection.age_estimate and photo.captured_at:
        birth_estimate = photo.captured_at - timedelta(
            days=detection.age_estimate * 365.25
        )
        weight = compute_detection_weight(detection)

        upsert_detection_birth_estimate(
            detection_id=detection.id,
            photo_date=photo.captured_at,
            age_estimate=detection.age_estimate,
            birth_date_estimate=birth_estimate,
            weight=weight,
        )
```

### Tier 1: Cluster Assignment (Hot Path)

```python
def on_detection_clustered(detection: PersonDetection, cluster: Cluster):
    """
    Mark person stale and record co-occurrences.
    Called during clustering stage. O(1) + O(k²) for k persons in photo.
    """
    # Mark person as needing recomputation
    if cluster.person_id:
        mark_person_stale(cluster.person_id)

    # Record co-occurrences with other persons in same photo
    persons_in_photo = get_persons_in_photo(detection.photo_id)
    if cluster.person_id and len(persons_in_photo) > 1:
        photo_date = get_photo_date(detection.photo_id)
        for other_person_id in persons_in_photo:
            if other_person_id != cluster.person_id:
                record_cooccurrence(
                    cluster.person_id,
                    other_person_id,
                    detection.photo_id,
                    photo_date
                )
```

### Tier 2: On-Demand Aggregation

```python
def get_person_estimates(person_id: int) -> PersonEstimates:
    """
    Get person estimates, recomputing if stale.
    Called by UI/API. O(n_detections).
    """
    person = get_person(person_id)

    if person.birth_estimate_stale:
        # Aggregate from cached detection estimates
        estimates = get_birth_estimates_for_person(person_id)

        if estimates:
            person.estimated_birth_date = weighted_median(estimates)
            person.birth_date_confidence = compute_confidence(estimates)
        else:
            person.estimated_birth_date = None
            person.birth_date_confidence = 0.0

        # Also update gender
        if not person.ground_truth_gender:
            person.gender, person.gender_confidence = estimate_gender(person)

        person.birth_estimate_stale = False
        person.birth_estimate_computed_at = datetime.now(timezone.utc)
        save_person(person)

    return person
```

### Tier 3: Propagation (Explicit Only)

```python
def on_ground_truth_set(person: Person, ground_truth: GroundTruth):
    """
    When user sets ground truth, propagate to nearby persons.
    Explicit action only, not part of hot path.
    """
    apply_ground_truth(person, ground_truth)

    # Scoped propagation - only persons within 2 hops
    propagate_from_anchor(person.id, max_hops=2)
```

---

## Database Schema Changes

### Migration: 018_add_person_birth_estimation.sql

```sql
-- ============================================================================
-- Person birth date and gender estimation fields
-- ============================================================================

-- Extend person table with estimation fields
ALTER TABLE person ADD COLUMN IF NOT EXISTS estimated_birth_date DATE;
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_date_confidence REAL;
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_date_source VARCHAR(20);
-- birth_date_source: 'estimated' | 'user' | 'derived'

-- Ground truth (user-provided, immutable)
ALTER TABLE person ADD COLUMN IF NOT EXISTS ground_truth_birth_date DATE;
ALTER TABLE person ADD COLUMN IF NOT EXISTS ground_truth_gender VARCHAR(1);
ALTER TABLE person ADD COLUMN IF NOT EXISTS ground_truth_notes TEXT;
ALTER TABLE person ADD COLUMN IF NOT EXISTS ground_truth_set_at TIMESTAMPTZ;
ALTER TABLE person ADD COLUMN IF NOT EXISTS ground_truth_set_by VARCHAR(100);

-- Anchor flag for propagation
ALTER TABLE person ADD COLUMN IF NOT EXISTS is_anchor BOOLEAN DEFAULT FALSE;

-- Gender source tracking
ALTER TABLE person ADD COLUMN IF NOT EXISTS gender_source VARCHAR(20);
-- gender_source: 'estimated' | 'user'

-- Staleness tracking for lazy recomputation (efficiency)
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_estimate_stale BOOLEAN DEFAULT TRUE;
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_estimate_computed_at TIMESTAMPTZ;

-- ============================================================================
-- Co-occurrence tracking for constraint propagation
-- ============================================================================

CREATE TABLE IF NOT EXISTS person_cooccurrence (
    id BIGSERIAL PRIMARY KEY,
    collection_id INTEGER NOT NULL,
    photo_id BIGINT NOT NULL,
    person_id_1 INTEGER NOT NULL,
    person_id_2 INTEGER NOT NULL,
    photo_date DATE,  -- Cached from metadata for faster queries
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (photo_id) REFERENCES photo(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id_1) REFERENCES person(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id_2) REFERENCES person(id) ON DELETE CASCADE,

    -- Canonical ordering: person_id_1 < person_id_2
    CHECK (person_id_1 < person_id_2),
    UNIQUE (photo_id, person_id_1, person_id_2)
);

CREATE INDEX IF NOT EXISTS idx_cooccurrence_person_1
    ON person_cooccurrence(person_id_1);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_person_2
    ON person_cooccurrence(person_id_2);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_collection
    ON person_cooccurrence(collection_id);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_photo_date
    ON person_cooccurrence(photo_date) WHERE photo_date IS NOT NULL;

-- ============================================================================
-- Materialized view for efficient co-occurrence queries (refreshed weekly)
-- ============================================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS person_cooccurrence_summary AS
SELECT
    collection_id,
    person_id_1,
    person_id_2,
    COUNT(*) as photo_count,
    MIN(photo_date) as first_seen_together,
    MAX(photo_date) as last_seen_together
FROM person_cooccurrence
WHERE photo_date IS NOT NULL
GROUP BY collection_id, person_id_1, person_id_2;

CREATE UNIQUE INDEX IF NOT EXISTS idx_cooccurrence_summary_pk
    ON person_cooccurrence_summary(collection_id, person_id_1, person_id_2);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_summary_person_1
    ON person_cooccurrence_summary(person_id_1);
CREATE INDEX IF NOT EXISTS idx_cooccurrence_summary_person_2
    ON person_cooccurrence_summary(person_id_2);

-- Refresh command (run weekly):
-- REFRESH MATERIALIZED VIEW CONCURRENTLY person_cooccurrence_summary;

-- ============================================================================
-- Birth date estimate cache per detection
-- Stores per-detection birth estimates for efficient aggregation
-- ============================================================================

CREATE TABLE IF NOT EXISTS detection_birth_estimate (
    detection_id BIGINT PRIMARY KEY,
    photo_date DATE NOT NULL,
    age_estimate REAL NOT NULL,
    birth_date_estimate DATE NOT NULL,
    weight REAL NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (detection_id) REFERENCES person_detection(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_detection_birth_estimate_date
    ON detection_birth_estimate(birth_date_estimate);

-- ============================================================================
-- Cluster error tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS cluster_error (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER NOT NULL,
    collection_id INTEGER NOT NULL,
    error_type VARCHAR(50) NOT NULL,
    -- 'likely_mixed_identity', 'high_variance', 'mixed_gender'
    message TEXT,
    suggested_action VARCHAR(50),
    suggested_split_point REAL,
    status VARCHAR(20) DEFAULT 'open',
    -- 'open', 'resolved', 'ignored'
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (cluster_id) REFERENCES cluster(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_cluster_error_cluster
    ON cluster_error(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_error_status
    ON cluster_error(status) WHERE status = 'open';

-- ============================================================================
-- Estimation history for debugging/analysis
-- ============================================================================

CREATE TABLE IF NOT EXISTS person_estimate_history (
    id BIGSERIAL PRIMARY KEY,
    person_id INTEGER NOT NULL,
    estimated_birth_date DATE,
    birth_date_confidence REAL,
    gender VARCHAR(1),
    gender_confidence REAL,
    estimation_source VARCHAR(50),
    -- 'initial', 'propagation_iter_N', 'user_override', 'new_detection'
    n_samples INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    FOREIGN KEY (person_id) REFERENCES person(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_estimate_history_person
    ON person_estimate_history(person_id);
```

---

## Configuration

```bash
# Birth date estimation
BIRTH_ESTIMATE_MIN_DETECTIONS=1       # Minimum detections for estimate
BIRTH_ESTIMATE_ANCHOR_THRESHOLD=0.7   # Confidence threshold to be an anchor
BIRTH_ESTIMATE_MAX_MAD_YEARS=5        # MAD above this = 0 confidence
BIRTH_ESTIMATE_PROPAGATION_ITERS=10   # Max iterations for propagation

# Cluster error detection
CLUSTER_ERROR_BIMODAL_THRESHOLD=5     # Years gap to detect bimodality
CLUSTER_ERROR_HIGH_VARIANCE=8         # Std dev years to flag high variance
CLUSTER_ERROR_GENDER_THRESHOLD=0.3    # Gender ratio threshold
```

---

## CLI Commands

### Maintenance CLI Extension

```bash
# Estimate birth dates for all persons
uv run photodb-maintenance estimate-birth-dates

# Estimate for specific collection
uv run photodb-maintenance estimate-birth-dates --collection-id 1

# Force re-estimation (ignore cached values)
uv run photodb-maintenance estimate-birth-dates --force

# Run constraint propagation only
uv run photodb-maintenance propagate-birth-dates

# Detect cluster errors
uv run photodb-maintenance detect-cluster-errors

# Show estimation statistics
uv run photodb-maintenance birth-stats
```

---

## Task List

### Task 1: Database Migration

**Files:**

- Create: `migrations/018_add_person_birth_estimation.sql`
- Create: `migrations/018_add_person_birth_estimation_rollback.sql`

**Steps:**

1. Write migration SQL (schema above)
2. Write rollback migration
3. Apply migration: `psql $DATABASE_URL -f migrations/018_add_person_birth_estimation.sql`
4. Verify with: `\d person` and `\d person_cooccurrence`
5. Commit

---

### Task 2: Update Database Models

**Files:**

- Modify: `src/photodb/database/models.py`

**Steps:**

1. Add new fields to `Person` dataclass:
   - `estimated_birth_date: Optional[date]`
   - `birth_date_confidence: Optional[float]`
   - `birth_date_source: Optional[str]`
   - `ground_truth_birth_date: Optional[date]`
   - `ground_truth_gender: Optional[str]`
   - `ground_truth_notes: Optional[str]`
   - `is_anchor: bool`
   - `gender_source: Optional[str]`
   - `birth_estimate_stale: bool`
   - `birth_estimate_computed_at: Optional[datetime]`
2. Add `PersonCooccurrence` dataclass
3. Add `DetectionBirthEstimate` dataclass
4. Add `ClusterError` dataclass
5. Commit

---

### Task 3: Add Repository Methods

**Files:**

- Modify: `src/photodb/database/pg_repository.py`

**Steps:**

1. Add co-occurrence methods:
   - `record_cooccurrence(person_id_1, person_id_2, photo_id, photo_date)`
   - `get_cooccurrences_for_person(person_id)`
   - `get_all_cooccurrences(collection_id)`
2. Add detection birth estimate methods:
   - `upsert_detection_birth_estimate(detection_id, ...)`
   - `get_birth_estimates_for_person(person_id)`
3. Add cluster error methods:
   - `create_cluster_error(cluster_id, error_type, ...)`
   - `get_open_cluster_errors(collection_id)`
   - `resolve_cluster_error(error_id, resolved_by)`
4. Add person update methods:
   - `update_person_birth_estimate(person_id, birth_date, confidence, source)`
   - `set_person_ground_truth(person_id, birth_date, gender, notes, set_by)`
   - `get_anchor_persons(collection_id)`
   - `mark_person_stale(person_id)`
5. Add `get_persons_in_photo(photo_id)` method
6. Commit

---

### Task 4: Create Estimation Utilities

**Files:**

- Create: `src/photodb/utils/birth_estimator.py`
- Create: `tests/test_birth_estimator.py`

**Steps:**

1. Implement `weighted_median(estimates)` function
2. Implement `median_absolute_deviation(values)` function
3. Implement `compute_detection_weight(detection)` function
4. Implement `estimate_gender(person, detections)` function
5. Implement `initial_birth_date_estimate(person, detections, photo_dates)` function
6. Implement `compute_birth_date_confidence(...)` function
7. Write unit tests for each function
8. Run tests: `uv run pytest tests/test_birth_estimator.py -v`
9. Commit

---

### Task 5: Create Constraint Propagation

**Files:**

- Create: `src/photodb/utils/birth_propagator.py`
- Create: `tests/test_birth_propagator.py`

**Steps:**

1. Implement `Constraint` dataclass
2. Implement `solve_constraints(constraints)` function
3. Implement `propagate_constraints(repository, collection_id, max_iterations)`
4. Implement `propagate_from_anchor(repository, anchor_person_id, max_hops)`
5. Implement convergence detection
6. Write unit tests with mock data
7. Run tests
8. Commit

---

### Task 6: Create Cluster Error Detection

**Files:**

- Create: `src/photodb/utils/cluster_error_detector.py`
- Create: `tests/test_cluster_error_detector.py`

**Steps:**

1. Implement `is_bimodal(values, threshold)` function using kernel density estimation or histogram gap detection
2. Implement `find_gap(values)` to locate split point
3. Implement `detect_cluster_errors(cluster, detections, photo_dates)` function
4. Implement `scan_all_clusters(repository, collection_id)` to check all clusters
5. Write unit tests
6. Run tests
7. Commit

---

### Task 7: Integrate Hot Path (Tier 1)

**Files:**

- Modify: `src/photodb/stages/age_gender.py`
- Modify: `src/photodb/stages/clustering.py`

**Steps:**

1. In age_gender stage: Call `upsert_detection_birth_estimate()` after setting age estimate
2. In clustering stage: Call `mark_person_stale()` when detection assigned to cluster
3. In clustering stage: Call `record_cooccurrence()` for multi-person photos
4. Ensure all hot path operations are O(1) or O(k²) for k persons in photo
5. Test integration
6. Commit

---

### Task 8: Create Lazy Aggregation (Tier 2)

**Files:**

- Create: `src/photodb/utils/person_estimator.py`
- Create: `tests/test_person_estimator.py`

**Steps:**

1. Implement `get_person_with_estimates(repository, person_id)` with lazy recomputation
2. Use cached detection_birth_estimate rows for aggregation
3. Mark person as not stale after recomputation
4. Write tests for stale/fresh behavior
5. Commit

---

### Task 9: Add CLI Commands and Maintenance

**Files:**

- Modify: `src/photodb/cli_maintenance.py`
- Modify: `src/photodb/utils/maintenance.py`

**Steps:**

1. Add `estimate-birth-dates` command (forces Tier 2 for all persons)
2. Add `propagate-birth-dates` command (runs Tier 3)
3. Add `detect-cluster-errors` command
4. Add `birth-stats` command
5. Add `set-ground-truth` command for user overrides
6. Add `refresh-cooccurrence-summary` to weekly maintenance (REFRESH MATERIALIZED VIEW)
7. Test commands manually
8. Commit

---

### Task 10: Update Documentation

**Files:**

- Modify: `CLAUDE.md`
- Modify: `docs/DESIGN.md`

**Steps:**

1. Add birth estimation configuration to CLAUDE.md
2. Add birth estimation tiered model to DESIGN.md
3. Document CLI commands
4. Document ground truth API
5. Document efficiency considerations
6. Commit

---

## Summary

| Component | Purpose |
| --------- | ------- |
| `birth_estimator.py` | Core estimation algorithms (weighted median, MAD, confidence) |
| `birth_propagator.py` | Constraint propagation logic (Tier 3) |
| `person_estimator.py` | Lazy aggregation with staleness (Tier 2) |
| `cluster_error_detector.py` | Cluster error detection via bimodal analysis |
| `person_cooccurrence` table | Tracks who appears with whom |
| `person_cooccurrence_summary` view | Materialized summary for fast queries |
| `detection_birth_estimate` table | Caches per-detection estimates (Tier 1) |
| `cluster_error` table | Tracks detected errors for review |

### Key Design Decisions

1. **Three-tiered computation model** for efficiency at scale
2. **Lazy aggregation** - only recompute when stale and requested
3. **Hot path is O(1)** - photo processing never triggers expensive operations
4. **Weighted median** for robustness to outliers (MiVOLO can be wildly wrong)
5. **Scoped propagation** - anchors only propagate within N hops
6. **Ground truth as hard constraints** that override estimates
7. **Cluster error detection** to surface data quality issues
8. **Confidence scores** at every level for transparency

### Expected Outcomes

- **Efficient at scale**: 100k+ photos, 1000+ persons
- More accurate birth year estimates than per-detection aggregation alone
- Automatic detection of cluster errors via bimodal distributions
- User-provided ground truth propagates to improve other estimates
- Confidence scores help users identify which estimates need verification

---

## Future Enhancements

1. **Age estimation model improvements**: Fine-tune MiVOLO on family photos
2. **Relationship inference**: Use age differences to infer parent/child/sibling
3. **Event-based anchoring**: Use known events (weddings, graduations) as additional constraints
4. **Active learning**: Prioritize which persons need user verification
5. **UI for ground truth**: Web interface for setting birth dates and reviewing cluster errors
