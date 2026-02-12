# Genealogical Relationships Schema Design

**Date:** 2026-02-11
**Status:** Draft
**Goal:** Add genealogical relationship modeling to person entities, enabling family tree display, age inference, and photo ordering by birth order.

## Overview

This design adds genealogical relationships to PhotoDB's person entities. The schema supports:

1. **Family tree display** — Centered on any person, traversing ancestors, descendants, siblings, and partners
2. **Sparse/incomplete trees** — Using placeholder persons for unknown intermediates (e.g., "Alice's mother" when we know Alice's grandmother but not her parent)
3. **Birth order** — Pairwise partial ordering among siblings/cousins, independent of exact birth dates
4. **Age inference** — Propagating birth year constraints through genealogical relationships to aid the age/gender detection pipeline

## Design Principles

### Placeholder Persons

Rather than separate tables for sparse relationships (e.g., "grandparent without known parent"), we use **placeholder person rows**:

- Names are nullable; placeholders have `is_placeholder = true`
- `placeholder_description` provides context (e.g., "John Smith's father")
- All relationships flow through uniform `person_parent` links
- UI can choose to show/hide placeholders
- Placeholders can be progressively refined as information is learned

### Materialized Closures with Recursive CTEs

Transitive closures are materialized for query efficiency during photo processing, but populated using PostgreSQL recursive CTEs (not explicit loops):

- `person_ancestor_closure` — All ancestor relationships with generation distance
- `person_birth_order_closure` — All "older than" relationships (direct + genealogy-implied + transitive)

Refresh functions use `WITH RECURSIVE` for clarity and performance.

## Schema Changes

### Person Table Modifications

```sql
-- Make names nullable for placeholders
ALTER TABLE person ALTER COLUMN first_name DROP NOT NULL;
ALTER TABLE person ALTER COLUMN last_name DROP NOT NULL;

-- Placeholder support
ALTER TABLE person ADD COLUMN is_placeholder BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE person ADD COLUMN placeholder_description TEXT;

-- Gender (for genealogical role assignment)
ALTER TABLE person ADD COLUMN gender CHAR(1) CHECK (gender IN ('M', 'F', 'U'));

-- Birth year constraints (for age inference)
ALTER TABLE person ADD COLUMN birth_year_min INT;
ALTER TABLE person ADD COLUMN birth_year_max INT;
ALTER TABLE person ADD COLUMN birth_year_source TEXT CHECK (birth_year_source IN ('exact', 'year', 'estimated', 'inferred'));

-- Exact birth date (when fully known)
ALTER TABLE person ADD COLUMN birth_date DATE;
ALTER TABLE person ADD CONSTRAINT person_birth_date_year_check
    CHECK (
        birth_date IS NULL
        OR (birth_year_min = EXTRACT(YEAR FROM birth_date)::INT
            AND birth_year_max = EXTRACT(YEAR FROM birth_date)::INT)
    );

-- Index for filtering
CREATE INDEX person_placeholder_idx ON person(is_placeholder) WHERE is_placeholder;
```

### Birth Date Handling

The schema supports both exact birth dates and uncertain birth year ranges:

| Knowledge Level | birth_date | birth_year_min | birth_year_max | birth_year_source |
|-----------------|------------|----------------|----------------|-------------------|
| Exact date known | 1985-03-15 | 1985 | 1985 | 'exact' |
| Year only known | NULL | 1985 | 1985 | 'year' |
| Estimated from photos | NULL | 1983 | 1987 | 'estimated' |
| Inferred from genealogy | NULL | 1970 | 1990 | 'inferred' |

**Reconciliation rules:**
- When `birth_date` is set, `birth_year_min` and `birth_year_max` must equal its year (enforced by constraint)
- Genealogy constraint propagation skips persons with `birth_year_source` of 'exact' or 'year'
- Age estimates from photos can narrow the range but won't override known dates/years

### New Tables

```sql
-- ============================================
-- GENEALOGICAL RELATIONSHIPS
-- ============================================

-- Parent-child relationships (the fundamental unit)
CREATE TABLE person_parent (
    person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    parent_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    parent_role TEXT CHECK (parent_role IN ('mother', 'father', 'parent')),
    is_biological BOOLEAN NOT NULL DEFAULT true,
    source TEXT CHECK (source IN ('user', 'inferred', 'imported')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (person_id, parent_id),
    CHECK (person_id != parent_id)
);

CREATE INDEX person_parent_parent_idx ON person_parent(parent_id);

-- Partnerships (marriages, relationships)
CREATE TABLE person_partnership (
    id BIGSERIAL PRIMARY KEY,
    person1_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    person2_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    partnership_type TEXT CHECK (partnership_type IN ('married', 'partner', 'divorced', 'separated')),
    start_year INT,
    end_year INT,
    is_current BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (person1_id < person2_id)
);

CREATE UNIQUE INDEX person_partnership_unique_idx
    ON person_partnership(person1_id, person2_id, COALESCE(start_year, 0));
CREATE INDEX person_partnership_p2_idx ON person_partnership(person2_id);

-- Pairwise birth order (partial ordering)
-- Only stores KNOWN orderings; transitive implications materialized separately
CREATE TABLE person_birth_order (
    older_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    younger_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    source TEXT CHECK (source IN ('exact_dates', 'user', 'inferred', 'photo_evidence')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (older_person_id, younger_person_id),
    CHECK (older_person_id != younger_person_id)
);

CREATE INDEX person_birth_order_younger_idx ON person_birth_order(younger_person_id);

-- Explicit non-relationships (sparse: only stores direct assertions)
-- Transitive implications computed via recursive queries against ancestor closure
CREATE TABLE person_not_related (
    person1_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    person2_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    source TEXT CHECK (source IN ('user', 'inferred')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (person1_id, person2_id),
    CHECK (person1_id < person2_id)  -- Canonical ordering to avoid duplicates
);

CREATE INDEX person_not_related_p2_idx ON person_not_related(person2_id);

-- ============================================
-- MATERIALIZED CLOSURES
-- ============================================

-- Transitive closure of ancestor relationships
CREATE TABLE person_ancestor_closure (
    descendant_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    ancestor_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    distance INT NOT NULL CHECK (distance >= 1),
    PRIMARY KEY (descendant_id, ancestor_id)
);

CREATE INDEX pac_ancestor_idx ON person_ancestor_closure(ancestor_id);
CREATE INDEX pac_distance_idx ON person_ancestor_closure(distance);

-- Transitive closure of birth order (includes genealogy-implied)
CREATE TABLE person_birth_order_closure (
    older_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    younger_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    inference_type TEXT CHECK (inference_type IN ('direct', 'genealogy', 'transitive')),
    PRIMARY KEY (older_person_id, younger_person_id)
);

CREATE INDEX pboc_younger_idx ON person_birth_order_closure(younger_person_id);
```

### Views

```sql
-- Siblings derived from shared parents
CREATE VIEW person_siblings AS
SELECT
    p1.person_id,
    p2.person_id AS sibling_id,
    CASE
        WHEN COUNT(DISTINCT p1.parent_id) >= 2 THEN 'full'
        ELSE 'half'
    END AS sibling_type,
    ARRAY_AGG(DISTINCT p1.parent_id) AS shared_parent_ids
FROM person_parent p1
JOIN person_parent p2
    ON p1.parent_id = p2.parent_id
    AND p1.person_id != p2.person_id
GROUP BY p1.person_id, p2.person_id;

-- Non-placeholder persons
CREATE VIEW person_known AS
SELECT * FROM person WHERE NOT is_placeholder;
```

## Functions

### Display Name

```sql
CREATE OR REPLACE FUNCTION person_display_name(p person)
RETURNS TEXT AS $$
BEGIN
    IF p.first_name IS NOT NULL OR p.last_name IS NOT NULL THEN
        RETURN TRIM(COALESCE(p.first_name || ' ', '') || COALESCE(p.last_name, ''));
    ELSIF p.placeholder_description IS NOT NULL THEN
        RETURN p.placeholder_description;
    ELSE
        RETURN 'Unknown #' || p.id;
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
```

### Placeholder Creation

```sql
-- Create a placeholder parent for a person
CREATE OR REPLACE FUNCTION create_placeholder_parent(
    child_id BIGINT,
    role TEXT DEFAULT 'parent'
) RETURNS BIGINT AS $$
DECLARE
    child_rec person;
    new_parent_id BIGINT;
BEGIN
    SELECT * INTO child_rec FROM person WHERE id = child_id;

    INSERT INTO person (is_placeholder, placeholder_description)
    VALUES (true, person_display_name(child_rec) || '''s ' || role)
    RETURNING id INTO new_parent_id;

    INSERT INTO person_parent (person_id, parent_id, parent_role, source)
    VALUES (child_id, new_parent_id, role, 'user');

    RETURN new_parent_id;
END;
$$ LANGUAGE plpgsql;

-- Link two people as siblings (creates placeholder parent if needed)
CREATE OR REPLACE FUNCTION link_siblings(
    person1_id BIGINT,
    person2_id BIGINT,
    sibling_type TEXT DEFAULT 'full'
) RETURNS BIGINT AS $$
DECLARE
    shared_parent_id BIGINT;
    p1_rec person;
    p2_rec person;
BEGIN
    -- Check if they already share a parent
    SELECT p1.parent_id INTO shared_parent_id
    FROM person_parent p1
    JOIN person_parent p2 ON p1.parent_id = p2.parent_id
    WHERE p1.person_id = person1_id AND p2.person_id = person2_id
    LIMIT 1;

    IF shared_parent_id IS NOT NULL THEN
        RETURN shared_parent_id;
    END IF;

    -- Create placeholder parent(s)
    SELECT * INTO p1_rec FROM person WHERE id = person1_id;
    SELECT * INTO p2_rec FROM person WHERE id = person2_id;

    INSERT INTO person (is_placeholder, placeholder_description)
    VALUES (true, 'Parent of ' || person_display_name(p1_rec) || ' & ' || person_display_name(p2_rec))
    RETURNING id INTO shared_parent_id;

    INSERT INTO person_parent (person_id, parent_id, parent_role, source)
    VALUES
        (person1_id, shared_parent_id, 'parent', 'user'),
        (person2_id, shared_parent_id, 'parent', 'user');

    -- For full siblings, create second placeholder parent
    IF sibling_type = 'full' THEN
        INSERT INTO person (is_placeholder, placeholder_description)
        VALUES (true, 'Parent of ' || person_display_name(p1_rec) || ' & ' || person_display_name(p2_rec))
        RETURNING id INTO shared_parent_id;

        INSERT INTO person_parent (person_id, parent_id, parent_role, source)
        VALUES
            (person1_id, shared_parent_id, 'parent', 'user'),
            (person2_id, shared_parent_id, 'parent', 'user');
    END IF;

    RETURN shared_parent_id;
END;
$$ LANGUAGE plpgsql;
```

### Closure Refresh (Using Recursive CTEs)

```sql
-- Refresh ancestor closure using recursive CTE
CREATE OR REPLACE FUNCTION refresh_ancestor_closure() RETURNS void AS $$
BEGIN
    TRUNCATE person_ancestor_closure;

    INSERT INTO person_ancestor_closure (descendant_id, ancestor_id, distance)
    WITH RECURSIVE ancestors AS (
        -- Base case: direct parents (distance 1)
        SELECT person_id AS descendant_id, parent_id AS ancestor_id, 1 AS distance
        FROM person_parent

        UNION

        -- Recursive case: ancestors of ancestors
        SELECT a.descendant_id, pp.parent_id AS ancestor_id, a.distance + 1
        FROM ancestors a
        JOIN person_parent pp ON pp.person_id = a.ancestor_id
        WHERE a.distance < 50  -- reasonable max depth
    )
    SELECT descendant_id, ancestor_id, MIN(distance)
    FROM ancestors
    GROUP BY descendant_id, ancestor_id;
END;
$$ LANGUAGE plpgsql;

-- Refresh birth order closure using recursive CTE
-- Must be called AFTER refresh_ancestor_closure()
CREATE OR REPLACE FUNCTION refresh_birth_order_closure() RETURNS void AS $$
BEGIN
    TRUNCATE person_birth_order_closure;

    INSERT INTO person_birth_order_closure (older_person_id, younger_person_id, inference_type)
    WITH RECURSIVE ordered AS (
        -- Direct birth order records
        SELECT older_person_id, younger_person_id,
               'direct'::TEXT AS inference_type
        FROM person_birth_order

        UNION

        -- Ancestors are older than descendants (from closure table)
        SELECT ancestor_id AS older_person_id, descendant_id AS younger_person_id,
               'genealogy'::TEXT
        FROM person_ancestor_closure

        UNION

        -- Transitive: if A > B and B > C, then A > C
        SELECT o.older_person_id, bo.younger_person_id,
               'transitive'::TEXT
        FROM ordered o
        JOIN person_birth_order bo ON bo.older_person_id = o.younger_person_id
        WHERE o.older_person_id != bo.younger_person_id
    )
    SELECT older_person_id, younger_person_id,
           -- Prefer direct > genealogy > transitive
           (ARRAY_AGG(inference_type ORDER BY
               CASE inference_type
                   WHEN 'direct' THEN 1
                   WHEN 'genealogy' THEN 2
                   ELSE 3
               END
           ))[1]
    FROM ordered
    GROUP BY older_person_id, younger_person_id;
END;
$$ LANGUAGE plpgsql;

-- Refresh all closures (convenience function)
CREATE OR REPLACE FUNCTION refresh_genealogy_closures() RETURNS void AS $$
BEGIN
    PERFORM refresh_ancestor_closure();
    PERFORM refresh_birth_order_closure();
END;
$$ LANGUAGE plpgsql;
```

### Family Tree Query

```sql
CREATE OR REPLACE FUNCTION get_family_tree(
    center_id BIGINT,
    max_generations INT DEFAULT 3,
    include_placeholders BOOLEAN DEFAULT true
)
RETURNS TABLE (
    person_id BIGINT,
    display_name TEXT,
    relation TEXT,
    generation_offset INT,
    is_placeholder BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH tree AS (
        -- Self
        SELECT center_id AS pid, 'self'::TEXT AS rel, 0 AS gen

        UNION ALL

        -- Ancestors (from closure)
        SELECT pac.ancestor_id,
            CASE pac.distance
                WHEN 1 THEN 'parent'
                WHEN 2 THEN 'grandparent'
                WHEN 3 THEN 'great-grandparent'
                ELSE 'ancestor-' || pac.distance
            END,
            -pac.distance
        FROM person_ancestor_closure pac
        WHERE pac.descendant_id = center_id
          AND pac.distance <= max_generations

        UNION ALL

        -- Descendants (from closure)
        SELECT pac.descendant_id,
            CASE pac.distance
                WHEN 1 THEN 'child'
                WHEN 2 THEN 'grandchild'
                WHEN 3 THEN 'great-grandchild'
                ELSE 'descendant-' || pac.distance
            END,
            pac.distance
        FROM person_ancestor_closure pac
        WHERE pac.ancestor_id = center_id
          AND pac.distance <= max_generations

        UNION ALL

        -- Siblings
        SELECT ps.sibling_id, ps.sibling_type || '-sibling', 0
        FROM person_siblings ps
        WHERE ps.person_id = center_id

        UNION ALL

        -- Partners
        SELECT
            CASE WHEN pp.person1_id = center_id THEN pp.person2_id ELSE pp.person1_id END,
            COALESCE(pp.partnership_type, 'partner'),
            0
        FROM person_partnership pp
        WHERE pp.person1_id = center_id OR pp.person2_id = center_id
    )
    SELECT t.pid, person_display_name(p.*), t.rel, t.gen, p.is_placeholder
    FROM tree t
    JOIN person p ON p.id = t.pid
    WHERE include_placeholders OR NOT p.is_placeholder
    ORDER BY t.gen, t.rel, t.pid;
END;
$$ LANGUAGE plpgsql STABLE;
```

### Birth Year Constraint Propagation

```sql
CREATE OR REPLACE FUNCTION propagate_birth_year_constraints(
    min_parent_gap INT DEFAULT 15,
    max_parent_gap INT DEFAULT 60
) RETURNS INT AS $$
DECLARE
    total_updates INT := 0;
    batch_updates INT;
BEGIN
    LOOP
        batch_updates := 0;

        -- Parent's max year constrained by child's min year
        WITH updates AS (
            UPDATE person p SET
                birth_year_max = LEAST(p.birth_year_max, sub.constrained_max),
                birth_year_source = COALESCE(p.birth_year_source, 'inferred')
            FROM (
                SELECT pp.parent_id, MIN(c.birth_year_min) - min_parent_gap AS constrained_max
                FROM person_parent pp
                JOIN person c ON c.id = pp.person_id
                WHERE c.birth_year_min IS NOT NULL
                GROUP BY pp.parent_id
            ) sub
            WHERE p.id = sub.parent_id
              AND (p.birth_year_max IS NULL OR p.birth_year_max > sub.constrained_max)
              AND (p.birth_year_source IS NULL OR p.birth_year_source NOT IN ('exact', 'year'))
            RETURNING 1
        )
        SELECT COUNT(*) INTO batch_updates FROM updates;

        -- Child's min year constrained by parent's min year
        WITH updates AS (
            UPDATE person p SET
                birth_year_min = GREATEST(p.birth_year_min, sub.constrained_min),
                birth_year_source = COALESCE(p.birth_year_source, 'inferred')
            FROM (
                SELECT pp.person_id, MAX(par.birth_year_min) + min_parent_gap AS constrained_min
                FROM person_parent pp
                JOIN person par ON par.id = pp.parent_id
                WHERE par.birth_year_min IS NOT NULL
                GROUP BY pp.person_id
            ) sub
            WHERE p.id = sub.person_id
              AND (p.birth_year_min IS NULL OR p.birth_year_min < sub.constrained_min)
              AND (p.birth_year_source IS NULL OR p.birth_year_source NOT IN ('exact', 'year'))
            RETURNING 1
        )
        SELECT COUNT(*) + batch_updates INTO batch_updates FROM updates;

        -- Child's max year constrained by parent's max year
        WITH updates AS (
            UPDATE person p SET
                birth_year_max = LEAST(p.birth_year_max, sub.constrained_max),
                birth_year_source = COALESCE(p.birth_year_source, 'inferred')
            FROM (
                SELECT pp.person_id, MIN(par.birth_year_max) + max_parent_gap AS constrained_max
                FROM person_parent pp
                JOIN person par ON par.id = pp.parent_id
                WHERE par.birth_year_max IS NOT NULL
                GROUP BY pp.person_id
            ) sub
            WHERE p.id = sub.person_id
              AND (p.birth_year_max IS NULL OR p.birth_year_max > sub.constrained_max)
              AND (p.birth_year_source IS NULL OR p.birth_year_source NOT IN ('exact', 'year'))
            RETURNING 1
        )
        SELECT COUNT(*) + batch_updates INTO batch_updates FROM updates;

        total_updates := total_updates + batch_updates;
        EXIT WHEN batch_updates = 0;
    END LOOP;

    RETURN total_updates;
END;
$$ LANGUAGE plpgsql;
```

### Non-Relationship Queries

These functions determine whether two persons are unrelated, using the sparse `person_not_related` table combined with transitive implications via the ancestor closure.

```sql
-- Check if two persons are explicitly or transitively unrelated
-- Returns true if:
--   1. Direct entry in person_not_related, OR
--   2. Person A is unrelated to any relative of Person B (or vice versa)
CREATE OR REPLACE FUNCTION are_persons_unrelated(
    person_a BIGINT,
    person_b BIGINT
) RETURNS BOOLEAN AS $$
BEGIN
    -- Check direct non-relationship (canonical ordering)
    IF EXISTS (
        SELECT 1 FROM person_not_related
        WHERE (person1_id = LEAST(person_a, person_b)
           AND person2_id = GREATEST(person_a, person_b))
    ) THEN
        RETURN true;
    END IF;

    -- Check transitive: A unrelated to any ancestor/descendant of B
    IF EXISTS (
        WITH b_relatives AS (
            -- B's ancestors
            SELECT ancestor_id AS relative_id FROM person_ancestor_closure
            WHERE descendant_id = person_b
            UNION
            -- B's descendants
            SELECT descendant_id FROM person_ancestor_closure
            WHERE ancestor_id = person_b
            UNION
            -- B itself
            SELECT person_b
        )
        SELECT 1 FROM person_not_related pnr
        JOIN b_relatives br ON (
            (pnr.person1_id = LEAST(person_a, br.relative_id)
             AND pnr.person2_id = GREATEST(person_a, br.relative_id))
        )
    ) THEN
        RETURN true;
    END IF;

    -- Check transitive: B unrelated to any ancestor/descendant of A
    IF EXISTS (
        WITH a_relatives AS (
            SELECT ancestor_id AS relative_id FROM person_ancestor_closure
            WHERE descendant_id = person_a
            UNION
            SELECT descendant_id FROM person_ancestor_closure
            WHERE ancestor_id = person_a
            UNION
            SELECT person_a
        )
        SELECT 1 FROM person_not_related pnr
        JOIN a_relatives ar ON (
            (pnr.person1_id = LEAST(person_b, ar.relative_id)
             AND pnr.person2_id = GREATEST(person_b, ar.relative_id))
        )
    ) THEN
        RETURN true;
    END IF;

    RETURN false;
END;
$$ LANGUAGE plpgsql STABLE;

-- Get all persons that are unrelated to the given person (direct + transitive)
CREATE OR REPLACE FUNCTION get_unrelated_persons(
    target_person_id BIGINT
) RETURNS TABLE (
    person_id BIGINT,
    is_direct BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH target_relatives AS (
        -- All relatives of target (ancestors, descendants, self)
        SELECT ancestor_id AS relative_id FROM person_ancestor_closure
        WHERE descendant_id = target_person_id
        UNION
        SELECT descendant_id FROM person_ancestor_closure
        WHERE ancestor_id = target_person_id
        UNION
        SELECT target_person_id
    ),
    direct_unrelated AS (
        -- Persons directly marked as unrelated to target
        SELECT
            CASE
                WHEN pnr.person1_id = target_person_id THEN pnr.person2_id
                ELSE pnr.person1_id
            END AS pid
        FROM person_not_related pnr
        WHERE pnr.person1_id = target_person_id
           OR pnr.person2_id = target_person_id
    ),
    transitive_unrelated AS (
        -- Persons unrelated to any relative of target
        SELECT
            CASE
                WHEN pnr.person1_id = tr.relative_id THEN pnr.person2_id
                ELSE pnr.person1_id
            END AS pid
        FROM person_not_related pnr
        JOIN target_relatives tr ON (
            pnr.person1_id = tr.relative_id OR pnr.person2_id = tr.relative_id
        )
        WHERE pnr.person1_id != target_person_id
          AND pnr.person2_id != target_person_id
    )
    SELECT DISTINCT du.pid, true AS is_direct
    FROM direct_unrelated du
    UNION
    SELECT DISTINCT tu.pid, false AS is_direct
    FROM transitive_unrelated tu
    WHERE tu.pid NOT IN (SELECT du.pid FROM direct_unrelated du);
END;
$$ LANGUAGE plpgsql STABLE;
```

## Non-Relationship Tracking

### Use Case: Filtering Genealogical Relation Candidates

When suggesting potential relatives for a person (e.g., "Who might be Alice's parent?"), the system should exclude persons who have been explicitly marked as unrelated. This improves the user experience by not repeatedly suggesting impossible matches.

**Example workflow:**
1. User views Alice's profile and sees suggested potential parents
2. User clicks "Not Related" on Bob (a neighbor who appears in many photos with Alice)
3. System inserts `(alice_id, bob_id)` into `person_not_related`
4. Future suggestions for Alice's relatives exclude Bob
5. Because Bob is unrelated to Alice, Bob is also transitively unrelated to Alice's children, parents, grandparents, etc.

### Sparse Storage with Transitive Computation

The `person_not_related` table is intentionally sparse - we only store explicit "these people are definitely not related" assertions. Transitive implications are computed on-demand via recursive queries against the ancestor closure:

- **If A is unrelated to B**, and **B is a relative of C** (via ancestor closure), then **A is unrelated to C**
- This avoids materializing O(n²) non-relationship pairs
- Queries use `are_persons_unrelated()` for point checks or `get_unrelated_persons()` for bulk filtering

### Example: Filter Relation Candidates

```sql
-- Suggest potential parents for person 42, excluding known non-relatives
SELECT p.*
FROM person p
WHERE p.id != 42
  AND NOT p.is_placeholder
  AND NOT are_persons_unrelated(42, p.id)
  -- Additional filters: appropriate age range, etc.
  AND p.birth_year_max IS NULL OR p.birth_year_max < (
      SELECT birth_year_min - 15 FROM person WHERE id = 42
  );
```

### Recording Non-Relationships

```sql
-- Mark two persons as unrelated (canonical ordering enforced by CHECK constraint)
INSERT INTO person_not_related (person1_id, person2_id, source)
VALUES (LEAST(42, 99), GREATEST(42, 99), 'user')
ON CONFLICT DO NOTHING;
```

## Integration with Age/Gender Pipeline

The genealogical constraints integrate with the MiVOLO age estimation from [2026-02-01-age-gender-body-detection-design.md](./2026-02-01-age-gender-body-detection-design.md):

### Constraint Flow

1. **MiVOLO estimates** age from photos → stored in `person_detection.age_estimate`
2. **Person aggregation** computes `person.estimated_birth_year` from photo dates + age estimates
3. **Genealogy constraints** tighten `birth_year_min`/`birth_year_max` via `propagate_birth_year_constraints()`
4. **Validation** — if estimated birth year falls outside min/max range, flag for review

### Photo Date Inference

When a photo lacks `captured_at`, infer date range from subject's age + birth year:

```sql
SELECT p.id, p.file_path,
    per.birth_year_min + pd.age_estimate AS photo_year_min,
    per.birth_year_max + pd.age_estimate AS photo_year_max
FROM photo p
JOIN person_detection pd ON pd.photo_id = p.id
JOIN person per ON per.id = pd.person_id
WHERE p.captured_at IS NULL
  AND pd.age_estimate IS NOT NULL
  AND (per.birth_year_min IS NOT NULL OR per.birth_year_max IS NOT NULL);
```

### Photo Ordering by Birth Order

Order photos by the relative birth order of their subjects:

```sql
-- Assign a topological order to persons based on birth order
WITH person_order AS (
    SELECT p.id,
        (SELECT COUNT(*) FROM person_birth_order_closure pboc
         WHERE pboc.younger_person_id = p.id) AS older_count
    FROM person p
)
SELECT ph.*, po.older_count
FROM photo ph
JOIN person_detection pd ON pd.photo_id = ph.id
JOIN person_order po ON po.id = pd.person_id
ORDER BY po.older_count, ph.captured_at;
```

## Usage Examples

### Set Birth Dates

```sql
-- Set exact birth date
UPDATE person SET
    birth_date = '1985-03-15',
    birth_year_min = 1985,
    birth_year_max = 1985,
    birth_year_source = 'exact'
WHERE id = 42;

-- Set year only (no exact date)
UPDATE person SET
    birth_date = NULL,
    birth_year_min = 1985,
    birth_year_max = 1985,
    birth_year_source = 'year'
WHERE id = 42;
```

### Create a Sparse Tree

Alice knows her grandmother but not her parent:

```sql
-- Create known persons
INSERT INTO person (first_name, last_name) VALUES ('Alice', 'Smith');  -- id: 1
INSERT INTO person (first_name, last_name) VALUES ('Grandma', 'Jones'); -- id: 2

-- Create placeholder for unknown parent
SELECT create_placeholder_parent(1, 'mother');  -- id: 3, "Alice Smith's mother"

-- Link placeholder to grandmother
INSERT INTO person_parent (person_id, parent_id, parent_role, source)
VALUES (3, 2, 'mother', 'user');

-- Refresh closures
SELECT refresh_genealogy_closures();

-- Now: Grandma Jones → [Alice Smith's mother] → Alice Smith
SELECT * FROM get_family_tree(1);
```

### Link Siblings with Unknown Parents

```sql
SELECT link_siblings(5, 6, 'full');
-- Creates two placeholder parents, links both children

SELECT refresh_genealogy_closures();
```

### Query Birth Order for Photo Sorting

```sql
-- All persons older than person 42
SELECT p.*, pboc.inference_type
FROM person_birth_order_closure pboc
JOIN person p ON p.id = pboc.older_person_id
WHERE pboc.younger_person_id = 42
ORDER BY pboc.inference_type;  -- direct > genealogy > transitive
```

### Propagate Age Constraints After Adding Relationships

```sql
-- After adding parent-child relationships
SELECT refresh_genealogy_closures();
SELECT propagate_birth_year_constraints();

-- Check for constraint violations
SELECT p.id, person_display_name(p.*),
       p.estimated_birth_year, p.birth_year_min, p.birth_year_max
FROM person p
WHERE p.estimated_birth_year IS NOT NULL
  AND (p.estimated_birth_year < p.birth_year_min
       OR p.estimated_birth_year > p.birth_year_max);
```

## Migration Strategy

1. Add new columns to `person` table
2. Create new relationship tables
3. Create closure tables
4. Create views and functions
5. Initial `refresh_genealogy_closures()` (empty, no relationships yet)

```sql
-- Migration script order:
-- 1. ALTER TABLE person ... (add gender, is_placeholder, placeholder_description, birth_year_*)
-- 2. CREATE TABLE person_parent ...
-- 3. CREATE TABLE person_partnership ...
-- 4. CREATE TABLE person_birth_order ...
-- 5. CREATE TABLE person_not_related ...
-- 6. CREATE TABLE person_ancestor_closure ...
-- 7. CREATE TABLE person_birth_order_closure ...
-- 8. CREATE VIEW person_siblings ...
-- 9. CREATE VIEW person_known ...
-- 10. CREATE FUNCTION person_display_name ...
-- 11. CREATE FUNCTION create_placeholder_parent ...
-- 12. CREATE FUNCTION link_siblings ...
-- 13. CREATE FUNCTION refresh_ancestor_closure ...
-- 14. CREATE FUNCTION refresh_birth_order_closure ...
-- 15. CREATE FUNCTION refresh_genealogy_closures ...
-- 16. CREATE FUNCTION get_family_tree ...
-- 17. CREATE FUNCTION propagate_birth_year_constraints ...
-- 18. CREATE FUNCTION are_persons_unrelated ...
-- 19. CREATE FUNCTION get_unrelated_persons ...
```

## When to Refresh Closures

The closure tables must be refreshed when relationships change:

| Event | Action |
|-------|--------|
| Add/remove `person_parent` row | `refresh_genealogy_closures()` |
| Add/remove `person_birth_order` row | `refresh_birth_order_closure()` |
| Delete a person | Automatic via `ON DELETE CASCADE` |
| Bulk import | `refresh_genealogy_closures()` after all inserts |

For interactive use, consider calling refresh after each edit. For batch imports, defer to end.

## Future Considerations

- **Cycle detection** — Validate no person is their own ancestor before inserting
- **Merge persons** — When a placeholder is identified as a known person, merge records
- **GEDCOM import** — Standard genealogy format import for existing family trees
- **Relationship inference** — Suggest relationships based on co-occurrence in photos, age gaps
- **Non-relationship materialization** — If query performance becomes an issue, consider materializing transitive non-relationships
