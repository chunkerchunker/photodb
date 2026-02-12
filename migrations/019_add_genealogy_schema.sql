-- Migration 019: Add Genealogy Schema
-- Description: Adds genealogical relationship modeling to person entities
-- Features:
--   - Family tree display (ancestors, descendants, siblings, partners)
--   - Sparse/incomplete trees via placeholder persons
--   - Birth order (pairwise partial ordering among siblings/cousins)
--   - Age inference via birth year constraint propagation
-- Design: docs/plans/2026-02-11-genealogy-schema-design.md
-- Created: 2026-02-11

BEGIN;

-- ============================================================================
-- Part 1: Person Table Modifications
-- ============================================================================

-- Make names nullable for placeholders
ALTER TABLE person ALTER COLUMN first_name DROP NOT NULL;
ALTER TABLE person ALTER COLUMN last_name DROP NOT NULL;

-- Placeholder support
ALTER TABLE person ADD COLUMN IF NOT EXISTS is_placeholder BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE person ADD COLUMN IF NOT EXISTS placeholder_description TEXT;

-- Gender
ALTER TABLE person ADD COLUMN IF NOT EXISTS gender CHAR(1);
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'person_gender_check') THEN
        ALTER TABLE person ADD CONSTRAINT person_gender_check CHECK (gender IS NULL OR gender IN ('M', 'F', 'U'));
    END IF;
END $$;

-- Birth year constraints (for age inference)
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_year_min INT;
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_year_max INT;
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_year_source TEXT;

-- Add constraint for birth_year_source values
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'person_birth_year_source_check') THEN
        ALTER TABLE person ADD CONSTRAINT person_birth_year_source_check
            CHECK (birth_year_source IS NULL OR birth_year_source IN ('exact', 'year', 'estimated', 'inferred'));
    END IF;
END $$;

-- Exact birth date (when known)
ALTER TABLE person ADD COLUMN IF NOT EXISTS birth_date DATE;

-- Constraint to ensure birth_date year matches birth_year_min/max when set
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'person_birth_date_year_check') THEN
        ALTER TABLE person ADD CONSTRAINT person_birth_date_year_check
            CHECK (
                birth_date IS NULL
                OR (birth_year_min = EXTRACT(YEAR FROM birth_date)::INT
                    AND birth_year_max = EXTRACT(YEAR FROM birth_date)::INT)
            );
    END IF;
END $$;

-- Index for filtering placeholder persons
CREATE INDEX IF NOT EXISTS idx_person_placeholder ON person(is_placeholder) WHERE is_placeholder;

COMMENT ON COLUMN person.is_placeholder IS 'True for placeholder persons representing unknown intermediates in family tree';
COMMENT ON COLUMN person.placeholder_description IS 'Description for placeholder persons (e.g., "John Smith''s father")';
COMMENT ON COLUMN person.gender IS 'Gender: M (male), F (female), U (unknown/other)';
COMMENT ON COLUMN person.birth_year_min IS 'Earliest possible birth year (from age inference or user input)';
COMMENT ON COLUMN person.birth_year_max IS 'Latest possible birth year (from age inference or user input)';
COMMENT ON COLUMN person.birth_year_source IS 'Source of birth year constraints: exact, year, estimated, or inferred';
COMMENT ON COLUMN person.birth_date IS 'Exact birth date if known; when set, birth_year_min and birth_year_max must equal the year';

-- ============================================================================
-- Part 2: Genealogical Relationship Tables
-- ============================================================================

-- Parent-child relationships (the fundamental unit)
CREATE TABLE IF NOT EXISTS person_parent (
    person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    parent_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    parent_role TEXT,
    is_biological BOOLEAN NOT NULL DEFAULT true,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (person_id, parent_id),
    CONSTRAINT person_parent_role_check CHECK (parent_role IS NULL OR parent_role IN ('mother', 'father', 'parent')),
    CONSTRAINT person_parent_source_check CHECK (source IS NULL OR source IN ('user', 'inferred', 'imported')),
    CONSTRAINT person_parent_no_self_reference CHECK (person_id != parent_id)
);

CREATE INDEX IF NOT EXISTS idx_person_parent_parent ON person_parent(parent_id);

COMMENT ON TABLE person_parent IS 'Parent-child relationships forming the genealogical tree';
COMMENT ON COLUMN person_parent.parent_role IS 'Role of parent: mother, father, or parent (unknown/non-binary)';
COMMENT ON COLUMN person_parent.is_biological IS 'True for biological parent, false for adoptive/step parent';
COMMENT ON COLUMN person_parent.source IS 'Source of the relationship: user, inferred, or imported';

-- Partnerships (marriages, relationships)
CREATE TABLE IF NOT EXISTS person_partnership (
    id BIGSERIAL PRIMARY KEY,
    person1_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    person2_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    partnership_type TEXT,
    start_year INT,
    end_year INT,
    is_current BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT person_partnership_type_check CHECK (partnership_type IS NULL OR partnership_type IN ('married', 'partner', 'divorced', 'separated')),
    CONSTRAINT person_partnership_canonical_order CHECK (person1_id < person2_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_person_partnership_unique
    ON person_partnership(person1_id, person2_id, COALESCE(start_year, 0));
CREATE INDEX IF NOT EXISTS idx_person_partnership_p2 ON person_partnership(person2_id);

COMMENT ON TABLE person_partnership IS 'Partnership relationships between persons (marriages, relationships)';
COMMENT ON COLUMN person_partnership.partnership_type IS 'Type of partnership: married, partner, divorced, or separated';
COMMENT ON COLUMN person_partnership.start_year IS 'Year the partnership started';
COMMENT ON COLUMN person_partnership.end_year IS 'Year the partnership ended (null if ongoing)';
COMMENT ON COLUMN person_partnership.is_current IS 'True if this is the current/active partnership';

-- Pairwise birth order (partial ordering)
-- Only stores KNOWN orderings; transitive implications materialized separately
CREATE TABLE IF NOT EXISTS person_birth_order (
    older_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    younger_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    source TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (older_person_id, younger_person_id),
    CONSTRAINT person_birth_order_source_check CHECK (source IS NULL OR source IN ('exact_dates', 'user', 'inferred', 'photo_evidence')),
    CONSTRAINT person_birth_order_no_self_reference CHECK (older_person_id != younger_person_id)
);

CREATE INDEX IF NOT EXISTS idx_person_birth_order_younger ON person_birth_order(younger_person_id);

COMMENT ON TABLE person_birth_order IS 'Pairwise birth order relationships (partial ordering among siblings/cousins)';
COMMENT ON COLUMN person_birth_order.older_person_id IS 'The person born first';
COMMENT ON COLUMN person_birth_order.younger_person_id IS 'The person born second';
COMMENT ON COLUMN person_birth_order.source IS 'Source: exact_dates, user, inferred, or photo_evidence';

-- Explicit non-relationships (people who are definitely NOT related)
CREATE TABLE IF NOT EXISTS person_not_related (
    person1_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    person2_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    source TEXT CHECK (source IS NULL OR source IN ('user', 'inferred', 'imported')),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (person1_id < person2_id),
    PRIMARY KEY (person1_id, person2_id)
);

CREATE INDEX IF NOT EXISTS person_not_related_p2_idx ON person_not_related(person2_id);

COMMENT ON TABLE person_not_related IS 'Explicit non-relationships tracking people who are definitely NOT related';
COMMENT ON COLUMN person_not_related.person1_id IS 'First person in the non-relationship (always < person2_id)';
COMMENT ON COLUMN person_not_related.person2_id IS 'Second person in the non-relationship (always > person1_id)';
COMMENT ON COLUMN person_not_related.source IS 'Source of the non-relationship: user, inferred, or imported';
COMMENT ON COLUMN person_not_related.notes IS 'Optional notes explaining why these persons are not related';

-- ============================================================================
-- Part 3: Materialized Closure Tables
-- ============================================================================

-- Transitive closure of ancestor relationships
CREATE TABLE IF NOT EXISTS person_ancestor_closure (
    descendant_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    ancestor_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    distance INT NOT NULL,
    PRIMARY KEY (descendant_id, ancestor_id),
    CONSTRAINT person_ancestor_closure_distance_check CHECK (distance >= 1)
);

CREATE INDEX IF NOT EXISTS idx_pac_ancestor ON person_ancestor_closure(ancestor_id);
CREATE INDEX IF NOT EXISTS idx_pac_distance ON person_ancestor_closure(distance);

COMMENT ON TABLE person_ancestor_closure IS 'Materialized transitive closure of ancestor relationships';
COMMENT ON COLUMN person_ancestor_closure.distance IS 'Generation distance (1=parent, 2=grandparent, etc.)';

-- Transitive closure of birth order (includes genealogy-implied)
CREATE TABLE IF NOT EXISTS person_birth_order_closure (
    older_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    younger_person_id BIGINT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    inference_type TEXT,
    PRIMARY KEY (older_person_id, younger_person_id),
    CONSTRAINT person_birth_order_closure_type_check CHECK (inference_type IS NULL OR inference_type IN ('direct', 'genealogy', 'transitive'))
);

CREATE INDEX IF NOT EXISTS idx_pboc_younger ON person_birth_order_closure(younger_person_id);

COMMENT ON TABLE person_birth_order_closure IS 'Materialized transitive closure of birth order (direct + genealogy-implied + transitive)';
COMMENT ON COLUMN person_birth_order_closure.inference_type IS 'How the ordering was derived: direct, genealogy, or transitive';

-- ============================================================================
-- Part 4: Views
-- ============================================================================

-- Siblings derived from shared parents
CREATE OR REPLACE VIEW person_siblings AS
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

COMMENT ON VIEW person_siblings IS 'Derived view of sibling relationships based on shared parents';

-- Non-placeholder persons (known/identified persons)
CREATE OR REPLACE VIEW person_known AS
SELECT * FROM person WHERE NOT is_placeholder;

COMMENT ON VIEW person_known IS 'View of non-placeholder persons (known/identified persons)';

-- ============================================================================
-- Part 5: Functions
-- ============================================================================

-- Display name function for persons (handles placeholders)
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

COMMENT ON FUNCTION person_display_name(person) IS 'Returns display name for a person, handling placeholders';

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

COMMENT ON FUNCTION create_placeholder_parent(BIGINT, TEXT) IS 'Creates a placeholder parent for a person and links them';

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

COMMENT ON FUNCTION link_siblings(BIGINT, BIGINT, TEXT) IS 'Links two persons as siblings, creating placeholder parents if needed';

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

COMMENT ON FUNCTION refresh_ancestor_closure() IS 'Rebuilds the person_ancestor_closure table from person_parent relationships';

-- Refresh birth order closure using recursive CTE
-- Must be called AFTER refresh_ancestor_closure()
CREATE OR REPLACE FUNCTION refresh_birth_order_closure() RETURNS void AS $$
BEGIN
    TRUNCATE person_birth_order_closure;

    INSERT INTO person_birth_order_closure (older_person_id, younger_person_id, inference_type)
    WITH RECURSIVE ordered AS (
        -- Direct birth order records
        SELECT older_person_id, younger_person_id, 'direct'::TEXT AS inference_type
        FROM person_birth_order

        UNION

        -- Ancestors are older than descendants (from closure table)
        SELECT ancestor_id AS older_person_id, descendant_id AS younger_person_id, 'genealogy'::TEXT
        FROM person_ancestor_closure

        UNION

        -- Transitive: if A > B and B > C, then A > C
        SELECT o.older_person_id, bo.younger_person_id, 'transitive'::TEXT
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

COMMENT ON FUNCTION refresh_birth_order_closure() IS 'Rebuilds the person_birth_order_closure table (call after refresh_ancestor_closure)';

-- Refresh all closures (convenience function)
CREATE OR REPLACE FUNCTION refresh_genealogy_closures() RETURNS void AS $$
BEGIN
    PERFORM refresh_ancestor_closure();
    PERFORM refresh_birth_order_closure();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_genealogy_closures() IS 'Refreshes both ancestor and birth order closure tables';

-- Family tree query function
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

COMMENT ON FUNCTION get_family_tree(BIGINT, INT, BOOLEAN) IS 'Returns family tree centered on a person with configurable depth';

-- Birth year constraint propagation function
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

COMMENT ON FUNCTION propagate_birth_year_constraints(INT, INT) IS 'Propagates birth year constraints through genealogical relationships';

-- Check if two people are explicitly unrelated (including transitive via family relationships)
CREATE OR REPLACE FUNCTION are_persons_unrelated(p1_id BIGINT, p2_id BIGINT)
RETURNS BOOLEAN AS $$
DECLARE
    canonical_p1 BIGINT;
    canonical_p2 BIGINT;
BEGIN
    -- Normalize order for person_not_related lookup
    IF p1_id < p2_id THEN
        canonical_p1 := p1_id;
        canonical_p2 := p2_id;
    ELSE
        canonical_p1 := p2_id;
        canonical_p2 := p1_id;
    END IF;

    -- Check direct non-relationship
    IF EXISTS (
        SELECT 1 FROM person_not_related
        WHERE person1_id = canonical_p1 AND person2_id = canonical_p2
    ) THEN
        RETURN TRUE;
    END IF;

    -- Check transitive non-relationship:
    -- If A is unrelated to B, and B is related to C (via family tree),
    -- then A is unrelated to C
    RETURN EXISTS (
        WITH RECURSIVE
        -- Get all family members of p1 (including p1)
        p1_family AS (
            SELECT p1_id AS pid
            UNION
            SELECT ancestor_id FROM person_ancestor_closure WHERE descendant_id = p1_id
            UNION
            SELECT descendant_id FROM person_ancestor_closure WHERE ancestor_id = p1_id
            UNION
            SELECT sibling_id FROM person_siblings WHERE person_id = p1_id
        ),
        -- Get all family members of p2 (including p2)
        p2_family AS (
            SELECT p2_id AS pid
            UNION
            SELECT ancestor_id FROM person_ancestor_closure WHERE descendant_id = p2_id
            UNION
            SELECT descendant_id FROM person_ancestor_closure WHERE ancestor_id = p2_id
            UNION
            SELECT sibling_id FROM person_siblings WHERE person_id = p2_id
        )
        -- Check if any member of p1's family is explicitly unrelated to any member of p2's family
        SELECT 1
        FROM p1_family f1
        CROSS JOIN p2_family f2
        WHERE f1.pid != f2.pid
          AND EXISTS (
              SELECT 1 FROM person_not_related pnr
              WHERE (pnr.person1_id = LEAST(f1.pid, f2.pid) AND pnr.person2_id = GREATEST(f1.pid, f2.pid))
          )
    );
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION are_persons_unrelated(BIGINT, BIGINT) IS 'Checks if two persons are explicitly unrelated (direct or transitive via family)';

-- Get all persons explicitly unrelated to a given person (for filtering candidates)
CREATE OR REPLACE FUNCTION get_unrelated_persons(target_id BIGINT)
RETURNS TABLE (person_id BIGINT, via_person_id BIGINT, is_direct BOOLEAN) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE
    -- Get all family members of target (including target itself)
    target_family AS (
        SELECT target_id AS pid
        UNION
        SELECT ancestor_id FROM person_ancestor_closure WHERE descendant_id = target_id
        UNION
        SELECT descendant_id FROM person_ancestor_closure WHERE ancestor_id = target_id
        UNION
        SELECT sibling_id FROM person_siblings WHERE person_id = target_id
    ),
    -- Direct non-relationships with target
    direct_unrelated AS (
        SELECT
            CASE WHEN pnr.person1_id = target_id THEN pnr.person2_id ELSE pnr.person1_id END AS unrelated_id,
            target_id AS via_id,
            TRUE AS direct
        FROM person_not_related pnr
        WHERE pnr.person1_id = target_id OR pnr.person2_id = target_id
    ),
    -- Transitive non-relationships (via target's family members)
    transitive_unrelated AS (
        SELECT
            CASE WHEN pnr.person1_id = tf.pid THEN pnr.person2_id ELSE pnr.person1_id END AS unrelated_id,
            tf.pid AS via_id,
            FALSE AS direct
        FROM target_family tf
        JOIN person_not_related pnr ON (pnr.person1_id = tf.pid OR pnr.person2_id = tf.pid)
        WHERE tf.pid != target_id
    ),
    -- Combined and expand to include family members of unrelated persons
    all_unrelated AS (
        SELECT unrelated_id, via_id, direct FROM direct_unrelated
        UNION
        SELECT unrelated_id, via_id, direct FROM transitive_unrelated
    ),
    -- Expand to include family of unrelated persons (they're also unrelated to target)
    expanded_unrelated AS (
        SELECT au.unrelated_id, au.via_id, au.direct
        FROM all_unrelated au
        UNION
        SELECT pac.descendant_id, au.unrelated_id, FALSE
        FROM all_unrelated au
        JOIN person_ancestor_closure pac ON pac.ancestor_id = au.unrelated_id
        UNION
        SELECT pac.ancestor_id, au.unrelated_id, FALSE
        FROM all_unrelated au
        JOIN person_ancestor_closure pac ON pac.descendant_id = au.unrelated_id
        UNION
        SELECT ps.sibling_id, au.unrelated_id, FALSE
        FROM all_unrelated au
        JOIN person_siblings ps ON ps.person_id = au.unrelated_id
    )
    SELECT DISTINCT ON (eu.unrelated_id)
        eu.unrelated_id AS person_id,
        eu.via_id AS via_person_id,
        eu.direct AS is_direct
    FROM expanded_unrelated eu
    WHERE eu.unrelated_id NOT IN (SELECT pid FROM target_family)  -- Exclude target's own family
    ORDER BY eu.unrelated_id, eu.direct DESC;  -- Prefer direct relationships
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION get_unrelated_persons(BIGINT) IS 'Returns all persons explicitly unrelated to the target (for filtering candidates)';

-- ============================================================================
-- Part 6: Initialize Empty Closures
-- ============================================================================

-- Run initial refresh (will be empty since no relationships exist yet)
SELECT refresh_genealogy_closures();

-- ============================================================================
-- Part 7: Record Migration
-- ============================================================================

INSERT INTO schema_migrations (version, description)
VALUES ('019', 'Add genealogy schema for family tree, birth order, and age inference')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ============================================================================
-- Post-migration notes:
-- ============================================================================
--
-- After running this migration:
--
-- 1. Person table now supports:
--    - Placeholders (is_placeholder, placeholder_description)
--    - Gender (M/F/U)
--    - Birth year constraints (birth_year_min, birth_year_max, birth_year_source)
--    - Exact birth date (birth_date) - when set, birth_year_min/max must match the year
-- 2. New tables: person_parent, person_partnership, person_birth_order, person_not_related
-- 3. Closure tables: person_ancestor_closure, person_birth_order_closure
-- 4. Views: person_siblings, person_known
-- 5. Functions for genealogy management:
--    - person_display_name(person) - Returns display name for persons
--    - create_placeholder_parent(child_id, role) - Creates placeholder parent
--    - link_siblings(person1_id, person2_id, sibling_type) - Links siblings
--    - refresh_ancestor_closure() - Rebuilds ancestor closure
--    - refresh_birth_order_closure() - Rebuilds birth order closure
--    - refresh_genealogy_closures() - Convenience function for both
--    - get_family_tree(center_id, max_generations, include_placeholders) - Query family tree
--    - propagate_birth_year_constraints(min_gap, max_gap) - Age inference propagation
--    - are_persons_unrelated(p1_id, p2_id) - Check if two persons are unrelated
--    - get_unrelated_persons(target_id) - Get all persons unrelated to target
--
-- Usage examples:
--
--   -- Create a placeholder parent
--   SELECT create_placeholder_parent(1, 'mother');
--
--   -- Link two siblings
--   SELECT link_siblings(5, 6, 'full');
--
--   -- Refresh closures after adding relationships
--   SELECT refresh_genealogy_closures();
--
--   -- Query family tree
--   SELECT * FROM get_family_tree(1, 3, true);
--
--   -- Propagate birth year constraints
--   SELECT propagate_birth_year_constraints();
--
--   -- Mark two people as unrelated
--   INSERT INTO person_not_related (person1_id, person2_id, source)
--   VALUES (LEAST(1, 5), GREATEST(1, 5), 'user');
--
--   -- Check if two people are unrelated
--   SELECT are_persons_unrelated(1, 10);
--
--   -- Get all persons unrelated to a person (for filtering)
--   SELECT * FROM get_unrelated_persons(1);
--
-- To verify the migration:
--   SELECT version, applied_at, description FROM schema_migrations WHERE version = '019';
--   \d person_parent
--   \d person_ancestor_closure
--   \df refresh_genealogy_closures
--
-- To rollback, run: migrations/019_add_genealogy_schema_rollback.sql
