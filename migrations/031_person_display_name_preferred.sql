-- Use preferred_name over first_name in person_display_name, matching site convention
CREATE OR REPLACE FUNCTION public.person_display_name(p person)
 RETURNS text
 LANGUAGE plpgsql
 IMMUTABLE
AS $function$
BEGIN
    IF p.first_name IS NOT NULL OR p.last_name IS NOT NULL THEN
        RETURN TRIM(COALESCE(COALESCE(p.preferred_name, p.first_name) || ' ', '') || COALESCE(p.last_name, ''));
    ELSIF p.placeholder_description IS NOT NULL THEN
        RETURN p.placeholder_description;
    ELSE
        RETURN 'Unknown #' || p.id;
    END IF;
END;
$function$;
