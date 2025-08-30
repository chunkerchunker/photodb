CREATE TABLE photo(
  id uuid PRIMARY KEY,
  filename text NOT NULL, -- path to original file
  created_at timestamp DEFAULT CURRENT_TIMESTMP,
);

CREATE UNIQUE INDEX idx_photo_filename ON photo(filename);

CREATE TABLE metadata(
  photo_id uuid PRIMARY KEY REFERENCES photo(id),
  captured_at timestamp,
  lat real, -- latitude for location info (Spatialite)
  lon real -- longitude for location info (Spatialite)
  extra jsonb -- additional exif/tiff/ifd data
);

-- Person (identified later through clustering)
CREATE TABLE person(
  id uuid PRIMARY KEY,
  name text -- user-assigned ("Dad", "Alice")
);

-- Faces detected in photos
CREATE TABLE faces(
  face_id uuid PRIMARY KEY,
  photo_id uuid REFERENCES photos(id),
  person_id uuid REFERENCES people(id), -- nullable until labeled
  bbox jsonb, -- bounding box {x,y,w,h}
  confidence float,
  created_at timestamp DEFAULT CURRENT_TIMESTMP
);

