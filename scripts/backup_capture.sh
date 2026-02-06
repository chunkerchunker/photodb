#!/usr/bin/env bash

INCLUDE_TABLES=(
  albums
  crop_faces
  crops
  orders
  pages
  persons
)

PG_DUMP_OPTS=$(printf -- '-t %s ' "${INCLUDE_TABLES[@]}")

# echo $PG_DUMP_OPTS

dotenv -e $(dirname $0)/../.env \
  -v STAGE=${STAGE:-"capture"} \
  -v PG_DUMP_OPTS="$PG_DUMP_OPTS -T *_seq" \
  -- bash $(dirname $0)/backup.sh
