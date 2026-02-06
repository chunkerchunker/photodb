if [ -z "$STAGE" ]; then
  echo "STAGE is not set"
  exit 1
fi
if [ -z "$DATABASE_URL" ]; then
  echo "DATABASE_URL is not set"
  exit 1
fi
if [ -z "$TIMESTAMP" ]; then
  echo "TIMESTAMP is not set"
  exit 1
fi


read -p "Are you sure you want to update ${STAGE}? (enter ${STAGE} to confirm) " -r
echo

if [[ ! $REPLY =~ ^${STAGE}$ ]]; then
  echo "Operation aborted."
  exit 1
fi

BACKUPS_DIR=`realpath $(dirname $0)/../bkup`

# ensure the backup file exists
BACKUP_FILE="${STAGE}__${TIMESTAMP}.sql.gz"
if [ ! -f "${BACKUPS_DIR}/$BACKUP_FILE" ]; then
  echo "Backup file not found: ${BACKUP_FILE}"
  exit 1
fi

echo "Restoring ${STAGE} from ${BACKUPS_DIR}/${BACKUP_FILE}"

PGGSSENCMODE=disable PGSSLMODE=allow psql ${DATABASE_URL} -c "drop schema public cascade;" > /dev/null 2>&1
PGGSSENCMODE=disable PGSSLMODE=allow psql ${DATABASE_URL} -c "create schema public;" > /dev/null 2>&1
PGGSSENCMODE=disable PGSSLMODE=allow psql ${DATABASE_URL} -c "CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;" > /dev/null 2>&1
OUTPUT=$(gzcat ${BACKUPS_DIR}/${BACKUP_FILE} | PGGSSENCMODE=disable PGSSLMODE=allow pg_restore --no-owner --no-privileges -d $DATABASE_URL 2>&1)
if echo "$OUTPUT" | grep -q "ERROR"; then
  echo "Error detected during restore:"
  echo "$OUTPUT"
  exit 1
fi

echo "Done"