if [ -z "$STAGE" ]; then
  echo "STAGE is not set"
  exit 1
fi
if [ -z "$DATABASE_URL" ]; then
  echo "DATABASE_URL is not set"
  exit 1
fi

# parse the DATABSE_URL manually, because pg_dump doesn't handle passwords with @ correctly
PROTOCOL=$(echo $DATABASE_URL | sed -n 's/\(.*\):\/\/.*/\1/p')
USER=$(echo $DATABASE_URL | sed -n 's/.*\/\/\(.*\):.*@.*/\1/p')
PASSWORD=$(echo $DATABASE_URL | sed -n 's/.*:\/\/.*:\(.*\)@.*/\1/p')
HOST=$(echo $DATABASE_URL | sed -n 's/.*@\(.*\):.*/\1/p')
PORT=$(echo $DATABASE_URL | sed -n 's/.*@.*:\(.*\)\/.*/\1/p')
DATABASE=$(echo $DATABASE_URL | sed -n 's/.*\/\([^?]*\).*/\1/p')

# echo "Backing up ${STAGE} database at ${PROTOCOL}://${HOST}:${PORT}/${DATABASE} as user ${USER} with password ${PASSWORD}"

TIMESTAMP=`date +%Y-%m-%d_%H%M%S`
BACKUPS_DIR=`realpath $(dirname $0)/../bkup`
mkdir -p $BACKUPS_DIR
echo "Backing up ${STAGE} to ${BACKUPS_DIR}/${STAGE}__${TIMESTAMP}.sql.gz"
USER_FLAG=""
if [ -n "$USER" ]; then
  USER_FLAG="-U ${USER}"
fi
PGPASSWORD=${PASSWORD} PGGSSENCMODE=disable PGSSLMODE=allow pg_dump ${PROTOCOL}://${HOST}:${PORT}/${DATABASE} ${USER_FLAG} -Fc -w --no-owner --no-privileges --no-comments --clean --if-exists -n public ${PG_DUMP_OPTS} | gzip > ${BACKUPS_DIR}/${STAGE}__${TIMESTAMP}.sql.gz
echo "Done"