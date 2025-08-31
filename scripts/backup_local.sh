dotenv -e $(dirname $0)/../.env \
  -v STAGE=${STAGE:-"local"} \
  -- bash -c $(dirname $0)/backup.sh
