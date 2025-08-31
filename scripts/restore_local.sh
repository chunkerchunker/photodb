dotenv -e $(dirname $0)/../.env \
  -v TIMESTAMP=${TIMESTAMP:-$1} \
  -v STAGE=${STAGE:-"local"} \
  -- bash -c $(dirname $0)/restore.sh
