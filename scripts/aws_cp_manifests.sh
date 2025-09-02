BASE_DIR=$(dirname $0)/../tmp/aws

aws s3 ls s3://wavio-tmp-us-east-1/batch-output/ --recursive \
  | awk '{print $4}' \
  | grep manifest \
  | while read -r key; do
      if [ ! -f "$BASE_DIR/$key" ]; then
        mkdir -p "$BASE_DIR/$(dirname "$key")"
        aws s3 cp "s3://wavio-tmp-us-east-1/$key" "$BASE_DIR/$key"
      fi
    done