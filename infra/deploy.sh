#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Get outputs from Pulumi
IP=$(cd "$SCRIPT_DIR" && pulumi stack output publicIp)
SSH_KEY_FILE="$HOME/.ssh/photodb"

# Save SSH key if not already saved
if [ ! -f "$SSH_KEY_FILE" ]; then
  echo "Saving SSH private key to $SSH_KEY_FILE..."
  cd "$SCRIPT_DIR" && pulumi stack output sshPrivateKey --show-secrets > "$SSH_KEY_FILE"
  chmod 600 "$SSH_KEY_FILE"
fi

SSH_OPTS="-i $SSH_KEY_FILE -o StrictHostKeyChecking=accept-new"
SSH="ssh $SSH_OPTS ec2-user@$IP"
RSYNC="rsync -avz -e \"ssh $SSH_OPTS\""

echo "Deploying to $IP..."

# Sync web app source
eval $RSYNC \
  --exclude node_modules \
  --exclude .git \
  --exclude build \
  "$PROJECT_DIR/web/" "ec2-user@$IP:~/photodb/web/"

# Sync docker-compose
eval $RSYNC \
  "$SCRIPT_DIR/docker-compose.prod.yml" "ec2-user@$IP:~/photodb/docker-compose.yml"

# Sync .env.deploy -> .env on server
ENV_FILE="$SCRIPT_DIR/.env.deploy"
if [ -f "$ENV_FILE" ]; then
  eval $RSYNC "$ENV_FILE" "ec2-user@$IP:~/photodb/.env"
else
  echo "Error: $ENV_FILE not found. Create it before running deploy.sh."
  echo "See DEMO.md for required variables."
  exit 1
fi

# Build and start
$SSH "cd ~/photodb && docker compose up -d --build"

echo ""
echo "Deployed: http://$IP:3000"
echo "SSH:      ssh -i ~/.ssh/photodb ec2-user@$IP"
