#!/bin/bash
set -euo pipefail

# --- Docker ---
dnf install -y docker git rsync
systemctl enable --now docker
usermod -aG docker ec2-user

# Docker CLI plugins (Compose + Buildx) for ARM
mkdir -p /usr/local/lib/docker/cli-plugins
curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-aarch64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
BUILDX_VERSION=$(curl -s https://api.github.com/repos/docker/buildx/releases/latest | grep tag_name | cut -d'"' -f4)
curl -SL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-arm64" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose \
         /usr/local/lib/docker/cli-plugins/docker-buildx

# --- PostgreSQL 16 client (for pg_dump/pg_restore) ---
dnf install -y postgresql16

# --- Data directories ---
mkdir -p /data/photos /data/postgres

# --- App directory ---
mkdir -p /home/ec2-user/photodb
chown -R ec2-user:ec2-user /home/ec2-user/photodb /data

# --- Shell config ---
echo 'export TERM=xterm-256color' >> /home/ec2-user/.bashrc
