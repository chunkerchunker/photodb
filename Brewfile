# PhotoDB - required Homebrew dependencies
# Install with: brew bundle

# Core
brew "git"
brew "just"
brew "node"
brew "pnpm"
brew "uv"

# Database
brew "postgresql@18", restart_service: :changed
brew "pgvector"

# Image processing (libvips for fast HEIF/JPEG → WebP conversion)
brew "vips"

# Utilities (used by scripts/aws_sum_token_counts.sh)
brew "jq"
