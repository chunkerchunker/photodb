# PhotoDB Web App — AWS Deployment

Deploys the web app (React Router v7 + PostgreSQL) to a single EC2 instance using Pulumi.

## What gets provisioned

| Resource       | Details                                       |
| -------------- | --------------------------------------------- |
| EC2            | `t4g.small` ARM (2 vCPU, 2GB RAM, ~$12/mo)    |
| Root EBS       | 40GB gp3, persists on termination             |
| Security group | Ports 22 + 3000, locked to allowlisted IPs    |
| Elastic IP     | Stable public address across stop/start       |
| SSH key        | ED25519, generated and stored in Pulumi state |

## Prerequisites

- [Pulumi CLI](https://www.pulumi.com/docs/install/)
- AWS credentials configured (`aws configure` or env vars)
- Node.js 20+

## Setup

```bash
cd infra
pnpm install
pulumi stack init prod
```

## Deploy infrastructure

```bash
pulumi up
```

Save the SSH key locally:

```bash
pulumi stack output sshPrivateKey --show-secrets > ~/.ssh/photodb
chmod 600 ~/.ssh/photodb
```

## Configure environment

Create `infra/.env.deploy` with your production environment variables. This file is copied to the server as `.env` on each deploy.

```bash
cp .env.deploy.example .env.deploy
# Edit with your values
```

Required variables:

```env
POSTGRES_PASSWORD=<strong-password>
```

## Deploy the app

```bash
./deploy.sh
```

This rsyncs the web app source and docker-compose file to the server, then runs `docker compose up -d --build`. The server runs two containers:

- **db** — PostgreSQL 17 + pgvector (data at `/data/postgres`)
- **web** — PhotoDB web app on port 3000

## First-time data migration

### 1. Sync photos

```bash
rsync -avz -e 'ssh -i ~/.ssh/photodb' \
  /path/to/local/photos/ \
  ec2-user@$(pulumi stack output publicIp):/data/photos/
```

### 2. Restore database

Use `--no-owner` to skip local role references and `--role` to remap ownership:

```bash
pg_dump --no-owner photodb | ssh -i ~/.ssh/photodb \
  ec2-user@$(pulumi stack output publicIp) \
  'docker exec -i photodb-db-1 psql -U photodb photodb'
```

### 3. Rewrite photo paths

The web app reads absolute file paths (`med_path`, `full_path`, `orig_path`) from the database. These need to match where photos are mounted inside the container (`/data/photos/`).

```bash
ssh -i ~/.ssh/photodb ec2-user@$(pulumi stack output publicIp) \
  'docker exec photodb-db-1 psql -U photodb photodb' <<'SQL'
UPDATE photo SET
  med_path  = replace(med_path,  '/old/local/path', '/data/photos'),
  full_path = replace(full_path, '/old/local/path', '/data/photos'),
  orig_path = replace(orig_path, '/old/local/path', '/data/photos');
SQL
```

Replace `/old/local/path` with whatever prefix your local photo paths use.

## Ongoing operations

```bash
# SSH into the server
ssh -i ~/.ssh/photodb ec2-user@$(pulumi stack output publicIp)

# View logs
ssh ... 'cd ~/photodb && docker compose logs -f web'

# Restart
ssh ... 'cd ~/photodb && docker compose restart'

# Redeploy after code changes
./deploy.sh

# Stop instance (keeps data, stops billing for compute)
aws ec2 stop-instances --instance-ids $(pulumi stack output instanceId)

# Start it back
aws ec2 start-instances --instance-ids $(pulumi stack output instanceId)
```

## Update allowed IPs

Edit `Pulumi.prod.yaml`:

```yaml
config:
  photodb:allowedIps:
    - "24.4.199.105"
    - "NEW.IP.HERE"
```

Then apply:

```bash
pulumi up
```

## Tear down

```bash
pulumi destroy
```

Note: the root EBS volume has `deleteOnTermination: false`, so it survives instance termination. Delete it manually in the AWS console if you want to remove all data.
