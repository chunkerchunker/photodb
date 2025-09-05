#!/bin/bash

# Script to increase PostgreSQL max_connections
# Run with: sudo ./increase_pg_connections.sh

echo "Current PostgreSQL max_connections:"
psql -d postgres -c "SHOW max_connections;"

echo ""
echo "To increase max_connections to 500:"
echo "1. Find your postgresql.conf file:"
psql -t -P format=unaligned -c 'SHOW config_file;'

echo ""
echo "2. Edit the file and change:"
echo "   max_connections = 100"
echo "   to:"
echo "   max_connections = 500"

echo ""
echo "3. Restart PostgreSQL:"
echo "   macOS: brew services restart postgresql@16"
echo "   Linux: sudo systemctl restart postgresql"

echo ""
echo "Note: Each connection uses ~400KB of RAM, so 500 connections = ~200MB RAM"