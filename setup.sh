#!/bin/bash

set -e

echo "Setting up your local environment..."

# Safely update UID and GID in .env without sourcing
if [ -f .env ]; then
  echo "Updating .env with current UID and GID..."
  grep -v '^UID=' .env | grep -v '^GID=' > .env.tmp || true
else
  echo "Creating new .env file..."
  touch .env.tmp
fi

echo "UID=$(id -u)" >> .env.tmp
echo "GID=$(id -g)" >> .env.tmp
mv .env.tmp .env

# Ensure .prefect exists and has correct permissions
if [ -d ".prefect" ]; then
  echo "Fixing permissions on .prefect..."
  chown -R $(id -u):$(id -g) .prefect
  chmod -R a+rw .prefect
else
  echo "Creating .prefect directory..."
  mkdir -p .prefect
  touch .prefect/profiles.toml
  chown -R $(id -u):$(id -g) .prefect
  chmod -R a+rw .prefect
fi

# Make all project files readable
echo "Ensuring all files are readable..."
chmod -R a+rX .

echo "Setup complete. You can now run: docker compose up --build"
