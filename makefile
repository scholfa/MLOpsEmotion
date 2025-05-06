# File: Makefile

# Name of your Compose project (optional)
PROJECT_NAME := mlopsemotion

# Default target
.PHONY: all
all: build init up

.PHONY: build
build:
	# Build the app image and services
	podman-compose build

# Rebuild the app image and services
.PHONY: rebuild
rebuild:
	podman-compose build --no-cache

# Run the init container to populate shared files (flows + prefect.yaml)
.PHONY: init
init:
	podman-compose run --rm init-shared

# Start all containers
.PHONY: up
up:
	podman-compose up

# Stop and remove containers
.PHONY: down
down:
	podman-compose down

# Reset the app-shared volume
.PHONY: reset-shared
reset-shared:
	-podman volume rm ${PROJECT_NAME}_app-shared || true

# Reset everything (data, shared, containers)
.PHONY: reset-all
reset-all: down
	-podman volume rm ${PROJECT_NAME}_app-data || true
	-podman volume rm ${PROJECT_NAME}_app-shared || true
