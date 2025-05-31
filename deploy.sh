#!/bin/bash

# Build and start containers
docker-compose up -d

# Show logs
docker-compose logs -f
