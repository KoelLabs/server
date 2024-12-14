#!/bin/bash

docker build --tag 'koel-labs-server' -f ./scripts/Dockerfile.dev .

open http://localhost:8080 || start chrome \"http://localhost:8080\" || google-chrome 'http://localhost:8080' || echo 'Could not open browser automatically. Please open http://localhost:8080 manually'
docker run -t -i -p 8080:8080 -v ./src:/app/src -v ./.cache/huggingface/hub:/app/.cache/huggingface/hub --env-file .env 'koel-labs-server'
