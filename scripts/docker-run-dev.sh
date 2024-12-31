#!/bin/bash

docker build --tag 'koel-labs-server' -f ./scripts/Dockerfile.dev .

if [[ "$OSTYPE" == "msys" ]]; then
    # windows
    explorer http://localhost:8080
else
    # not windows
    open http://localhost:8080 || start chrome \"http://localhost:8080\" || google-chrome 'http://localhost:8080' || echo 'Could not open browser automatically. Please open http://localhost:8080 manually'
fi

docker run -t -i -p 8080:8080 -v "/$(pwd)/src":/app/src -v "/$(pwd)/.cache/huggingface/hub":/app/.cache/huggingface/hub --env-file .env 'koel-labs-server'
