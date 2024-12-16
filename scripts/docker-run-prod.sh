#!/bin/bash

# Prepare the secret files and build the Docker image
while IFS='=' read -r key value; do
    echo -n "$value" > "$key.secret"
done < .env

docker build --tag 'koel-labs-server-prod' -f ./scripts/Dockerfile.prod $(while IFS='=' read -r key value; do echo "--secret id=$key,src=$key.secret "; done < .env) .

# Clean up the temporary secret files
while IFS='=' read -r key value; do
  rm "$key.secret"
done < .env

open http://localhost:8080 || start chrome \"http://localhost:8080\" || google-chrome 'http://localhost:8080' || echo 'Could not open browser automatically. Please open http://localhost:8080 manually'
docker run -t -i -p 8080:8080 --env-file .env 'koel-labs-server-prod'
