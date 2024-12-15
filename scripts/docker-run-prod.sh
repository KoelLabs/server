#!/bin/bash

docker build --tag 'koel-labs-server-prod' -f ./scripts/Dockerfile.prod $(for i in `cat .env`; do out+="--build-arg $i "; done; echo $out; out="") .

open http://localhost:8080 || start chrome \"http://localhost:8080\" || google-chrome 'http://localhost:8080' || echo 'Could not open browser automatically. Please open http://localhost:8080 manually'
docker run -t -i -p 8080:8080 --env-file .env 'koel-labs-server-prod'
