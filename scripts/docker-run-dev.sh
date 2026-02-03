#!/bin/bash

docker build --tag 'koel-labs-server' -f ./scripts/Dockerfile.dev .

if [[ "$OSTYPE" == "msys" ]]; then
    # windows
    explorer http://localhost:8080
else
    if command -v osascript &> /dev/null; then
        # if osascript is available, we can open an existing tab with http://localhost:8080 if available
        osascript -e 'on run argv' -e 'set targetURL to item 1 of argv' -e 'tell application "Google Chrome"' -e 'set found to false' -e 'repeat with wi from 1 to count of windows' -e 'set w to window wi' -e 'set tabCount to count of tabs of w' -e 'repeat with ti from 1 to tabCount' -e 'set t to tab ti of w' -e 'if (URL of t) starts with targetURL then' -e 'set index of w to 1' -e 'set active tab index of w to ti' -e 'activate' -e 'set found to true' -e 'exit repeat' -e 'end if' -e 'end repeat' -e 'if found then exit repeat' -e 'end repeat' -e 'if not found then' -e 'tell window 1 to set URL of (make new tab) to targetURL' -e 'activate' -e 'end if' -e 'end tell' -e 'end run' -- http://localhost:8080
    else
        # otherwise we'll just open a new tab everytime
        open http://localhost:8080 || start chrome \"http://localhost:8080\" || google-chrome 'http://localhost:8080' || echo 'Could not open browser automatically. Please open http://localhost:8080 manually'
    fi
fi

docker run -t -i -p 8080:8080 -v "/$(pwd)/src":/app/src -v "/$(pwd)/.cache/huggingface/hub":/app/.cache/huggingface/hub --env-file .env 'koel-labs-server'
