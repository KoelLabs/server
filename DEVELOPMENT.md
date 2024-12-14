# Development

## Setup

### With Docker (easiest)

0. `git clone https://github.com/KoelLabs/server.git`
1. Install Docker and Docker Compose
    - [Docker Desktop for Mac](https://docs.docker.com/docker-for-mac/install/) or `brew cask install docker` with [Homebrew](https://brew.sh/)
        - If it repeatedly complains about the daemon not running, make sure Docker Desktop is running and add `export DOCKER_HOST=unix:///Users/$USER/Library/Containers/com.docker.docker/Data/docker.raw.sock` to your shell profile (e.g. `~/.zshrc`)
    - [Docker Desktop for Windows](https://docs.docker.com/docker-for-windows/install/) or `choco install docker-desktop` with [Chocolatey](https://chocolatey.org/)
    - [Docker Engine for Linux](https://docs.docker.com/engine/install/) or `sudo apt install docker.io` with APT on Ubuntu
2. Duplicate the `.env.example` file and rename it to `.env`. Fill in the necessary environment variables.
    - You can find your `HF_TOKEN` on your [Settings Page](https://huggingface.co/settings/tokens). It just needs read access to `gated repos`.
3. Run the application
    - `. ./scripts/docker-run-dev.sh` to start the development server
    - If `http://localhost:8080` doesn't automatically open in your browser, open it manually
    - `ctrl+c` to stop the server

To add new dependencies, update the `requirements.txt` file and run `. ./scripts/docker-run-dev.sh` again.

### Directly on your machine (runs fastest)

0. `git clone https://github.com/KoelLabs/server.git`
1. Install Python 3.8.10 or higher
    - [Install pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)
    - Run `pyenv install 3.10.12`
    - Pyenv should automatically use this version in this directory. If not, run `pyenv local 3.10.12`
2. Create a virtual environment
    - Run `python -m venv ./venv` to create it
    - Run `. venv/bin/activate` when you want to activate it
        - Run `deactivate` when you want to deactivate it
    - Pro-tip: select the virtual environment in your IDE, e.g. in VSCode, click the Python version in the bottom left corner and select the virtual environment
3. Install dependencies
    - Run `pip install -r requirements.txt`
    - [Install the huggingface cli](https://huggingface.co/docs/huggingface_hub/en/guides/cli) with `pip install -U "huggingface_hub[cli]"`
    - Run `huggingface-cli login` using an [Access Token](https://huggingface.co/docs/hub/security-tokens) with read access from your [Settings Page](https://huggingface.co/settings/tokens)
4. Duplicate the `.env.example` file and rename it to `.env`. Fill in the necessary environment variables.
    - You can find your `HF_TOKEN` on your [Settings Page](https://huggingface.co/settings/tokens). It just needs read access to `gated repos`.
5. Run the server
    - Run `python src/server.py` to start the development server
    - Open your browser to `http://localhost:8080`
    - `ctrl+c` to stop the server

To save dependencies you `pip install`, then run `pip freeze > requirements.txt`.

## Formatting, Linting, Automated Tests and Secret Scanning

All checks are run as github actions when you push code. You can also run them manually with `. scripts/alltests.sh`.

- We use [Black](https://black.readthedocs.io/en/stable/) for formatting. It is recommended you [integrate it with your IDE](https://black.readthedocs.io/en/stable/integrations/editors.html) to run on save. You can run it manually with `black .`.

- We scan the repo for leaked secrets with [gitleaks](https://github.com/gitleaks/gitleaks). You can run it manually with `gitleaks detect`.

- We use [zizmor](https://woodruffw.github.io/zizmor/) for static analysis and security audits of github actions. You can run it manually with `zizmor .`.

- We use [pytest](https://flask.palletsprojects.com/en/stable/testing/) for testing. Tests live in the `tests` directory and can be run with `pytest`. You can run a specific test with `pytest tests/test_example.py::test_example`. Place all fixtures in `tests/conftest.py`.

## File Structure

```
server/
├── .github/                     # Actions and Templates
├── scripts/                     # Shell+Python scripts
├── src/                         # Flask server
│   ├── static/                  # Test client
│   ├── feedback.py              # Feedback logic
│   └── server.py                # Routes and setup
├── tests/                       # Automated tests
│   ├── conftest.py              # Pytest fixtures
│   └── test_*.py                # Test files
├── .env.example                 # Example .env file
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── CONTRIBUTING.md              # Guidelines
├── DEVELOPMENT.md               # Development setup
├── LICENSE                      # License information
└── README.md                    # Readme
```

## Branches

`main` is the default branch containing the latest code (this is where pull requests will be merged in).

`test` will automatically deploy to the test environment on push/merge.

`prod` will automatically deploy to the production environment on push/merge.
