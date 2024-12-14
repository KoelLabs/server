<img width="100%" alt="KoelLabsLogoLong" src="https://github.com/user-attachments/assets/b8232261-eb8f-40a8-a33e-630eca206c9f">

[![Mozilla Builders](https://img.shields.io/badge/Mozilla-000000.svg?style=for-the-badge&logo=Mozilla&logoColor=white)](https://future.mozilla.org/builders/)
![Patreon](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)
![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)

# Koel Labs - Server

Contains the Python server that runs our ML inference. It is used by our [webapp](https://github.com/KoelLabs/webapp) repository and can be run directly as part of the webapp following instructions in that repository.

Read about all our repositories [here](https://github.com/KoelLabs).

## Development

### Run with Docker (Easiest Setup)

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
    - If `http://localhost:8080` didn't automatically open in your browser, open it manually
    - `ctrl+c` to stop the server

To add new dependencies, update the `requirements.txt` file and run `. ./scripts/docker-run-dev.sh` again.

### Run directly on your machine (Runs Fastest)

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
4. Run the server
    - Run `python src/server.py` to start the development server
    - Open your browser to `http://localhost:8080`
    - `ctrl+c` to stop the server

To save dependencies you `pip install`, run `pip freeze > requirements.txt`.

### Formatting, Linting, Automated Tests and Secret Scanning

TODO: Alex will add

### File Structure

TODO: Alex will add

## Deployment

TODO: Alex will add

## Contributing

TODO: Alex will add contribution guidelines and PR template/consent form.

## License

The code in this repository is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

We retain all rights to the Koel Labs brand, logos, blog posts and website content.
