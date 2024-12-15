<img width="100%" alt="KoelLabsLogoLong" src="https://github.com/user-attachments/assets/b8232261-eb8f-40a8-a33e-630eca206c9f">

[![Mozilla Builders](https://img.shields.io/badge/Mozilla-000000.svg?style=for-the-badge&logo=Mozilla&logoColor=white)](https://future.mozilla.org/builders/)
![Patreon](https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white)
![PayPal](https://img.shields.io/badge/PayPal-00457C?style=for-the-badge&logo=paypal&logoColor=white)

# Koel Labs - Server
![Black Formatting](https://github.com/KoelLabs/server/actions/workflows/black.yml/badge.svg)
![PyTest](https://github.com/KoelLabs/server/actions/workflows/tests.yml/badge.svg)
![Zizmor](https://github.com/KoelLabs/server/actions/workflows/zizmor.yml/badge.svg)
![Gitleaks Secret Scanning](https://github.com/KoelLabs/server/actions/workflows/gitleaks.yml/badge.svg)
![Deploy to Production](https://github.com/KoelLabs/server/actions/workflows/prod.yml/badge.svg)

Contains the Python server that runs our ML inference. It is used by our [webapp](https://github.com/KoelLabs/webapp) repository and can be run directly as part of the webapp following instructions in that repository.

Read about all our repositories [here](https://github.com/KoelLabs).

## Setup

See the [DEVELOPMENT.md](DEVELOPMENT.md) file for instructions on how to set up the server for development as well as for alternative setup instructions.

0. `git clone https://github.com/KoelLabs/server.git`
1. Install [Docker and Docker Compose](https://www.docker.com/get-started/)
2. Duplicate the `.env.example` file and rename it to `.env`. Fill in the necessary environment variables.
    - You can find your `HF_TOKEN` on your [Settings Page](https://huggingface.co/settings/tokens). It just needs read access to `gated repos`.
3. Run the application
    - `. ./scripts/docker-run-dev.sh` to start the development server
    - If `http://localhost:8080` doesn't automatically open in your browser, open it manually
    - `ctrl+c` to stop the server

## Contributing

Check out the [CONTRIBUTING.md](CONTRIBUTING.md) file for specific guidelines on contributing to this repository.

## License

The code in this repository is licensed under the [GNU Affero General Public License](https://www.gnu.org/licenses/agpl-3.0.en.html).

We retain all rights to the Koel Labs brand, logos, blog posts and website content.
