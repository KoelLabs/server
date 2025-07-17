import json
import subprocess

CMD = "pipx run licensecheck --requirements-paths ./requirements.txt --format json --only-licenses 'PUBLIC DOMAIN' UNLICENSE BOOST MIT BSD ISC NCSA PYTHON APACHE ECLIPSE LGPL AGPL GPL MPL EUPL"
result = subprocess.run(CMD, shell=True, check=True, stdout=subprocess.PIPE)
licenses: list[dict] = json.loads(result.stdout)["packages"]

dependencies = []
with open("requirements.txt") as f:
    dependencies = [l.split("==")[0] for l in f.readlines()]

licenses = [l for l in licenses if l["name"] in dependencies]

incompatible = [
    l
    for l in licenses
    if not l["licenseCompat"] and l["license"] != "MOZILLA PUBLIC LICENSE 2.0 (MPL 2.0)"
]
if incompatible:
    print("Incompatible licenses found:")
    for l in incompatible:
        print(f"{l['name']}: {l['license']}")
else:
    print("No incompatible licenses found.")
    print("name | license | author | url")
    print("---- | ------- | ------ | ---")
    for l in licenses:
        print(f"{l['name']} | {l['license']} | {l['author']} | {l['homePage']}")
