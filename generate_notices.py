import subprocess
import sys
import json


def install_pip_licenses():
    """Ensure pip-licenses is installed."""
    try:
        import piplicenses
    except ImportError:
        print("Installing pip-licenses...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pip-licenses"])


def generate_license_report(output_file="THIRD_PARTY_LICENSES.json", fmt="json"):
    """
    Generate a license report for all installed Python packages.
    Supported formats: markdown, json, plain, html, rst, etc.
    """
    print(f"Generating license report in {output_file}...")

    cmd = [
        sys.executable,
        "-m",
        "piplicenses",
        "--from=mixed",
        "--with-urls",
        "--with-license-file",
        "--with-authors",
        f"--format={fmt}",
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, encoding="utf-8", check=True
        )
        print(f"✅ License report generated")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print("❌ Failed to generate license report:", e)
        sys.exit(1)


def format_notices(report):
    packages = json.loads(report)

    line = "=" * 80

    with open("THIRD_PARTY_NOTICES.txt", "w") as output:
        for package in packages:
            name = package["Name"]
            license = package["License"]
            text = package["LicenseText"]
            info = f"""{line}
Package: {name}
License Type: {license}
--------------------------------------------

{text}

"""
            output.write(info)


def main():
    install_pip_licenses()
    report = generate_license_report()
    format_notices(report)


if __name__ == "__main__":
    main()
