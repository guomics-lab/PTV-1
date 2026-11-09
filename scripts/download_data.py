"""Download the ProteinTalks dataset from the public Google Drive folder."""

import argparse
from pathlib import Path

import gdown

DATASET_URL = "https://drive.google.com/drive/folders/1QRV-bD3zabVkFjznLR--Cca90fdU6Pr6"
DEFAULT_OUTPUT = Path("data/raw")


def list_files(url: str) -> list[str]:
    """Return relative paths in the Drive folder without downloading them."""
    entries = gdown.download_folder(url=url, skip_download=True, quiet=True)
    return [entry.path for entry in entries]


def download_dataset(url: str, output: Path) -> list[str]:
    """Download the folder into output. Existing files are resumed."""
    output.mkdir(parents=True, exist_ok=True)
    return gdown.download_folder(
        url=url,
        output=str(output),
        resume=True,
        quiet=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ProteinTalks data from Google Drive."
    )
    parser.add_argument("--url", default=DATASET_URL, help="Google Drive folder URL")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for the downloaded files",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print file names and do not download",
    )
    args = parser.parse_args()

    if args.list:
        for path in list_files(args.url):
            print(path)
        return

    saved = download_dataset(args.url, args.output)
    print(f"Downloaded {len(saved)} files to {args.output}")


if __name__ == "__main__":
    main()
