"""Upload the cropped Alps DEM to Hugging Face Hub.

This uploads the alps_dem.tif file to a Hugging Face dataset for free hosting.
Streamlit Cloud can then download directly from HF instead of using Git LFS bandwidth.

Usage:
1. pip install huggingface_hub
2. Run: python scripts/upload_dem_to_huggingface.py
3. On first run, you'll be prompted to login (creates ~/.huggingface/token)
"""

from huggingface_hub import HfApi, login

from skiresort_planner.constants import DEMConfig


def upload_dem_to_hf() -> None:
    """Upload the Alps DEM file to Hugging Face Hub."""
    dem_file = DEMConfig.EURODEM_PATH
    repo_id = DEMConfig.HF_REPO_ID
    filename = DEMConfig.HF_FILENAME

    if not dem_file.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_file}")

    print(f"File to upload: {dem_file}")
    print(f"File size: {dem_file.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Target repo: {repo_id}")

    # Login (will prompt for token on first use)
    login()

    # Upload the file
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(dem_file),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\nUploaded successfully!")
    print(f"Download URL: {DEMConfig.HF_DOWNLOAD_URL}")


if __name__ == "__main__":
    upload_dem_to_hf()
