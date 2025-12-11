import logging
import os
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)


def download_gtzan():
    logger.info("Downloading GTZAN dataset from Kaggle")
    try:
        path = kagglehub.dataset_download("murataktan/gtzan-fixed")
        logger.info(f"Downloaded to cache: {path}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return

    project_root = Path(__file__).parent
    target_dir = project_root / "Data" / "genres_original"

    # Check is the target directory already exists and contains data
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info(f"GTZAN dataset already exists at {target_dir}, skipping extraction.")
        return str(target_dir)

    logger.info(f"Extracting GTZAN dataset to {target_dir}")
    os.makedirs(target_dir.parent, exist_ok=True)

    source_path = Path(path)

    if (source_path / "Data" / "genres_original").exists():
        source_data = source_path / "Data" / "genres_original"
    elif (source_path / "genres_original").exists():
        source_data = source_path / "genres_original"
    else:
        source_data = source_path

    try:
        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.copytree(source_data, target_dir)
        logger.info(f"GTZAN dataset extracted to {target_dir}")
        return str(target_dir)
    except Exception as e:
        logger.error(f"Failed to extract dataset: {e}")
        return


if __name__ == "__main__":
    download_gtzan()
