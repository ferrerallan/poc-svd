"""
* what? - Downloads the original DINOv2-Small model from HuggingFace into models/dinov2-small/.
* why?  - Step 1 of everything: the BEFORE weights that every later comparison starts from.
* how?  - huggingface_hub.snapshot_download of facebook/dinov2-small into a local folder.

Run from the repository root:  python3 doutorado/experimento1/etapa1/scripts/download_model.py
"""
from huggingface_hub import snapshot_download


def main():
    """
    * what? - Downloads the model snapshot and prints where it went.
    * why?  - Keeps a local, never-modified copy of the pre-trained weights.
    * how?  - snapshot_download with local_dir, so the files are real copies and not cache links.
    """
    #region 1 - Download the snapshot
    # what? - Fetches facebook/dinov2-small.
    # how?  - Files already present are not downloaded again.
    path = snapshot_download("facebook/dinov2-small", local_dir="models/dinov2-small")
    print("downloaded to:", path)
    #endregion


if __name__ == "__main__":
    main()
