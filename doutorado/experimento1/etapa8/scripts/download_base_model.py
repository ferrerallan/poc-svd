"""
* what? - Downloads the original DINOv2-Base model from HuggingFace into models/dinov2-base/.
* why?  - Stage 8 (session 19 of the diary) repeats the skin protocol on a larger model; these are its BEFORE weights, never modified.
* how?  - huggingface_hub.snapshot_download of facebook/dinov2-base into a local folder, keeping only the safetensors weights and the JSON files.

Run from the repository root:  python3 doutorado/experimento1/etapa8/scripts/download_base_model.py
"""
from huggingface_hub import snapshot_download


def main():
    """
    * what? - Downloads the model snapshot and prints where it went.
    * why?  - Keeps a local, never-modified copy of the pre-trained weights, like models/dinov2-small.
    * how?  - snapshot_download with local_dir and allow_patterns; the duplicate pytorch_model.bin is skipped to save disk.
    """
    #region 1 - Download the snapshot
    # what? - Fetches facebook/dinov2-base.
    # how?  - Files already present are not downloaded again.
    path = snapshot_download("facebook/dinov2-base", local_dir="models/dinov2-base",
                             allow_patterns=["*.json", "*.safetensors", "README.md", ".gitattributes"])
    print("downloaded to:", path)
    #endregion


if __name__ == "__main__":
    main()
