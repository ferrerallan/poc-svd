"""
* what? - Constants and paths of STAGE 4: RESTRICTED fine-tuning (sigma only / rotation only / 72 matrices frozen).
* why?  - One place for the variant list, the step grid and the stage 3 data paths, shared by every stage 4 script.
* how?  - Pure constants plus two path helpers; the repository root is derived from this file location.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "doutorado/experimento1"
STAGE3, STAGE4 = EXPERIMENT / "etapa3", EXPERIMENT / "etapa4"
MODELS = ROOT / "models/etapa4"
ORIGINAL_MODEL = ROOT / "models/dinov2-small"
SEEDS = list(range(10))
WEIGHT_DECAY = 0.0
# stage 3 domain (skin lesions), same sub-sample and same files
SKIN_TRAIN = STAGE3 / "dados/pele/data/train-00000-of-00001-2b84ce2361bc8c42.parquet"
SKIN_TEST = STAGE3 / "dados/pele/data/test-00000-of-00001-3a9b253c406a4af7.parquet"
VARIANTS = ["congelado_72", "so_sigma", "so_rotacao"]          # 'completo' = stage 3 models (models/etapa3/s0..s9), reused
OFT_BLOCK = 32                                                   # block size of the orthogonal rotation (Cayley, block-diagonal)
VALIDATION_FRACTION = 0.10                                       # step pilots: 10% of the train set becomes validation
LR_GRID = {"so_sigma": ["1e-5", "1e-4", "1e-3", "1e-2"], "so_rotacao": ["1e-5", "1e-4", "1e-3"]}   # same strings as reproduce_etapa4.sh (folder names)
PILOT_PREFIX = {"so_sigma": "piloto_so_sigma_lr", "so_rotacao": "piloto_so_rotacao_exp_lr"}   # the Cayley rotation pilots stay stored but are not used


def variant_model(variant, seed):
    """
    * what? - Returns the safetensors path of a restricted variant trained with a seed.
    * why?  - Every stage 4 measurement loads these models.
    * how?  - Folder name is <variant>_s<seed> inside models/etapa4.
    """
    #region 1 - Resolve the path
    # what? - Builds the model path.
    # how?  - Variant names are frozen data names (so_sigma, so_rotacao, congelado_72).
    return MODELS / f"{variant}_s{seed}" / "model.safetensors"
    #endregion


def full_model(seed):
    """
    * what? - Returns the safetensors path of the stage 3 fully fine-tuned model of a seed (read-only).
    * why?  - The 'completo' variant of stage 4 reuses the stage 3 models instead of retraining.
    * how?  - Folder name is s<seed> inside models/etapa3.
    """
    #region 1 - Resolve the path
    # what? - Builds the model path.
    # how?  - Stage 3 models are never modified.
    return ROOT / f"models/etapa3/s{seed}/model.safetensors"
    #endregion
