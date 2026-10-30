"""
* what? - Constants and paths of STAGE 3 (everything the stage writes goes to etapa3/ and models/etapa3/).
* why?  - One place for seeds, paths and sub-sampling constants, so every stage 3 script agrees.
* how?  - Pure constants plus two path helpers; the repository root is derived from this file location.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]           # repository root
EXPERIMENT = ROOT / "doutorado/experimento1"
STAGE3 = EXPERIMENT / "etapa3"                        # this stage
DATA = STAGE3 / "dados"
MODELS = ROOT / "models/etapa3"
ORIGINAL_MODEL = ROOT / "models/dinov2-small"
SEEDS = list(range(10))
WEIGHT_DECAY = 0.0                                    # inherited from stage 2
MAX_TRAIN, MAX_TEST = 3680, 3669                      # same sizes as Oxford-IIIT Pet, so the number of steps is the same
SUBSAMPLE_SEED = 12345                                # fixed: every training seed sees the SAME subset
LABEL_SEED = 777                                      # fixed: the shuffled-labels control uses the SAME permutation in every seed


def stage_model(stage, seed):
    """
    * what? - Returns the safetensors path of the fine-tuned model of a stage (read-only) and seed.
    * why?  - The corrected ruler is applied to the models of stages 1, 2 and 3 alike.
    * how?  - Stage 1 uses its legacy folder names, stage 2 models/etapa2, stage 3 the folders of this stage.
    """
    #region 1 - Resolve the path
    # what? - Picks the naming rule of the stage.
    # how?  - Stage 1 seed 0 has no suffix; later stages use the sN pattern.
    if stage == 1:
        return ROOT / ("models/dinov2-small-pets" if seed == 0 else f"models/dinov2-small-pets-s{seed}") / "model.safetensors"
    if stage == 2:
        return ROOT / f"models/etapa2/s{seed}/model.safetensors"
    return MODELS / f"s{seed}" / "model.safetensors"
    #endregion


def shuffled_model(seed):
    """
    * what? - Returns the safetensors path of the shuffled-labels control model of a seed.
    * why?  - The control trainings live next to the real ones, in models/etapa3/embaralhado_sN.
    * how?  - Formats the folder name from the seed.
    """
    #region 1 - Resolve the path
    # what? - Builds the control model path.
    # how?  - Folder name uses the Portuguese word embaralhado: it is a frozen data name.
    return MODELS / f"embaralhado_s{seed}" / "model.safetensors"
    #endregion
