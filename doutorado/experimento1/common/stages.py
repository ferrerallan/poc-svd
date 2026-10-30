"""
* what? - Path map for stages 1 and 2 of experiment 1: where models, comparisons and aggregates live.
* why?  - One place for the layout, so the analysis scripts of both stages never disagree about paths.
* how?  - Constants plus three small resolvers; stage 1 reads models/dinov2-small-pets*, stage 2 models/etapa2.

Stage 1: weight_decay = 0.05   models in models/dinov2-small-pets*   results in doutorado/experimento1/etapa1/
Stage 2: weight_decay = 0      models in models/etapa2/s0..s9        results in doutorado/experimento1/etapa2/
"""
from pathlib import Path

EXPERIMENT = Path("doutorado/experimento1")
SEEDS = list(range(10))
WEIGHT_DECAY = {1: 0.05, 2: 0.0}
STAGE_DIR = {1: EXPERIMENT / "etapa1", 2: EXPERIMENT / "etapa2"}
REPRODUCIBILITY = {1: STAGE_DIR[1] / "reprodutibilidade.json", 2: STAGE_DIR[2] / "reprodutibilidade.json"}
# stage 1 keeps its 55 comparison files in a subfolder; stage 2 keeps its 10 next to the other results
COMPARISON_DIR = {1: STAGE_DIR[1] / "comparacoes", 2: STAGE_DIR[2]}
TEN_SEEDS_JSON = STAGE_DIR[1] / "resultado_10sementes.json"


def model(stage, seed):
    """
    * what? - Returns the safetensors path of the fine-tuned model of a stage and seed.
    * why?  - Stage 1 used legacy folder names (dinov2-small-pets, -s1..-s9) that stage 2 did not repeat.
    * how?  - Branches on the stage number and formats the folder name from the seed.
    """
    #region 1 - Resolve the model path
    # what? - Picks the folder naming of the requested stage.
    # how?  - Seed 0 of stage 1 has no suffix; every other case follows the sN pattern.
    if stage == 1:
        return "models/dinov2-small-pets/model.safetensors" if seed == 0 else f"models/dinov2-small-pets-s{seed}/model.safetensors"
    return f"models/etapa2/s{seed}/model.safetensors"
    #endregion


def model_dir(stage, seed):
    """
    * what? - Returns the folder that holds the model of a stage and seed, with its relatorio.json.
    * why?  - Scripts read the training report that sits next to the weights.
    * how?  - Takes the parent of the safetensors path.
    """
    #region 1 - Take the parent folder
    # what? - Derives the folder from the safetensors path.
    # how?  - Reuses model() so the naming rule lives in one place.
    return Path(model(stage, seed)).parent
    #endregion


def comparison(stage, seed):
    """
    * what? - Returns the JSON written by compare.py for original model -> model trained with this seed.
    * why?  - Stage 1 seed 0 is named resultado.json for historical reasons; the others follow resultado_sN.json.
    * how?  - Branches on the stage and builds the path inside that stage's comparison folder.
    """
    #region 1 - Resolve the comparison path
    # what? - Applies the naming rule of the requested stage.
    # how?  - The legacy name resultado.json is kept so the frozen files are found without renaming.
    if stage == 1:
        return COMPARISON_DIR[1] / ("resultado.json" if seed == 0 else f"resultado_s{seed}.json")
    return COMPARISON_DIR[2] / f"resultado_s{seed}.json"
    #endregion
