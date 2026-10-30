#!/usr/bin/env bash
# * what? - Recipe of STAGE 1 of experiment 1: fine-tune DINOv2-Small on 37 breeds (Oxford-IIIT Pet) with weight_decay = 0.05.
# * why?  - Documents every command that produced the stage 1 models and results, in order and with the exact parameters.
# * how?  - Downloads the model, trains 10 seeds, compares each against the original, then runs the analyses and aggregation.
#
# HONESTY NOTICE: in stage 1 these commands were run one by one in an interactive session. This file is the EQUIVALENT
# RECIPE, rebuilt later with the same parameters (the defaults of common/train.py). Only the complements were actually
# re-run afterwards (see complements_etapa1.sh). Running this file retrains and OVERWRITES models/dinov2-small-pets*.
set -euo pipefail
cd "$(dirname "$0")/../.."                       # repository root
E=doutorado/experimento1
mkdir -p $E/etapa1/logs $E/etapa1/comparacoes

# region 1 - Download the original model
# what? - Fetches facebook/dinov2-small into models/dinov2-small.
# how?  - The dataset is downloaded by train.py on its first run.
python3 $E/etapa1/scripts/download_model.py
# endregion

# region 2 - Attempt 0 and the frozen-body diagnosis
# what? - Attempt 0 (discarded): body unfrozen from the start and step 5e-5; then the frozen-body accuracy.
# how?  - Zero warm-up epochs reproduce the attempt that failed; the diagnosis shows the body was fine.
python3 -u $E/common/train.py --seed 0 --body-lr 5e-5 --warmup-epochs 0 --output models/etapa1_tentativa0 \
   > $E/etapa1/logs/treino_s0_tentativa0_reexecucao.log 2>&1
python3 $E/etapa1/scripts/diagnose_frozen_body.py
# endregion

# region 3 - Train the 10 seeds
# what? - Standard protocol: 1 warm-up epoch, step 1e-5, weight_decay 0.05, seeds 0 to 9.
# how?  - Seed 0 keeps the legacy folder name models/dinov2-small-pets.
python3 -u $E/common/train.py --seed 0 --output models/dinov2-small-pets    > $E/etapa1/logs/treino_s0.log 2>&1
for S in 1 2 3 4 5 6 7 8 9; do
  python3 -u $E/common/train.py --seed $S --output models/dinov2-small-pets-s$S > $E/etapa1/logs/treino_s$S.log 2>&1
done
# endregion

# region 4 - Compare against the original and between trainings
# what? - original -> each training (10) and training i -> training j (45 pairs, used only in the first analysis).
# how?  - M() prints the model path of a seed; the (0, 1) pair keeps the legacy file name resultado_ruido.json.
ORIG=models/dinov2-small/model.safetensors
M() { if [ "$1" = 0 ]; then echo models/dinov2-small-pets/model.safetensors; else echo models/dinov2-small-pets-s$1/model.safetensors; fi; }
python3 $E/common/compare.py --before $ORIG --after $(M 0) --output $E/etapa1/comparacoes/resultado.json > /dev/null
for S in 1 2 3 4 5 6 7 8 9; do python3 $E/common/compare.py --before $ORIG --after $(M $S) --output $E/etapa1/comparacoes/resultado_s$S.json > /dev/null; done
for I in 0 1 2 3 4 5 6 7 8 9; do for J in 0 1 2 3 4 5 6 7 8 9; do
  [ $I -lt $J ] || continue
  OUT=$E/etapa1/comparacoes/ruido_${I}_${J}.json; [ "$I $J" = "0 1" ] && OUT=$E/etapa1/comparacoes/resultado_ruido.json
  python3 $E/common/compare.py --before $(M $I) --after $(M $J) --output $OUT > /dev/null
done; done
# endregion

# region 5 - Analyses, dataset exploration and environment
# what? - Seed analysis, agreement, weight decomposition, whole-matrix cosine, aggregation, dataset exploration, environment.
# how?  - Each script writes its JSON inside etapa1/; the exploration also writes the article figures 15 to 18.
python3 $E/etapa1/scripts/analyze_seeds.py
python3 $E/common/reproducibility_weights_directions.py --stage 1
python3 $E/common/weight_decomposition.py --stage 1
python3 $E/common/whole_matrix_cosine.py --stage 1
python3 $E/common/aggregate_stage.py --stage 1
python3 $E/etapa1/scripts/oxfordpet_exploration.py
python3 $E/common/record_environment.py --output $E/etapa1/ambiente.json
# endregion
