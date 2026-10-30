#!/usr/bin/env bash
# * what? - Recipe of session 8 of the diary: 10 trainings of the head only (body frozen for all 5 epochs) on Oxford-IIIT Pet.
# * why?  - Shows that the head alone reaches a good accuracy and that the body does not change.
# * how?  - Runs common/train.py with --warmup-epochs 5 for seeds 0 to 9, then summarizes the runs.
#
# These are the exact commands that were run in the session (one script, run once). Running this file again OVERWRITES models/etapa1_so_cabeca/.
set -euo pipefail
cd "$(dirname "$0")/../.."                       # repository root
E=doutorado/experimento1
mkdir -p $E/etapa1/logs $E/etapa1/estudos models/etapa1_so_cabeca

# region 1 - Train the head with the body frozen
# what? - Ten models in models/etapa1_so_cabeca/s0 to s9 and their logs.
# how?  - As many warm-up epochs as epochs (5) keeps the body frozen from start to end.
for S in 0 1 2 3 4 5 6 7 8 9; do
  python3 -u $E/common/train.py --seed $S --epochs 5 --warmup-epochs 5 --output models/etapa1_so_cabeca/s$S \
     > $E/etapa1/logs/treino_so_cabeca_s$S.log 2>&1
done
# endregion

# region 2 - Summarize the ten runs
# what? - estudos/so_cabeca.json and its log: accuracy per seed and proof that no body tensor changed.
# how?  - head_only_summary.py compares every body tensor with the original model.
python3 $E/etapa1/scripts/head_only_summary.py | tee $E/etapa1/estudos/so_cabeca.log
# endregion
