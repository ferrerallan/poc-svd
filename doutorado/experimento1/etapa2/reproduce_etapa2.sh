#!/usr/bin/env bash
# * what? - Recipe of STAGE 2 of experiment 1: the SAME protocol as stage 1, changing only weight_decay from 0.05 to 0.
# * why?  - Tests whether part of the weight change was an optimizer effect (weight decay) and not an effect of the task.
# * how?  - Trains 10 seeds with --weight-decay 0, checks repeatability, compares, analyses, aggregates and rebuilds the pages.
#
# Everything else (data, model, steps 1e-5/1e-3, 5 epochs, warm-up, label smoothing, seeds 0 to 9) is identical to stage 1.
# Running this file retrains and OVERWRITES models/etapa2/.
set -euo pipefail
cd "$(dirname "$0")/../.."                       # repository root
E=doutorado/experimento1
mkdir -p $E/etapa2/logs
date -Iseconds > $E/etapa2/logs/inicio.txt

# region 1 - Record the environment
# what? - Versions, machine, git and hashes of this run.
# how?  - Written to etapa2/ambiente.json.
python3 $E/common/record_environment.py --output $E/etapa2/ambiente.json
# endregion

# region 2 - Train the 10 seeds with weight_decay = 0
# what? - Seeds 0 to 9, one log per seed.
# how?  - The only difference from stage 1 is --weight-decay 0.
for S in 0 1 2 3 4 5 6 7 8 9; do
  python3 -u $E/common/train.py --seed $S --weight-decay 0 --output models/etapa2/s$S > $E/etapa2/logs/treino_s$S.log 2>&1
done
# endregion

# region 3 - Repeatability check
# what? - Does seed 0 trained a second time give the same model?
# how?  - Trains seed 0 again into models/etapa2/s0b and compares tensor by tensor.
python3 -u $E/common/train.py --seed 0 --weight-decay 0 --output models/etapa2/s0b > $E/etapa2/logs/treino_s0b_repeticao.log 2>&1
python3 $E/etapa2/scripts/verify_repeat.py > $E/etapa2/logs/verificar_repeticao.log 2>&1
# endregion

# region 4 - Compare original -> each training
# what? - SVD of the matrices of both models and the comparison measures.
# how?  - One JSON per seed, next to the other stage 2 results.
for S in 0 1 2 3 4 5 6 7 8 9; do
  python3 $E/common/compare.py --before models/dinov2-small/model.safetensors --after models/etapa2/s$S/model.safetensors \
     --output $E/etapa2/resultado_s$S.json > $E/etapa2/logs/comparar_s$S.log 2>&1
done
# endregion

# region 5 - Analyses, aggregation and pages
# what? - Agreement, decomposition, whole-matrix cosine, aggregate and the stage 1-2 pages.
# how?  - build_diary.py writes diario.html and resultados.html in doutorado/experimento1/.
python3 $E/common/reproducibility_weights_directions.py --stage 2 > $E/etapa2/logs/reprodutibilidade.log 2>&1
python3 $E/common/weight_decomposition.py --stage 2            > $E/etapa2/logs/decomposicao.log 2>&1
python3 $E/common/whole_matrix_cosine.py --stage 2             > $E/etapa2/logs/cosseno_matriz.log 2>&1
python3 $E/common/aggregate_stage.py --stage 2                 > $E/etapa2/logs/agregar.log 2>&1
python3 $E/etapa2/scripts/build_diary.py                       > $E/etapa2/logs/construir_diario.log 2>&1
date -Iseconds > $E/etapa2/logs/fim.txt
# endregion
