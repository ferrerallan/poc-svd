#!/usr/bin/env bash
# * what? - Complements of STAGE 1, redone in a reproducible way during stage 2.
# * why?  - The original logs and scripts of these steps were lost when a reboot cleaned the temporary directory.
# * how?  - Re-runs the environment record, attempt 0, the frozen-body diagnosis, the decomposition and the aggregation.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=doutorado/experimento1
mkdir -p $E/etapa1/logs

# region 1 - Re-run the complements
# what? - Five steps with their logs in etapa1/logs/.
# how?  - The first and last commands write the start and end timestamps.
date -Iseconds > $E/etapa1/logs/complementos_inicio.txt
python3 $E/common/record_environment.py --output $E/etapa1/ambiente.json
python3 -u $E/common/train.py --seed 0 --body-lr 5e-5 --warmup-epochs 0 --output models/etapa1_tentativa0 \
   > $E/etapa1/logs/treino_s0_tentativa0_reexecucao.log 2>&1
python3 $E/etapa1/scripts/diagnose_frozen_body.py 2>&1 | grep -v "Loading weights" > $E/etapa1/logs/diagnostico.log
python3 $E/common/weight_decomposition.py --stage 1 > $E/etapa1/logs/decomposicao.log 2>&1
python3 $E/common/aggregate_stage.py --stage 1 > $E/etapa1/logs/agregar.log 2>&1
date -Iseconds > $E/etapa1/logs/complementos_fim.txt
# endregion
