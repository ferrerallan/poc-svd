#!/usr/bin/env bash
# * what? - Recipe of STAGE 6 (session 17 of the diary): the sigma adapter predicted from the footprint of free fine-tuning, measured with the frozen-body protocol.
# * why?  - Decides whether the SVD measurement of a free fine-tuning predicts where sigma should work, or only resembles it.
# * how?  - One read-only script over the models of stages 3 and 5; skipped when its JSON exists.
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=doutorado/experimento1; E6=$E/etapa6; L=$E6/logs
mkdir -p $L; date -Iseconds >> $L/execucoes.txt

# region 1 - Environment and the adapter test
# what? - Records the environment and runs the predicted-adapter test.
# how?  - The test writes etapa6/adaptador_previsto.json.
python3 $E/common/record_environment.py --output $E6/ambiente.json > $L/ambiente.log 2>&1
[ -f $E6/adaptador_previsto.json ] || python3 -u $E6/scripts/predicted_sigma_adapter.py > $L/adaptador_previsto.log 2>&1
echo "end $(date -Iseconds)" >> $L/execucoes.txt
# endregion
