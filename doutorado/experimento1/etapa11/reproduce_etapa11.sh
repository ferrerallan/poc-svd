#!/usr/bin/env bash
# * what? - Recipe of STAGE 11 (session 22 of the diary, the checkpoint): re-checks by code the claims that hold in all five model and domain pairs.
# * why?  - The checkpoint must come from the stored numbers of stages 4, 5, 7, 8, 9 and 10, not from memory; the script fails if any claim stops holding.
# * how?  - Idempotent and GPU free: records the environment and runs check_claims_etapa11.py, which writes etapa11/afirmacoes_etapa11.json.
#
# Nothing from stages 1-10 is changed; everything goes to etapa11/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=doutorado/experimento1; E11=$E/etapa11; L=$E11/logs
mkdir -p $L; date -Iseconds >> $L/execucoes.txt

# region 1 - Environment
# what? - Records versions, machine, git and hashes.
# how?  - Written to etapa11/ambiente.json.
python3 $E/common/record_environment.py --output $E11/ambiente.json > $L/ambiente.log 2>&1
# endregion

# region 2 - Check the claims -> etapa11/afirmacoes_etapa11.json
# what? - Nine claims evaluated over the five pairs and the two GPU cost files.
# how?  - Exit code 1 stops the recipe if a claim fails.
python3 $E11/scripts/check_claims_etapa11.py | tee $L/afirmacoes.log
# endregion
echo "end $(date -Iseconds)"
