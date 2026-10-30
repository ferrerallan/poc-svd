#!/usr/bin/env bash
# * what? - Recipe of STAGE 4 of experiment 1: RESTRICTED fine-tuning (sigma only / rotation only / 72 matrices frozen) on skin lesions.
# * why?  - Measures what each kind of freedom, alone, can learn, with the same protocol as stage 3.
# * how?  - Idempotent: each step is skipped if its output exists. Step pilots choose the learning rate by validation.
#
# Nothing from stages 1-3 is changed; everything goes to etapa4/ and models/etapa4/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=doutorado/experimento1; E4=$E/etapa4; L=$E4/logs; M=models/etapa4; T=$E4/scripts/train_etapa4.py
mkdir -p $L $M; date -Iseconds >> $L/execucoes.txt

# region 1 - Environment
# what? - Records versions, machine, git and hashes.
# how?  - Written to etapa4/ambiente.json.
python3 $E/common/record_environment.py --output $E4/ambiente.json > $L/ambiente.log 2>&1
# endregion

# region 2 - Step pilots (seed 0, 10% of the train set as validation; the test set is NOT used in the choice)
# what? - Pilots for sigma-only and for rotation-only.
# how?  - The first rotation pilots used the Cayley parametrization and are kept; the exponential ones choose the step.
for LR in 1e-5 1e-4 1e-3 1e-2; do
  [ -f $M/piloto_so_sigma_lr$LR/relatorio.json ] || python3 -u $T --variant so_sigma --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_sigma_lr$LR > $L/piloto_so_sigma_lr$LR.log 2>&1
done
for LR in 1e-5 1e-4 1e-3; do
  [ -f $M/piloto_so_rotacao_lr$LR/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param cayley --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_rotacao_lr$LR > $L/piloto_so_rotacao_lr$LR.log 2>&1
done
# rotation pilots redone with the exponential parametrization (the Cayley solver on CUDA failed in a full training); these choose the step
for LR in 1e-5 1e-4 1e-3; do
  [ -f $M/piloto_so_rotacao_exp_lr$LR/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param exp --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_rotacao_exp_lr$LR > $L/piloto_so_rotacao_exp_lr$LR.log 2>&1
done
# endregion

# region 3 - Choose the step by validation -> etapa4/passos_escolhidos.json
# what? - Writes the chosen step of each variant and reads it back.
# how?  - A tiny inline Python call extracts the lr from the JSON.
python3 $E4/scripts/choose_step.py > $L/escolher_passo.log 2>&1
LR_SIG=$(python3 -c "import json;print(json.load(open('$E4/passos_escolhidos.json'))['so_sigma']['lr'])")
LR_ROT=$(python3 -c "import json;print(json.load(open('$E4/passos_escolhidos.json'))['so_rotacao']['lr'])")
# endregion

# region 4 - 10 seeds per variant (test set), with the chosen step
# what? - 30 trainings: frozen 72, sigma only and rotation only.
# how?  - A training is done when its relatorio.json exists.
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $M/congelado_72_s$S/relatorio.json ] || python3 -u $T --variant congelado_72 --seed $S --output $M/congelado_72_s$S > $L/treino_congelado_72_s$S.log 2>&1
  [ -f $M/so_sigma_s$S/relatorio.json ]     || python3 -u $T --variant so_sigma --restricted-lr $LR_SIG --seed $S --output $M/so_sigma_s$S > $L/treino_so_sigma_s$S.log 2>&1
  [ -f $M/so_rotacao_s$S/relatorio.json ]   || python3 -u $T --variant so_rotacao --rotation-param exp --restricted-lr $LR_ROT --seed $S --output $M/so_rotacao_s$S > $L/treino_so_rotacao_s$S.log 2>&1
done
# endregion

# region 5 - Measures of the variants
# what? - Measures what each variant did to the 72 matrices and checks the restrictions.
# how?  - Writes etapa4/medidas_variantes.json.
python3 -u $E4/scripts/measure_variants.py > $L/medir_variantes.log 2>&1
# endregion

# region 6 - GPU cost of each variant (added later, session 14 of the diary)
# what? - Peak memory, optimizer state, gradients and time per step of the four variants.
# how?  - A few training steps on preloaded batches; skipped when custo_gpu.json exists.
[ -f $E4/custo_gpu.json ] || python3 $E4/scripts/measure_gpu_cost.py > $L/custo_gpu.log 2>&1
# endregion

# region 7 - Alignment between free fine-tuning and the restricted trainings (added later, session 15 of the diary)
# what? - Cosine between the sigma footprint of free training and the sigma-only delta, per matrix; same for directions.
# how?  - Read-only over the models; skipped when the JSON exists.
[ -f $E4/correlacao_livre_vs_restrito.json ] || python3 $E4/scripts/correlate_free_vs_restricted.py > $L/correlacao_livre_vs_restrito.log 2>&1
# endregion

# region 8 - The page
# what? - Inserts stage 4 into resultados.html.
# how?  - The builder regenerates the whole page from etapa4/base, so it is safe to run again.
if [ -f $E4/scripts/build_results_etapa4.py ]; then python3 $E4/scripts/build_results_etapa4.py > $L/construir.log 2>&1; else echo "build_results_etapa4.py does not exist yet" >> $L/execucoes.txt; fi
echo "end $(date -Iseconds)" >> $L/execucoes.txt
# endregion
