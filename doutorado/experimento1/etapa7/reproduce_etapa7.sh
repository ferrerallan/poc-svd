#!/usr/bin/env bash
# * what? - Recipe of STAGE 7 (session 18 of the diary): the closed protocol on a SECOND domain, KMNIST: step pilots, then 10 seeds of frozen 72, sigma only, small-block rotation and full training.
# * why?  - First repetition of the skin findings in another domain; the ordering sigma > small rotation and the recovery fractions are what is being tested.
# * how?  - Idempotent: each step is skipped if its output exists. Uses train_etapa7.py, a thin wrapper that runs train_etapa4.py on the KMNIST files.
#
# Nothing from stages 1-6 is changed; everything goes to etapa7/ and models/etapa7/. Needs the GPU (about 40 trainings of 1 to 2 minutes).
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=doutorado/experimento1; E7=$E/etapa7; L=$E7/logs; M=models/etapa7; T=$E7/scripts/train_etapa7.py
mkdir -p $L $M; date -Iseconds >> $L/execucoes.txt

# region 1 - Environment
# what? - Records versions, machine, git and hashes.
# how?  - Written to etapa7/ambiente.json.
python3 $E/common/record_environment.py --output $E7/ambiente.json > $L/ambiente.log 2>&1
# endregion

# region 2 - Step pilots (seed 0, 10% of the train set as validation; the test set is NOT used in the choice)
# what? - Sigma-only over four steps and the 3-line-block rotation over three.
# how?  - Same pilot protocol as stages 4 and 5 (--validation 0.1).
for LR in 1e-3 1e-2 3e-2 1e-1; do
  [ -f $M/piloto_so_sigma_lr$LR/relatorio.json ] || python3 -u $T --variant so_sigma --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_sigma_lr$LR > $L/piloto_so_sigma_lr$LR.log 2>&1
done
for LR in 1e-4 1e-3 1e-2; do
  [ -f $M/piloto_so_rotacao_bloco3_lr$LR/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param exp --oft-block 3 --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_rotacao_bloco3_lr$LR > $L/piloto_so_rotacao_bloco3_lr$LR.log 2>&1
done
# endregion

# region 3 - Choose the steps by validation -> etapa7/passos_escolhidos.json
# what? - Best step of each restricted variant on this domain.
# how?  - choose_step_etapa7.py reads the pilots of this stage.
python3 $E7/scripts/choose_step_etapa7.py > $L/escolher_passo.log 2>&1
LR_SIG=$(python3 -c "import json;print(json.load(open('$E7/passos_escolhidos.json'))['so_sigma']['lr'])")
LR_ROT=$(python3 -c "import json;print(json.load(open('$E7/passos_escolhidos.json'))['so_rotacao_bloco3']['lr'])")
# endregion

# region 4 - 10 seeds per variant (test set), with the chosen steps
# what? - 40 trainings: frozen 72, sigma only, small-block rotation and full.
# how?  - A training is done when its relatorio.json exists.
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $M/congelado_72_s$S/relatorio.json ]      || python3 -u $T --variant congelado_72 --seed $S --output $M/congelado_72_s$S > $L/treino_congelado_72_s$S.log 2>&1
  [ -f $M/so_sigma_s$S/relatorio.json ]          || python3 -u $T --variant so_sigma --restricted-lr $LR_SIG --seed $S --output $M/so_sigma_s$S > $L/treino_so_sigma_s$S.log 2>&1
  [ -f $M/so_rotacao_bloco3_s$S/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param exp --oft-block 3 --restricted-lr $LR_ROT --seed $S --output $M/so_rotacao_bloco3_s$S > $L/treino_so_rotacao_bloco3_s$S.log 2>&1
  [ -f $M/completo_s$S/relatorio.json ]          || python3 -u $T --variant completo --seed $S --output $M/completo_s$S > $L/treino_completo_s$S.log 2>&1
done
# endregion

# region 5 - Summary of the stage
# what? - Accuracies, recovery fractions and the restriction checks, next to the skin numbers.
# how?  - summarize_etapa7.py writes etapa7/resumo_etapa7.json.
python3 -u $E7/scripts/summarize_etapa7.py > $L/resumo.log 2>&1
echo "end $(date -Iseconds)" >> $L/execucoes.txt
# endregion
