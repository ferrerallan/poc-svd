#!/usr/bin/env bash
# * what? - Recipe of STAGE 5 (session 16 of the diary): the rotation with the same number of trainable numbers as sigma-only, and larger steps for sigma-only, on skin lesions.
# * why?  - Stage 4 compared sigma-only (27,648 numbers) with a rotation of 642,816 numbers; this makes the comparison at equal cost and checks whether sigma was limited by its step.
# * how?  - Idempotent: each step is skipped if its output exists. Reuses train_etapa4.py unchanged, with --oft-block 3.
#
# Nothing from stages 1-4 is changed; everything goes to etapa5/ and models/etapa5/.
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=doutorado/experimento1; E5=$E/etapa5; L=$E5/logs; M=models/etapa5; T=$E/etapa4/scripts/train_etapa4.py
mkdir -p $L $M; date -Iseconds >> $L/execucoes.txt

# region 1 - Environment
# what? - Records versions, machine, git and hashes.
# how?  - Written to etapa5/ambiente.json.
python3 $E/common/record_environment.py --output $E5/ambiente.json > $L/ambiente.log 2>&1
# endregion

# region 2 - Step pilots (seed 0, 10% of the train set as validation; the test set is NOT used in the choice)
# what? - Rotation in 3-line blocks (41,472 effective numbers) over four steps; sigma-only over two steps larger than the stage 4 grid.
# how?  - Same pilot protocol as stage 4 (--validation 0.1).
for LR in 1e-5 1e-4 1e-3 1e-2; do
  [ -f $M/piloto_rotacao3_lr$LR/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param exp --oft-block 3 --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_rotacao3_lr$LR > $L/piloto_rotacao3_lr$LR.log 2>&1
done
for LR in 3e-2 1e-1; do
  [ -f $M/piloto_so_sigma_lr$LR/relatorio.json ] || python3 -u $T --variant so_sigma --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_sigma_lr$LR > $L/piloto_so_sigma_lr$LR.log 2>&1
done
# endregion

# region 3 - Choose the steps by validation -> etapa5/passos_escolhidos.json
# what? - Best step of the small-block rotation, and whether a larger sigma step beat the stage 4 choice.
# how?  - choose_step_etapa5.py reads the pilots of stages 4 and 5.
python3 $E5/scripts/choose_step_etapa5.py > $L/escolher_passo.log 2>&1
LR_ROT=$(python3 -c "import json;print(json.load(open('$E5/passos_escolhidos.json'))['so_rotacao_bloco3']['lr'])")
LR_SIG=$(python3 -c "import json;print(json.load(open('$E5/passos_escolhidos.json'))['so_sigma']['lr'])")
SIG_NEW=$(python3 -c "import json;print(json.load(open('$E5/passos_escolhidos.json'))['so_sigma']['novo_passo_venceu'])")
# endregion

# region 4 - 10 seeds of the small-block rotation, and of sigma-only only if a new step won
# what? - The final trainings of the stage, on the test set.
# how?  - A training is done when its relatorio.json exists; the sigma retraining is skipped when 1e-2 stays the best step.
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $M/so_rotacao_bloco3_s$S/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param exp --oft-block 3 --restricted-lr $LR_ROT --seed $S --output $M/so_rotacao_bloco3_s$S > $L/treino_so_rotacao_bloco3_s$S.log 2>&1
  if [ "$SIG_NEW" = "True" ]; then
    [ -f $M/so_sigma_lr${LR_SIG}_s$S/relatorio.json ] || python3 -u $T --variant so_sigma --restricted-lr $LR_SIG --seed $S --output $M/so_sigma_lr${LR_SIG}_s$S > $L/treino_so_sigma_lr${LR_SIG}_s$S.log 2>&1
  fi
done
# endregion

# region 5 - Summary of the stage
# what? - Accuracies of the new variants next to the stage 4 ones, and the check that the restrictions held.
# how?  - summarize_etapa5.py writes etapa5/resumo_etapa5.json.
python3 -u $E5/scripts/summarize_etapa5.py > $L/resumo.log 2>&1
echo "end $(date -Iseconds)" >> $L/execucoes.txt
# endregion
