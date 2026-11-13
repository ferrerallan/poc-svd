#!/usr/bin/env bash
# * what? - Recipe of STAGE 8 (session 19 of the diary): the closed skin protocol on a LARGER model, DINOv2-Base: triage, step pilots, 10 seeds of frozen 72, sigma only, small-block rotation and full training, GPU cost.
# * why?  - First test of whether the skin findings depend on the size of the model; domain, data and protocol are the ones of stages 3 to 5, only the model changes.
# * how?  - Idempotent: each step is skipped if its output exists. Uses train_etapa8.py, a thin wrapper that runs train_etapa4.py with models/dinov2-base.
#
# Nothing from stages 1-7 is changed; everything goes to etapa8/ and models/etapa8/. Needs the GPU (about 47 trainings, several minutes each).
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=doutorado/experimento1; E8=$E/etapa8; L=$E8/logs; M=models/etapa8; T=$E8/scripts/train_etapa8.py
mkdir -p $L $M; date -Iseconds >> $L/execucoes.txt

# region 1 - Environment and the original Base model
# what? - Records versions, machine, git and hashes; downloads facebook/dinov2-base once.
# how?  - Written to etapa8/ambiente.json; the model goes to models/dinov2-base and is never modified.
python3 $E/common/record_environment.py --output $E8/ambiente.json > $L/ambiente.log 2>&1
[ -f models/dinov2-base/model.safetensors ] || python3 $E8/scripts/download_base_model.py > $L/download.log 2>&1
# endregion

# region 2 - Frozen-body triage of the Base on skin
# what? - CLS features of the frozen Base plus logistic regression, the instrument of session 12.
# how?  - screen_skin_base.py writes etapa8/triagem_base.json.
[ -f $E8/triagem_base.json ] || python3 -u $E8/scripts/screen_skin_base.py > $L/triagem_base.log 2>&1
# endregion

# region 3 - Step pilots (seed 0, 10% of the train set as validation; the test set is NOT used in the choice)
# what? - Sigma-only over four steps and the 3-line-block rotation over three.
# how?  - Same pilot protocol as stages 4, 5 and 7 (--validation 0.1).
for LR in 1e-3 1e-2 3e-2 1e-1; do
  [ -f $M/piloto_so_sigma_lr$LR/relatorio.json ] || python3 -u $T --variant so_sigma --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_sigma_lr$LR > $L/piloto_so_sigma_lr$LR.log 2>&1
done
for LR in 1e-4 1e-3 1e-2; do
  [ -f $M/piloto_so_rotacao_bloco3_lr$LR/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param exp --oft-block 3 --restricted-lr $LR --seed 0 --validation 0.1 --output $M/piloto_so_rotacao_bloco3_lr$LR > $L/piloto_so_rotacao_bloco3_lr$LR.log 2>&1
done
# endregion

# region 4 - Choose the steps by validation -> etapa8/passos_escolhidos.json
# what? - Best step of each restricted variant for this model.
# how?  - choose_step_etapa8.py reads the pilots of this stage.
python3 $E8/scripts/choose_step_etapa8.py > $L/escolher_passo.log 2>&1
LR_SIG=$(python3 -c "import json;print(json.load(open('$E8/passos_escolhidos.json'))['so_sigma']['lr'])")
LR_ROT=$(python3 -c "import json;print(json.load(open('$E8/passos_escolhidos.json'))['so_rotacao_bloco3']['lr'])")
# endregion

# region 5 - 10 seeds per variant (test set), with the chosen steps
# what? - 40 trainings: frozen 72, sigma only, small-block rotation and full.
# how?  - A training is done when its relatorio.json exists.
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $M/congelado_72_s$S/relatorio.json ]      || python3 -u $T --variant congelado_72 --seed $S --output $M/congelado_72_s$S > $L/treino_congelado_72_s$S.log 2>&1
  [ -f $M/so_sigma_s$S/relatorio.json ]          || python3 -u $T --variant so_sigma --restricted-lr $LR_SIG --seed $S --output $M/so_sigma_s$S > $L/treino_so_sigma_s$S.log 2>&1
  [ -f $M/so_rotacao_bloco3_s$S/relatorio.json ] || python3 -u $T --variant so_rotacao --rotation-param exp --oft-block 3 --restricted-lr $LR_ROT --seed $S --output $M/so_rotacao_bloco3_s$S > $L/treino_so_rotacao_bloco3_s$S.log 2>&1
  [ -f $M/completo_s$S/relatorio.json ]          || python3 -u $T --variant completo --seed $S --output $M/completo_s$S > $L/treino_completo_s$S.log 2>&1
done
# endregion

# region 6 - GPU cost of the four variants on the Base
# what? - The session 14 measurement (memory, optimizer state, time per step) repeated on the larger model.
# how?  - measure_gpu_cost_etapa8.py writes etapa8/custo_gpu_base.json.
[ -f $E8/custo_gpu_base.json ] || python3 -u $E8/scripts/measure_gpu_cost_etapa8.py > $L/custo_gpu_base.log 2>&1
# endregion

# region 7 - Summary of the stage
# what? - Accuracies, recovery fractions and the restriction checks, next to the Small numbers.
# how?  - summarize_etapa8.py writes etapa8/resumo_etapa8.json.
python3 -u $E8/scripts/summarize_etapa8.py > $L/resumo.log 2>&1
echo "end $(date -Iseconds)" >> $L/execucoes.txt
# endregion
