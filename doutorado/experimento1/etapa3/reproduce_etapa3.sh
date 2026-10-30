#!/usr/bin/env bash
# * what? - Recipe of STAGE 3 of experiment 1: fine-tune on a DISTANT domain (skin lesions) with the stage 2 protocol.
# * why?  - Stages 1-2 used a task the model already solves; stage 3 forces the body to change and adds the corrected ruler.
# * how?  - Idempotent: each step is skipped if its output exists (delete the output to redo it).
#
# Nothing from stages 1 and 2 is changed. Everything this stage writes goes to doutorado/experimento1/etapa3/ and models/etapa3/.
set -euo pipefail
cd "$(dirname "$0")/../../.."                       # repository root
E=doutorado/experimento1; E3=$E/etapa3; L=$E3/logs; D=$E3/dados; M=models/etapa3; S3=$E3/scripts
mkdir -p $L $M
date -Iseconds >> $L/execucoes.txt
PELE_TR=$D/pele/data/train-00000-of-00001-2b84ce2361bc8c42.parquet
PELE_TE=$D/pele/data/test-00000-of-00001-3a9b253c406a4af7.parquet
DTD_TR=$D/dtd/data/train-00000-of-00001.parquet;        DTD_TE=$D/dtd/data/test-00000-of-00001.parquet
KM_TR=$D/kmnist/kmnist/train-00000-of-00001.parquet;    KM_TE=$D/kmnist/kmnist/test-00000-of-00001.parquet

# region 1 - Environment and critical check of stages 1-2 (read-only)
# what? - Records the environment and verifies the conclusions of stages 1-2.
# how?  - The verification is skipped when verificacao_etapas12.json already exists.
python3 $E/common/record_environment.py --output $E3/ambiente.json > $L/ambiente.log 2>&1
[ -f $E3/verificacao_etapas12.json ] || python3 $S3/verify_stage12_conclusions.py > $L/verificacao_etapas12.log 2>&1
# endregion

# region 2 - Candidate datasets and screening with the frozen body
# what? - Downloads the candidates (needs network) and screens them with a logistic regression.
# how?  - Both steps are skipped when their output already exists.
[ -f $D/candidatos_inspecao.json ] || python3 $S3/download_candidates.py > $L/baixar_candidatos.log 2>&1
[ -f $E3/triagem.json ] || python3 -u $S3/domain_screening.py > $L/triagem.log 2>&1
# endregion

# region 3 - Fine-tuning pilots on the three distant candidates
# what? - One seed on each candidate.
# how?  - A pilot is done when its relatorio.json exists.
[ -f $M/piloto_pele/relatorio.json ]   || python3 -u $S3/train_etapa3.py --dataset pele   --train $PELE_TR --test $PELE_TE --seed 0 --output $M/piloto_pele   > $L/piloto_pele.log 2>&1
[ -f $M/piloto_dtd/relatorio.json ]    || python3 -u $S3/train_etapa3.py --dataset dtd    --train $DTD_TR  --test $DTD_TE  --seed 0 --output $M/piloto_dtd    > $L/piloto_dtd.log 2>&1
[ -f $M/piloto_kmnist/relatorio.json ] || python3 -u $S3/train_etapa3.py --dataset kmnist --train $KM_TR   --test $KM_TE   --seed 0 --output $M/piloto_kmnist > $L/piloto_kmnist.log 2>&1
# endregion

# region 4 - Chosen domain: skin lesions, 10 real seeds and 10 shuffled-label seeds (weight_decay = 0)
# what? - The 20 trainings of the stage.
# how?  - The shuffled-label control uses the same permutation in every seed.
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $M/s$S/relatorio.json ] || python3 -u $S3/train_etapa3.py --dataset pele --train $PELE_TR --test $PELE_TE --seed $S --output $M/s$S > $L/treino_s$S.log 2>&1
done
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $M/embaralhado_s$S/relatorio.json ] || python3 -u $S3/train_etapa3.py --dataset pele --train $PELE_TR --test $PELE_TE --seed $S --shuffle-labels --output $M/embaralhado_s$S > $L/treino_embaralhado_s$S.log 2>&1
done
# endregion

# region 5 - Compare original -> training with the stage 1-2 compare.py
# what? - One comparison JSON per seed.
# how?  - The script is called as it is, for continuity with the old measures.
for S in 0 1 2 3 4 5 6 7 8 9; do
  [ -f $E3/resultado_s$S.json ] || python3 $E/common/compare.py --before models/dinov2-small/model.safetensors --after $M/s$S/model.safetensors --output $E3/resultado_s$S.json > $L/comparar_s$S.log 2>&1
done
# endregion

# region 6 - Corrected ruler, functional test and shuffled control
# what? - The ruler on four model sets, the transplant test on two and the shuffled-label control.
# how?  - Each step writes one JSON and is skipped when it exists.
for CJ in etapa1 etapa2 etapa3 embaralhado; do [ -f $E3/regua_$CJ.json ] || python3 -u $S3/measure_ruler.py --set $CJ > $L/regua_$CJ.log 2>&1; done
for CJ in etapa2 etapa3; do [ -f $E3/transplante_$CJ.json ] || python3 -u $S3/transplant.py --set $CJ > $L/transplante_$CJ.log 2>&1; done
[ -f $E3/controle_embaralhado.json ] || python3 -u $S3/shuffled_control.py > $L/controle_embaralhado.log 2>&1
# endregion

# region 7 - Random-slice control for the transplant (added later, session 12 of the diary)
# what? - Puts back a random slice of the change with the dimension of the sigma part and measures accuracy.
# how?  - Skipped when its JSON exists; reuses the loader and accuracy function of transplant.py.
[ -f $E3/controle_fatia_aleatoria.json ] || python3 -u $S3/random_slice_control.py > $L/controle_fatia_aleatoria.log 2>&1
# endregion

# region 8 - The page
# what? - Inserts stage 3 into resultados.html from the copy in etapa3/base and the JSONs of this stage.
# how?  - The builder regenerates the whole page, so it is safe to run again.
if [ -f $S3/build_results_etapa3.py ]; then python3 $S3/build_results_etapa3.py > $L/construir.log 2>&1; else echo "build_results_etapa3.py does not exist yet" >> $L/execucoes.txt; fi
echo "end $(date -Iseconds)" >> $L/execucoes.txt
# endregion
