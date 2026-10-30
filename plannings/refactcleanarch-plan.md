# Refatoração do repositório para contar a história da pesquisa (refactcleanarch)

## 1. Problema

O repositório precisa contar a história da pesquisa: cada etapa com seus scripts e seus dados, e uma única página, `artigo/index.html`, que liga cada etapa aos arquivos usados. Hoje o código e os dados da etapa 1 estão soltos na raiz de `doutorado/experimento1/`, e há scripts que ninguém mais executa. Os scripts restantes precisam seguir o padrão da skill `/commentator` e o código inteiro precisa estar em inglês.

Requisitos do pedido:

1. Cada etapa tem seu código e seus dados.
2. A rastreabilidade de cada etapa aparece em `artigo/index.html`: scripts usados e dados, por etapa.
3. Todo script que não é mais usado é removido.
4. Todo script que sobra segue o padrão do `/commentator`.
5. Todo código está em inglês.
6. As etapas ficam claras no `index.html`, com os arquivos de script e de dados de cada seção.

**Estado atual:**

- Existem 33 arquivos `.py`, 5 `.sh` e 1 `.js` fora de `models/` e `data/`.
- `doutorado/experimento1/` mistura 13 scripts de uso compartilhado (`treinar.py`, `comparar.py`, `etapas.py` e outros), as páginas HTML, `resultado*.json`, 44 arquivos `ruido_*.json`, `reprodutibilidade.json` e `resultado_10sementes.json`, todos da etapa 1.
- `doutorado/experimento1/etapa1/ferramentas/` guarda 3 scripts e 5 templates que geraram páginas antigas. `gerar_mapa.py` grava em um diretório temporário que já não existe. `treinar_ANTES_da_etapa2.py` e `construir_diario_ANTES_do_resultados.py` foram substituídos por `treinar.py` e `construir_diario.py`. Nenhum script de `reproduzir_etapaN.sh` os chama.
- `mapa-de-calor.html` e `mapa-de-calor-etapa1.html` têm o mesmo `md5`. Só o segundo é citado por `diario.html` e `resultados.html`.
- Os scripts misturam português e inglês, usam flags em português (`--semente`, `--saida`, `--epocas-aquecer`), e vários são código solto no nível do módulo (`decomposicao_pesos.py`, `cosseno_matriz_inteira.py`, `verificar_repeticao.py`). A skill `/commentator` só cobre corpo de função.
- `reproduzir_etapa1.sh` e `reproduzir_etapa2.sh` não chamam `cosseno_matriz_inteira.py`, embora `etapa1/cosseno_matriz.json` e `etapa2/cosseno_matriz.json` existam e `construir_diario.py` os leia. As etapas 3 e 4 têm scripts `reproduzir_etapaN.sh` idempotentes que chamam `registrar_ambiente.py` e `comparar.py` pelo caminho da raiz de `experimento1/`.
- `registrar_ambiente.py` grava `sha256_scripts` usando `E.glob("*.py")` na raiz de `experimento1/`. Depois da mudança essa lista ficaria vazia.
- O treino não é bit-determinístico: `etapa2/repeticao.json` mostra 1 tensor idêntico de 223 entre duas execuções da mesma semente, mas a mesma acurácia final (94,17).
- Os arquivos `doutorado/`, `artigo/`, `learning/`, `wisdom/` e `.gitignore` não estão versionados (`git status`). Remover algo ali é irreversível sem cópia.

**Critério de aceite:**

1. `python3 artigo/check_repo.py` termina com código 0. Ele verifica: (a) todo `.py`, `.sh` e `.js` de `doutorado/` e `artigo/` aparece em `artigo/index.html`, na seção da sua etapa, e todo caminho citado nessa seção existe; (b) toda `def` e `class` tem docstring de 3 linhas `what?/why?/how?` e todo corpo de função está coberto por `#region N`; (c) comentários, docstrings e identificadores não têm letra acentuada nem palavra portuguesa da lista de controle.
2. `python3 -m pyflakes` sem avisos em todos os `.py`.
3. Todo dado congelado (JSON, CSV, logs, páginas) existe no novo caminho com `sha256` igual ao do manifesto de antes. As únicas exceções são os arquivos que um script regera de propósito (`resultados.html` e os `ambiente.json` regerados no teste).
4. `diario.html` regerado por `build_diary.py` numa cópia de teste é byte a byte igual ao atual. `resultados.html` regerado por `build_results_etapa3.py` e `build_results_etapa4.py` difere do atual só pelo aviso novo no topo.
5. `reproduce_etapa3.sh` e `reproduce_etapa4.sh`, executados na cópia de teste com `models/` ligado por link simbólico, terminam sem erro e sem retreinar.
6. Um treino curto de cada script de treino roda até o fim e acerta no primeiro ciclo dentro de 1 ponto do valor registrado em `logs/`.
7. A raiz do repositório contém só `README.md`, `artigo/`, `doutorado/`, `learning/`, `plannings/`, `wisdom/`, `models/`, `data/` e `.gitignore`.

## 2. Planejamento

### Abordagem

Montar o novo layout escrevendo cada script novo ao lado do antigo, verificar, e só no fim apagar os antigos. Antes de começar, gerar uma cópia compacta (tarball) do código e das páginas fora do repositório e um manifesto de `sha256` dos dados.

Premissas que o pedido deixa em aberto, decididas aqui:

- **Idioma:** são em inglês identificadores, comentários, docstrings, flags de linha de comando, nomes de arquivo de script e mensagens de console. Ficam em português o texto que o leitor vê nas páginas HTML e nas figuras, as chaves dos JSON já gerados, os nomes das pastas de dados e de modelos (`models/etapa4/so_sigma_s0`), os nomes de variante (`so_sigma`, `so_rotacao`, `congelado_72`) e os nomes das pastas `etapaN`, `dados`, `logs`, `paginas`, `base`. Mudar esses nomes alteraria dados congelados. O código os cita como strings.
- **Código solto vira função:** a skill exige regiões dentro de função. Todo script ganha `main()` e `if __name__ == "__main__"`, sem mudar o que o código faz.
- **Shell e JavaScript:** a skill é só de Python. Os `.sh` e o `.js` recebem um cabeçalho `what?/why?/how?` em inglês e comentários `# region N` equivalentes.
- **Páginas HTML não são scripts e não são removidas.** Páginas de narrativa (`diario.html`, `resultados.html`, `results.html`, `usecase.html`, `concordancia-sementes.html`, `mapa-de-calor-detalhado.html`, `mapa-de-calor-etapa1.html`, `etapa1/paginas/*`, `etapa3/base/*`, `etapa4/base/*`) ficam onde estão, porque se linkam por caminho relativo e as cópias em `base/` são congeladas. Só se remove `mapa-de-calor.html`, duplicata exata. O texto delas conserva os nomes antigos de arquivo, e `artigo/index.html` traz o mapa antigo para novo.
- **Layout por etapa:** `scripts/` guarda o código da etapa, a raiz da etapa guarda os JSON e o `reproduce_etapaN.sh`, `comparacoes/` guarda as saídas de `compare.py`, e `logs/`, `paginas/`, `base/` e `dados/` seguem como estão. O código usado por mais de uma etapa vai para `doutorado/experimento1/common/`. O aquecimento da raiz (`svd_imagem.py`) vira `doutorado/experimento0/`.
- **Mantidos como estão:** `models/`, `data/`, `learning/`, `wisdom/` (gerenciada pelas skills `arch-*`), `doutorado/referencias/`, `doutorado/transfer-learning-class.html`.
- **Renomeio de `oxfordpet_explorartion.py`:** passa a `oxfordpet_exploration.py`, corrigindo o erro de digitação.

### Arquivos novos

| Arquivo | Conteúdo |
| --- | --- |
| `doutorado/experimento1/common/stages.py` | Caminhos e funções `model()`, `model_dir()`, `comparison()` das etapas 1 e 2. Origem: `etapas.py`. |
| `doutorado/experimento1/common/train.py` | Fine-tuning do DINOv2-Small em 37 raças. Origem: `treinar.py`. Flags `--seed`, `--output`, `--weight-decay`, `--body-lr`, `--epochs`, `--warmup-epochs`. |
| `doutorado/experimento1/common/compare.py` | SVD das 72 matrizes antes e depois e medidas de mudança. Origem: `comparar.py`. Flags `--before`, `--after`, `--output`. |
| `doutorado/experimento1/common/record_environment.py` | Grava versões, máquina, git e hashes. Origem: `registrar_ambiente.py`. `sha256_scripts` passa a varrer `common/` e `etapa*/scripts/`. |
| `doutorado/experimento1/common/weight_decomposition.py` | Decomposição da mudança dos pesos. Origem: `decomposicao_pesos.py`. |
| `doutorado/experimento1/common/reproducibility_weights_directions.py` | Concordância entre sementes em pesos e direções. Origem: `reprodutibilidade_pesos_direcoes.py`. |
| `doutorado/experimento1/common/whole_matrix_cosine.py` | Cosseno entre mudanças da matriz inteira. Origem: `cosseno_matriz_inteira.py`. |
| `doutorado/experimento1/common/aggregate_stage.py` | Agrega resultados de uma etapa em `agregado.json`. Origem: `agregar_etapa.py`. |
| `doutorado/experimento1/etapa1/scripts/download_model.py` | Baixa o DINOv2-Small. Origem: `baixar_modelo.py`. |
| `doutorado/experimento1/etapa1/scripts/diagnose_frozen_body.py` | Acurácia do corpo congelado. Origem: `diagnostico_corpo_congelado.py`. |
| `doutorado/experimento1/etapa1/scripts/analyze_seeds.py` | Análise de 10 sementes e 45 pares de ruído. Origem: `analisar_sementes.py`. |
| `doutorado/experimento1/etapa1/scripts/oxfordpet_exploration.py` | Exploração do dataset de pets. Origem: `etapa1/oxfordpet_explorartion.py`. |
| `doutorado/experimento1/etapa1/reproduce_etapa1.sh` | Receita da etapa 1 com as flags novas. Passa a chamar `whole_matrix_cosine.py`. Origem: `reproduzir_etapa1.sh`. |
| `doutorado/experimento1/etapa1/complements_etapa1.sh` | Complementos da etapa 1. Origem: `complementos_etapa1.sh`. |
| `doutorado/experimento1/etapa2/scripts/verify_repeat.py` | Mesma semente duas vezes dá o mesmo modelo? Origem: `verificar_repeticao.py`. |
| `doutorado/experimento1/etapa2/scripts/build_diary.py` | Gera `diario.html` e `resultados.html` das etapas 1 e 2. Origem: `construir_diario.py`. |
| `doutorado/experimento1/etapa2/scripts/diary_template.html` | Mesmo conteúdo de `diario_template.html`, movido. |
| `doutorado/experimento1/etapa2/scripts/explorer_template.js` | `explorador_template.js` com comentários em inglês e cabeçalho. |
| `doutorado/experimento1/etapa2/reproduce_etapa2.sh` | Receita da etapa 2 com flags novas. Passa a chamar `whole_matrix_cosine.py` e `build_diary.py`. Origem: `reproduzir_etapa2.sh`. |
| `doutorado/experimento1/etapa3/scripts/config_etapa3.py` | Constantes e caminhos da etapa 3. |
| `doutorado/experimento1/etapa3/scripts/train_etapa3.py` | Treino em domínio novo. Origem: `treinar_etapa3.py`. |
| `doutorado/experimento1/etapa3/scripts/download_candidates.py` | Baixa os datasets candidatos. Origem: `baixar_candidatos.py`. |
| `doutorado/experimento1/etapa3/scripts/domain_screening.py` | Triagem de domínio com corpo congelado. Origem: `triagem_dominio.py`. |
| `doutorado/experimento1/etapa3/scripts/verify_stage12_conclusions.py` | Verificação crítica das etapas 1 e 2. Origem: `verificar_conclusoes_etapas12.py`. |
| `doutorado/experimento1/etapa3/scripts/measure_ruler.py` | Régua corrigida. Origem: `medir_regua.py`. |
| `doutorado/experimento1/etapa3/scripts/transplant.py` | Teste funcional por transplante. Origem: `transplante.py`. |
| `doutorado/experimento1/etapa3/scripts/shuffled_control.py` | Controle de rótulos embaralhados. Origem: `controle_embaralhado.py`. |
| `doutorado/experimento1/etapa3/scripts/build_results_etapa3.py` | Insere a etapa 3 em `resultados.html`. Origem: `construir_resultados_etapa3.py`. |
| `doutorado/experimento1/etapa3/scripts/etapa3_template.html` | Mesmo conteúdo de `etapa3_template.html`, movido. |
| `doutorado/experimento1/etapa3/reproduce_etapa3.sh` | Receita idempotente da etapa 3 com caminhos e flags novos. |
| `doutorado/experimento1/etapa4/scripts/config_etapa4.py` | Constantes da etapa 4. |
| `doutorado/experimento1/etapa4/scripts/train_etapa4.py` | Fine-tuning restrito. Origem: `treinar_etapa4.py`. Flags em inglês (`--variant`, `--restricted-lr`, `--validation`, `--oft-block`, `--rotation-param`). |
| `doutorado/experimento1/etapa4/scripts/choose_step.py` | Escolha do passo por validação. Origem: `escolher_passo.py`. |
| `doutorado/experimento1/etapa4/scripts/measure_variants.py` | Medidas das variantes. Origem: `medir_variantes.py`. |
| `doutorado/experimento1/etapa4/scripts/build_results_etapa4.py` | Insere a etapa 4 em `resultados.html` e grava o aviso de renomeação no topo. Origem: `construir_resultados_etapa4.py`. |
| `doutorado/experimento1/etapa4/scripts/etapa4_template.html` | Mesmo conteúdo de `etapa4_template.html`, movido. |
| `doutorado/experimento1/etapa4/reproduce_etapa4.sh` | Receita idempotente da etapa 4 com caminhos e flags novos. |
| `doutorado/experimento1/etapa4/base/resultados_ate_etapa4_antes_refatoracao.html` | Cópia do `resultados.html` de antes da refatoração. |
| `doutorado/experimento0/svd_image.py` | Aquecimento: SVD de uma imagem. Origem: `svd_imagem.py`. Lê `input.png` e grava em `output/`. |
| `doutorado/experimento0/README.md` | Texto atual de `README.md` da raiz, com caminhos corrigidos. |
| `artigo/generate_figures.py` | Figuras 1 a 14 do artigo. Origem: `artigo/gerar_imagens.py`. |
| `artigo/check_repo.py` | Verificador do critério de aceite 1 (rastreabilidade, padrão `/commentator`, idioma). |
| `artigo/versoes/index_v2_antes_da_refatoracao.html` | Cópia do `artigo/index.html` de antes desta mudança. |
| `plannings/refactcleanarch-baseline.sha256` | Manifesto `sha256` dos dados congelados antes da mudança. |

### Arquivos alterados

| Arquivo | Alteração |
| --- | --- |
| `artigo/index.html` | Nova seção "Rastreabilidade por etapa" no fim, com: árvore do repositório; um quadro por etapa 0 a 4 com pergunta, o que mudou, scripts (caminho e papel), dados e modelos (caminho), logs, páginas, comando de reprodução; tabela de nomes antigos para novos; tabela dos scripts removidos e do motivo. As seções 2, 5 a 9 e 10 ganham uma caixa "Arquivos desta seção" com scripts e dados. Atualiza o sumário e o rodapé. |
| `README.md` | Reescrito como mapa curto: a história em 5 etapas, a estrutura de pastas e o link para `artigo/index.html`. |
| `.gitignore` | Acrescenta `__pycache__/`. |
| `doutorado/experimento1/resultado.json`, `resultado_s1.json` a `resultado_s9.json`, `resultado_ruido.json`, `ruido_*.json` | Movidos para `doutorado/experimento1/etapa1/comparacoes/`. Conteúdo idêntico. |
| `doutorado/experimento1/reprodutibilidade.json`, `resultado_10sementes.json`, `resultado_5sementes.json` | Movidos para `doutorado/experimento1/etapa1/`. Conteúdo idêntico. `resultado_5sementes.json` é saída de uma versão antiga de `analisar_sementes.py`, mantida como dado. |
| `doutorado/experimento1/etapa2/resultado_s0.json` a `resultado_s9.json` | Movidos para `doutorado/experimento1/etapa2/comparacoes/`. |
| `doutorado/experimento1/etapa3/resultado_s0.json` a `resultado_s9.json` | Movidos para `doutorado/experimento1/etapa3/comparacoes/`. |
| `doutorado/experimento1/achados_etapa2.html` | Movido para `doutorado/experimento1/etapa2/`. Entrada de `build_diary.py`. |
| `doutorado/experimento1/resultados.html` | Regerado por `build_results_etapa4.py` com o aviso de renomeação no topo. É a única alteração de conteúdo. |

### Arquivos removidos

| Arquivo | Motivo |
| --- | --- |
| `svd_imagem.py`, `entrada.png`, `saida/`, `README.md` (da raiz) | Movidos para `doutorado/experimento0/` (`input.png`, `output/`). |
| `doutorado/experimento1/{etapas,treinar,comparar,registrar_ambiente,decomposicao_pesos,reprodutibilidade_pesos_direcoes,cosseno_matriz_inteira,agregar_etapa}.py` | Substituídos pelos arquivos de `common/`. |
| `doutorado/experimento1/{baixar_modelo,diagnostico_corpo_congelado,analisar_sementes,verificar_repeticao,construir_diario}.py` e `etapa1/oxfordpet_explorartion.py` | Substituídos pelos arquivos de `etapa1/scripts/` e `etapa2/scripts/`. |
| `doutorado/experimento1/{reproduzir_etapa1,reproduzir_etapa2,complementos_etapa1}.sh`, `etapa3/reproduzir_etapa3.sh`, `etapa4/reproduzir_etapa4.sh` | Substituídos pelos `reproduce_etapaN.sh` e `complements_etapa1.sh`. |
| `doutorado/experimento1/{diario_template.html,explorador_template.js}`, `etapa3/etapa3_template.html`, `etapa4/etapa4_template.html` | Movidos para `scripts/` da etapa dona. |
| `doutorado/experimento1/etapa3/{config_etapa3,treinar_etapa3,baixar_candidatos,triagem_dominio,verificar_conclusoes_etapas12,medir_regua,transplante,controle_embaralhado,construir_resultados_etapa3}.py` e `etapa4/{config_etapa4,treinar_etapa4,escolher_passo,medir_variantes,construir_resultados_etapa4}.py` | Substituídos pelos arquivos de `scripts/`. |
| `doutorado/experimento1/etapa1/ferramentas/gerar_mapa.py` | Grava em diretório temporário inexistente. Nenhuma receita o chama. |
| `doutorado/experimento1/etapa1/ferramentas/construir_diario_ANTES_do_resultados.py` | Substituído por `build_diary.py`. Ninguém o chama. |
| `doutorado/experimento1/etapa1/ferramentas/treinar_ANTES_da_etapa2.py` | Substituído por `common/train.py`. A etapa 1 usa os valores padrão de `train.py`. |
| `doutorado/experimento1/etapa1/ferramentas/{conc_template,diario_template_ANTES_do_resultados,mapa_template,mapa2_template,mapa3_template}.html` | Templates sem script que os use. As páginas geradas permanecem em `paginas/` e na raiz de `experimento1/`. |
| `doutorado/experimento1/mapa-de-calor.html` | Duplicata byte a byte de `mapa-de-calor-etapa1.html`. Nenhuma página a cita. |
| Todos os `__pycache__/` | Gerados pelo Python. |

Os arquivos substituídos só são apagados no passo 11, depois de todos os testes passarem.

### Banco de dados

Nenhuma alteração de schema.

### Recursos de cloud

Nenhum recurso novo. Os testes de treino usam a GPU local e os dados já baixados. Nenhum script baixa dado nesta mudança.

### Passos

1. Gerar o tarball `/home/afd/gitallan/poc-svd-antes-refatoracao.tar.gz` com tudo, menos `.git/`, `models/`, `data/` e `doutorado/experimento1/etapa3/dados/`. Gerar `plannings/refactcleanarch-baseline.sha256` com o `sha256` de todo arquivo que não é script nem template (JSON, CSV, `.log`, `.txt`, `.html`).
2. Copiar `artigo/index.html` para `artigo/versoes/index_v2_antes_da_refatoracao.html` e `doutorado/experimento1/resultados.html` para `etapa4/base/resultados_ate_etapa4_antes_refatoracao.html`.
3. Mover os dados com `mv` conforme a tabela de alterados, criando `comparacoes/` em cada etapa. Conferir o `sha256` de cada arquivo movido contra o manifesto.
4. Escrever `common/` na ordem `stages.py`, `train.py`, `compare.py`, `record_environment.py`, `weight_decomposition.py`, `reproducibility_weights_directions.py`, `whole_matrix_cosine.py`, `aggregate_stage.py`. Em cada arquivo: traduzir identificadores, comentários, docstrings, flags e mensagens; trocar caminhos de dados pelos novos; envolver o código solto em funções; aplicar docstring de 3 linhas e `#region`. O comportamento de cada função não muda.
5. Escrever `etapa1/scripts/`, `etapa2/scripts/`, `etapa3/scripts/` e `etapa4/scripts/` da mesma forma. As etapas 3 e 4 trocam `sys.path.insert(0, ...)` para apontar para `common/` e `scripts/`. `build_results_etapa4.py` acrescenta ao `resultados.html` um aviso no topo: os nomes de arquivo citados nas etapas são os da época, e o mapa atual está em `artigo/index.html#rastreabilidade`.
6. Escrever os cinco `.sh` e o `.js` com o cabeçalho e os comandos novos. Acrescentar `whole_matrix_cosine.py` às receitas das etapas 1 e 2 e `build_diary.py` à da etapa 2.
7. Mover `svd_imagem.py`, `entrada.png`, `saida/` e `README.md` para `doutorado/experimento0/` e traduzir `svd_image.py`. Apontar `artigo/generate_figures.py` para `doutorado/experimento0/input.png`.
8. Escrever `artigo/check_repo.py` e a seção de rastreabilidade em `artigo/index.html`, incluindo as caixas "Arquivos desta seção".
9. Reescrever `README.md` e `.gitignore`.
10. Rodar os testes abaixo. Corrigir e repetir até todos passarem.
11. Apagar os arquivos antigos da tabela de removidos, e os `__pycache__/`. Rodar `check_repo.py` e o teste de manifesto de novo.

### Testes

Todos rodam da raiz do repositório. Os testes que escrevem arquivos usam uma cópia de teste em `/tmp/claude-1000/-home-afd-gitallan-poc-svd/d0f3021b-e44f-4a1b-8f89-9da3a69e6396/scratchpad/copia/`, com `models/` e `data/` ligados por link simbólico e sem as saídas de treino, para não tocar nos dados reais.

| Teste | Como | Passa quando |
| --- | --- | --- |
| Estático | `python3 -m pyflakes $(find doutorado artigo -name "*.py")`; `bash -n` em cada `.sh`; `python3 -m py_compile` em cada `.py` | Sem saída de erro. |
| Flags | Para cada `--flag` usado nos `.sh`, conferir que aparece em `python3 <script> --help` | Todas existem. |
| Padrão e idioma | `python3 artigo/check_repo.py` | Código de saída 0. |
| Rastreabilidade | O mesmo `check_repo.py`, parte (a) | Código de saída 0. |
| Dados congelados | Reconferir `sha256` contra `plannings/refactcleanarch-baseline.sha256`, ignorando só `resultados.html` | Nenhuma diferença. |
| Páginas | Na cópia de teste: `build_diary.py` e `cmp` de `diario.html`; `build_results_etapa3.py` e `build_results_etapa4.py` e `diff` de `resultados.html` | `diario.html` idêntico; `resultados.html` difere só do aviso. |
| Receitas 3 e 4 | Na cópia de teste: `bash doutorado/experimento1/etapa3/reproduce_etapa3.sh` e `.../etapa4/reproduce_etapa4.sh` | Terminam com código 0, sem retreino, e os JSON regerados batem com os originais (tolerância relativa 1e-4). |
| Análises da etapa 1 e 2 | Na cópia de teste: `compare.py` (semente 0, etapas 1 e 2), `analyze_seeds.py`, `weight_decomposition.py`, `aggregate_stage.py`, `whole_matrix_cosine.py` | JSON regerados batem com os originais (tolerância relativa 1e-4). |
| Treino curto | `train.py --seed 0 --weight-decay 0` (protocolo da etapa 2), `train_etapa3.py` (pele, semente 0), `train_etapa4.py` com cada uma das três variantes, todos com `--output` na cópia de teste | Terminam e a acurácia do primeiro ciclo fica a 1 ponto do valor em `logs/` ou em `relatorio.json` da mesma semente. |

Não se testa `download_model.py` nem `download_candidates.py` por execução, porque baixam dados. Eles passam por `pyflakes`, `--help` e pelo `check_repo.py`.
