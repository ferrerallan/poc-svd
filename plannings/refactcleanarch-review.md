# Revisão do plano refactcleanarch

Verificado contra o código: contagens de arquivos (33 `.py`, 5 `.sh`, 44 `ruido_*.json`), os três construtores de página, as receitas `reproduzir_etapaN.sh`, `registrar_ambiente.py`, `config_etapa3.py`, a duração dos treinos e os textos de `diario.html` e `resultados.html`.

Resultado dos testes de premissa, rodados numa cópia de `doutorado/`:

- `construir_diario.py` regera `diario.html` e `resultados.html` das etapas 1 e 2 byte a byte iguais às cópias congeladas em `etapa3/base/`.
- `construir_resultados_etapa3.py` regera uma página byte a byte igual a `etapa4/base/resultados_ate_etapa3.html`.
- `construir_resultados_etapa4.py` regera um `resultados.html` byte a byte igual ao atual.
- Um treino completo de uma semente leva de 1,2 a 1,5 minuto (logs de `etapa2/`, `etapa3/` e `etapa4/`).

## Veredito

Aprovado com ajustes. O desenho está certo, mas o plano remove um arquivo que as páginas prometem manter, tem um critério de aceite que se contradiz, deixa metade dos scripts de medida sem teste e altera uma página sem que o pedido exija.

## Requisitos

| Requisito | Onde está no plano | Status |
| --- | --- | --- |
| 1. Cada etapa tem seu código e seus dados | Abordagem (layout por etapa), tabelas de novos e alterados, passos 3 a 5 | Atendido |
| 2. Rastreabilidade em `artigo/index.html`, por etapa | Tabela de alterados (`artigo/index.html`), passo 8, critério 1(a) | Parcial: o critério 1(a) exige que todo caminho citado exista, mas a seção cita nomes antigos e scripts removidos; a receita da etapa 1 não chama `oxfordpet_exploration.py`, então o script não aparece em nenhuma receita |
| 3. Remover script não usado | Tabela de removidos (`ferramentas/`) | Atendido. Os três scripts de `ferramentas/` e os cinco templates são os únicos órfãos. Os outros 30 são chamados por alguma receita ou lidos por um construtor |
| 4. Padrão `/commentator` nos scripts que sobram | Passos 4 a 6, critério 1(b) | Parcial: `check_repo.py` verifica só `.py`; os cinco `.sh` e o `.js` não têm verificação |
| 5. Todo código em inglês | Premissa "Idioma", passos 4 a 7, critério 1(c) | Atendido, com a premissa de que chaves de JSON, nomes de variante e nomes de pasta de dados ficam em português (ver Perguntas) |
| 6. Etapas, scripts e dados claros por seção do `index.html` | Tabela de alterados: caixas nas seções 2, 5 a 9 e 10 | Parcial: as seções 3 e 4 citam `models/dinov2-small`, `download_model.py` e a figura 4 e ficaram sem caixa |
| Implícito: nada de dado é perdido | Passos 1 e 3, manifesto `sha256` | Atendido |

## Precisa mudar

**1. Os scripts de medida não são exercitados pelos testes de receita.**
Onde: Testes, linha "Receitas 3 e 4".
Por quê: `reproduce_etapa3.sh` e `reproduce_etapa4.sh` pulam cada passo cuja saída existe (`[ -f ... ] ||`). Na cópia de teste todas as saídas existem, então `measure_ruler.py`, `transplant.py`, `shuffled_control.py`, `domain_screening.py`, `verify_stage12_conclusions.py`, `choose_step.py` e `measure_variants.py` nunca rodam. Sete scripts traduzidos e reestruturados ficariam sem execução.
Mude para: na cópia de teste, apagar antes de rodar as saídas desses passos (`regua_*.json`, `transplante_*.json`, `controle_embaralhado.json`, `verificacao_etapas12.json`, `triagem.json`, `passos_escolhidos.json`, `medidas_variantes.json`) e comparar os JSON regerados com os originais. Manter `candidatos_inspecao.json` e os modelos, para não baixar nem treinar.

**2. O teste de treino está especificado errado.**
Onde: Testes, linha "Treino curto", e critério de aceite 6.
Por quê: "primeiro ciclo" não exercita o ciclo inteiro (corpo solto, otimizador, salvamento). Um treino completo leva 1,2 a 1,5 minuto, então não há motivo para encurtar. O treino não é bit-determinístico (`etapa2/repeticao.json`), mas a acurácia da época 1 foi igual nas duas execuções (91,58) e nas três variantes da mesma semente (51,69).
Mude para: rodar o treino completo da semente 0 de cada script: `train.py` (protocolo da etapa 1 e da etapa 2), `train_etapa3.py` (pele), `train_etapa4.py` (as três variantes). Comparar com `acuracias_por_epoca` do `relatorio.json` da mesma semente: época 1 a até 0,5 ponto, época 5 a até 1,5 ponto. A tolerância vem da diferença de 0,4 ponto entre as duas execuções da etapa 2.

**3. O critério 1(a) se contradiz com a tabela de nomes antigos.**
Onde: Critério de aceite 1(a) e passo 8.
Por quê: a seção nova cita os caminhos antigos e os scripts removidos, que por definição não existem. `check_repo.py` falharia no primeiro dia.
Mude para: marcar no HTML os caminhos atuais com `data-trace` e os antigos com `data-legacy`. O verificador confere a existência só dos `data-trace` e exige que todo script do repositório apareça em algum `data-trace`.

**4. `mapa-de-calor.html` não pode ser removido.**
Onde: Tabela de removidos e Estado atual ("Nenhuma página a cita").
Por quê: é falso. `diario.html` e `resultados.html` dizem que `mapa-de-calor-etapa1.html` é "igual ao `mapa-de-calor.html`, que continua intacto", e a regra do projeto é não perder nenhuma página. Apagar o arquivo desmente a página e quebra a regra.
Mude para: manter `mapa-de-calor.html`. Tirar a linha da tabela de removidos e a afirmação do Estado atual.

**5. Regerar `resultados.html` com aviso novo não é requisito.**
Onde: Tabela de alterados (`resultados.html`), tabela de novos (`resultados_ate_etapa4_antes_refatoracao.html`), passos 2 e 5, critério 4.
Por quê: nenhum requisito pede alterar uma página de narrativa, e é a única alteração de conteúdo do plano. O mapa de nomes antigos e novos já fica em `artigo/index.html`. O teste de byte a byte é mais forte sem o aviso.
Mude para: não regerar `resultados.html` no repositório real, não criar a cópia `..._antes_refatoracao.html` e não editar `build_results_etapa4.py` além da tradução. O critério passa a exigir que, na cópia de teste, as três páginas regeradas sejam byte a byte iguais às atuais.

**6. `comparacoes/` nas etapas 2 e 3 é reorganização sem requisito.**
Onde: Tabela de alterados (linhas de `etapa2/` e `etapa3/`), Abordagem ("Layout por etapa").
Por quê: os `resultado_s*.json` dessas etapas já estão dentro da pasta da etapa. Movê-los obriga a editar caminhos em `stages.py`, `config_etapa3.py`, nos construtores e nas receitas, sem ganho para os requisitos. Na etapa 1 o movimento se justifica: 55 arquivos soltos na raiz de `experimento1/` pertencem à etapa 1.
Mude para: criar `comparacoes/` só em `etapa1/`. Deixar `etapa2/resultado_s*.json` e `etapa3/resultado_s*.json` onde estão.

**7. O aquecimento renomeia dados, contra a própria premissa de idioma.**
Onde: Tabela de novos (`svd_image.py` "lê `input.png` e grava em `output/`"), Tabela de removidos (`entrada.png`, `saida/`).
Por quê: a premissa mantém nomes de pasta e de arquivo de dado em português para não alterar dados congelados. `entrada.png` e `saida/` (28 MB versionados no git) são dados. `artigo/generate_figures.py`, o README antigo e o texto do artigo citam esses nomes.
Mude para: mover `entrada.png` e `saida/` com o nome atual para `doutorado/experimento0/`. `svd_image.py` resolve os dois pelo diretório do próprio script, sem depender de onde é chamado.

**8. As seções 3 e 4 ficam sem lista de arquivos.**
Onde: Tabela de alterados (`artigo/index.html`), passo 8.
Por quê: viola o requisito 6, "inclusive quais arquivos script foram utilizados e também dados para cada sessão". A seção 3 usa `download_model.py` e `models/dinov2-small`; a seção 4 usa a figura 14.
Mude para: caixa "Arquivos desta seção" em todas as seções que citam script, dado ou figura: 2, 3, 4, 5, 6, 7, 8, 9 e 10. O quadro por etapa também lista `oxfordpet_exploration.py` na etapa 1.

**9. A receita da etapa 1 deixa o script de exploração órfão, e as receitas 1 e 2 não têm teste.**
Onde: Tabela de novos (`reproduce_etapa1.sh`), Testes.
Por quê: um script fora de toda receita é o mesmo estado que o requisito 3 manda eliminar. As receitas 1 e 2 retreinam e sobrescrevem `models/`, então não podem ser executadas; hoje só são conferidas por `bash -n` e pela lista de flags, o que não detecta caminho errado.
Mude para: (a) `reproduce_etapa1.sh` chama `oxfordpet_exploration.py` depois do diagnóstico; (b) teste novo: rodar cada receita 1 e 2 com `python3` substituído por uma função que só registra os argumentos, e conferir que todo script citado existe e que cada flag aparece em `--help` do script.

**10. O verificador não cobre `.sh` e `.js`.**
Onde: Critério de aceite 1(b), `artigo/check_repo.py`.
Por quê: o requisito 4 vale para todo script que sobra. A premissa decide aplicar a convenção aos `.sh` e ao `.js`, mas nada confere.
Mude para: `check_repo.py` verifica também que cada `.sh` e o `.js` começam com cabeçalho de três linhas `what?`, `why?`, `how?` e têm comentários `region N` numerados em sequência.

## Sugestões

**Ajuste de caminho em `scripts/`.** `config_etapa3.py`, `config_etapa4.py` e `oxfordpet_explorartion.py` calculam a raiz com `Path(__file__).resolve().parents[3]`. Em `scripts/` passa a ser `parents[4]`. Os testes pegam o erro, mas vale listar no passo 5.

**Tolerância dos JSON regerados.** Subir de 1e-4 para 1e-3 relativo. As análises usam SVD em ponto flutuante e não há registro de que rodaram no mesmo dispositivo que hoje.

**Verificação de idioma mais simples.** Limitar a parte (c) a letras fora do ASCII e a uma lista fixa de palavras portuguesas tiradas dos identificadores antigos. Uma lista genérica de palavras vazias dá falso positivo em comentário técnico.

## Perguntas

1. As chaves dos JSON já gerados (`acuracia_antes`, `bloco`, `tipo`), os nomes de variante (`so_sigma`, `so_rotacao`, `congelado_72`) e as pastas de dados e modelos continuam em português, e o código as cita como strings. Traduzir essas chaves exige reescrever dados congelados e os `relatorio.json` de `models/`, o que o histórico do projeto proíbe. Recomendo manter, como o plano faz. Se você quiser "todo código em inglês" incluindo o esquema de dados, a mudança é outra e precisa de migração dos dados.
2. As páginas de narrativa conservam os nomes antigos de script no texto. Recomendo manter, com o mapa antigo para novo no artigo. Se preferir texto atualizado, é preciso regerar as páginas e abrir mão da regra de cópias congeladas.
