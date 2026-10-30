# Aprendizados do Experimento 1 (etapas 1 a 3)

> O que foi feito tecnicamente, por que, e o que cada etapa mostrou. Linguagem técnica, mas direta.
> Todos os números vêm dos arquivos em `doutorado/experimento1/` (JSONs e `relatorio.json` de cada treino).
> A versão completa, com tabelas e mapas, está em `doutorado/experimento1/resultados.html`.

---

## 0. A pergunta e o desenho geral

**Pergunta da tese.** Quando um foundation model é especializado (fine-tuning) numa tarefa nova, o que muda nas
matrizes de pesos: os **valores singulares** (σ, o "peso" de cada direção que a matriz já usava) ou os
**vetores singulares** (as direções em si)?

**Instrumento.** A SVD de cada matriz de pesos, `W = U Σ Vᵀ`, antes e depois do fine-tuning. Comparar `σ`
responde "mudou o peso"; comparar `U` responde "mudou a direção".

**Objeto.** O corpo do **DINOv2-Small** (Meta, 2023): Vision Transformer com 12 layers, bitola 384, 22,1 M de
parâmetros, pré-treinado sem rótulos em ~142 M de imagens. Cada layer tem 6 matrizes de pesos (`query`, `key`,
`value`, `dense` na atenção; `fc1`, `fc2` no MLP), total de **72 matrizes**. Só elas entram na medição. O modelo vem
sem cabeça de classificação; uma cabeça linear `384 → K` é colada sobre o token CLS e treinada junto.

**Protocolo de fine-tuning (igual em todas as etapas).** 5 épocas, lote 32, AdamW, passo `1e-5` no corpo e
`1e-3` na cabeça, agenda cosseno, label smoothing 0,1, bf16, corpo congelado na 1ª época (aquecimento), **10
sementes** (0 a 9) por configuração. O que muda entre etapas está dito em cada uma.

**Convenção de medição.** Para cada matriz, SVD de `W0` (original) e de `W1` (fine-tuned); compara-se o 1º σ com o
1º, o 2º com o 2º etc., e a 1ª direção com a 1ª etc. (10 principais). Tira-se a média das 10 diferenças (um valor
por treino) e depois a média dos 10 treinos. Isso é o número de cada "quadradinho" dos mapas.

---

## 1. Etapa 1 — primeira rodada completa (`weight_decay = 0,05`)

### O que foi feito

1. **Tarefa de calibração:** Oxford-IIIT Pet, 37 raças de cães e gatos, 3.680 fotos de treino e 3.669 de teste.
   Domínio próximo do pré-treino, escolhido de propósito para errar barato.
2. **Tentativa 0 (descartada):** corpo destravado desde o passo zero, passo `5e-5`, sem aquecimento. Acurácia foi de
   2% para apenas 9,7% em 3 épocas: a cabeça aleatória gera gradientes grandes que degradam o corpo antes de ela
   aprender a ler. Diagnóstico: corpo **congelado** + regressão logística sobre o CLS = **94,8%**. O corpo estava
   intacto; o problema era o método. Correção: aquecimento de 1 época e passo `1e-5`.
3. **10 treinos** com a correção. Acurácia final **94,15% ± 0,25**.
4. **Medidas** (`comparar.py`): deslocamento `‖W1−W0‖/‖W0‖`; variação dos 10 maiores σ em %; giro vetor a vetor das 10
   principais direções (com o conserto do sinal: cosseno em valor absoluto); e uma "concentração" (fração da
   energia da mudança que cai na diagonal da base da SVD original, normalizada pelo acaso).
5. **Piso de ruído:** a mesma medida entre dois treinos de sementes diferentes (45 pares).

### Resultados

| Medida (média das 72 matrizes) | Valor |
|---|---|
| Deslocamento da matriz | 0,52% |
| Variação dos 10 maiores σ | 0,019% |
| Giro das 10 direções principais (vetor a vetor) | 0,95° (≈ 1,66% do comprimento do vetor) |
| Concordância entre treinos (cosseno), mudança inteira | 0,218 |
| Concordância, mudança dos σ / das direções | 0,53 / 0,32 |

### O que foi aprendido

- **Especializar mexe muito pouco:** meio por cento de deslocamento, σ mudando duas centésimas de por cento.
- **Primeira leitura do piso de ruído estava errada.** A diferença entre dois treinos (0,65%) é maior que a mudança
  de cada um (0,52%), e isso foi lido como "a adaptação está abaixo do ruído". Mas a diferença de duas mudanças
  independentes seria 1,41× maior; o observado foi 1,25×, o que corresponde a um cosseno de 0,22 entre as duas
  mudanças: **cerca de um quinto da mudança é compartilhada** entre treinos (sistemática), o resto é específico de
  cada semente.
- **Hipótese levantada ao fim:** a mudança de tamanho dos σ pode ser pouco informativa. Decompondo a mudança dos 10
  maiores σ em uma parte **comum** (todos mudam juntos, escala geral) e uma **diferencial** (um σ contra outro):
  escala geral média de **−0,0092%**, quase sempre negativa, igual ao encolhimento que o **weight decay** do AdamW
  produziria sozinho com este cronograma (**−0,0088%**). Decisão: refazer com `weight_decay = 0`.

---

## 2. Etapa 2 — o mesmo protocolo com `weight_decay = 0`

### O que foi feito

Única mudança: `weight_decay` de 0,05 para 0. Mesmos dados, hiperparâmetros e sementes. Mais uma verificação de
reprodutibilidade: a semente 0 treinada duas vezes.

### Resultados

| Medida | Etapa 1 | Etapa 2 |
|---|---|---|
| Acurácia | 94,15% | 94,13% |
| Variação dos 10 maiores σ | 0,0191% | 0,0173% |
| Escala geral dos σ (média com sinal) | −0,0092% | −0,0002% |
| Parte diferencial dos σ | 0,0166% | 0,0166% |
| Fração da energia dos σ que é escala geral | 24,3% | 12,1% (acaso com 10 σ independentes: 10%) |
| Giro das direções | 0,95° | 0,94° |
| Concordância da mudança dos σ | 0,53 | 0,45 |
| Concordância da mudança das direções | 0,32 | 0,32 |

Repetição da semente 0: só 1 dos 223 tensores idêntico; a diferença entre as duas execuções é ~20% do tamanho da
mudança do fine-tuning (a GPU não é determinística), contra ~125% entre sementes diferentes. Acurácia igual.

### O que foi aprendido

- **A hipótese se confirmou em parte.** A escala geral era efeito do otimizador: sumiu. A parte diferencial (reponderar
  um σ em relação a outro) ficou idêntica, portanto **não é artefato**. As direções não dependem do weight decay.
- A concordância dos σ caiu de 0,53 para 0,45 porque a escala geral, igual em todas as sementes, a inflava.
- O procedimento é reprodutível em parâmetros e resultado, mas não bit a bit.

---

## 3. Revisão crítica antes da etapa 3 (o que estava equivocado)

Verificação feita sobre os próprios modelos e JSONs (`etapa3/verificar_conclusoes_etapas12.py`):

1. **A mudança medida não tinha efeito funcional.** Corpo congelado + regressão logística: **94,79%**; fine-tuned:
   94,13%. O fine-tuning do corpo deu **−0,65 ponto**. Em pets o corpo não precisava mudar; o salto de 2% para 94%
   vinha da cabeça. Medimos uma deriva sem função.
2. **"Direções mudam ~87× mais que pesos" é estrutural.** Ruído gaussiano isotrópico de 0,5% somado às matrizes, sem
   aprendizado, dá razão de **112×** (mediana). Um σ é um escalar; a direção acumula mudança em 383 componentes.
3. **O giro vetor a vetor estava inflado pela degenerescência.** Quando σ vizinhos são quase iguais, a direção
   individual é mal definida e gira sozinha. Medindo o **subespaço** top-10 por ângulos principais: **0,38°**, contra
   0,97° vetor a vetor.
4. Logo, "σ mudam de forma mais consistente que direções" era uma comparação viciada pelo item 3.
5. **Achado subdeclarado:** na etapa 2, o sinal da escala geral é consistente em ≥ 9 dos 10 treinos em **37 de 72**
   matrizes (acaso ≈ 1,5). Existe uma reescala por matriz, sistemática, independente do weight decay.
6. **Menores:** teste pareado (mesmas sementes) em vez de Welch; e a "concentração" projetava a mudança em span(U0),
   descartando ~24% da energia nas matrizes retangulares (`fc1`, `fc2`).

---

## 4. Etapa 3 — domínio onde o corpo precisa mudar, régua corrigida, teste causal

### O que foi feito

1. **Triagem de domínio** (corpo congelado + regressão logística, mesma subamostra do treino):
   lesões de pele **56,6%** · DTD (texturas) 78,6% · KMNIST 79,3% · EuroSAT 94,9% · pets 94,8%.
2. **Pilotos de fine-tuning** (1 semente, `weight_decay = 0`): pele 72,5% (**+15,9 pontos**), KMNIST 87,2% (+7,9),
   DTD 81,6% (+3,0). Escolhido: **lesões de pele** (`Pranavkpba2000/skin_cancer_small_dataset`, 8 classes; 3.680 de
   treino usadas, 2.840 de teste). Proveniência das fotos não documentada: serve como teste metodológico, não clínico.
3. **10 treinos reais** (72,84% ± 0,54; 81,6% no treino) e **10 com rótulos embaralhados** (permutação fixa, igual para
   todas as sementes; teste 12,0%, acaso 12,5%; treino 20,4%).
4. **Régua corrigida** (`medir_regua.py`), aplicada igualmente aos 40 modelos (etapas 1, 2, 3 e controle):
   - σ em todo o espectro, não só nos 10 maiores;
   - direções por **ângulos principais de subespaço** (k = 10, 50, 100), estáveis a degenerescência e sinal;
   - fração da energia na "parte σ" com a norma completa de ΔW;
   - **zero isotrópico**: as mesmas medidas para ruído gaussiano com a mesma energia da mudança real de cada
     matriz e semente, isto é, o que a régua marca sem nenhum aprendizado;
   - concordância entre sementes separada em parte σ (diagonal) e parte direções (fora da diagonal).
5. **Teste funcional por transplante** (`transplante.py`). Para cada modelo, as 72 matrizes recebem uma variante;
   bias, LayerNorm, embeddings e a cabeça ficam os do modelo fine-tuned:
   `completo` (W1) · `so_sigma` (W0 + D, D = U0·diag(diag(U0ᵀΔW V0))·V0ᵀ) · `so_direcoes` (W0 + ΔW − D) ·
   `direcoes_eq` (a parte direções reescalada para a **mesma energia** de D) · `nenhuma` (W0).
6. **Controle embaralhado:** cosseno entre mudanças real × real, embaralhado × embaralhado e real × embaralhado.

### Resultados

**Régua corrigida** (média das 72 matrizes; entre parênteses, o zero isotrópico de mesma energia):

| Medida | Pets, wd 0,05 | Pets, wd 0 | Pele, wd 0 | Pele, embaralhado |
|---|---|---|---|---|
| Deslocamento | 0,52% | 0,52% | 0,54% | 0,42% |
| \|Δσ/σ\|, 10 maiores | 0,019% (0,008) | 0,017% (0,008) | 0,018% (0,008) | 0,015% (0,006) |
| \|Δσ/σ\|, do 11º em diante | 0,20% (0,81) | 0,20% (0,81) | 0,21% (0,83) | 0,14% (0,62) |
| Giro vetor a vetor top-10 (régua antiga) | 0,95° (0,56) | 0,94° (0,56) | 0,97° (0,59) | 0,78° (0,46) |
| Subespaço top-10, ângulo principal médio | 0,38° (0,27) | 0,38° (0,27) | 0,40° (0,28) | 0,31° (0,22) |
| Subespaço top-50 | 0,37° (0,35) | 0,37° (0,35) | 0,37° (0,36) | 0,28° (0,28) |
| Fração da energia na parte σ | 0,30% (0,20) | 0,26% (0,20) | 0,26% (0,20) | 0,28% (0,20) |
| Concordância: inteira / parte σ / parte direções | 0,22 / 0,40 / 0,22 | 0,22 / 0,29 / 0,22 | 0,25 / 0,31 / 0,25 | 0,09 / 0,11 / 0,09 |

**Transplante** (acurácia de teste, média de 10 sementes):

| Variante | Pele | Pets (etapa 2) |
|---|---|---|
| `nenhuma` (72 matrizes originais) | 48,5% | 92,4% |
| `so_sigma` | 48,5% | 92,4% |
| `direcoes_eq` (mesma energia da parte σ) | 50,2% | 92,6% |
| `so_direcoes` | 72,8% | 94,1% |
| `completo` | 72,8% | 94,1% |

A parte σ tem ~0,24% da energia da mudança. Em pele, a mudança das 72 matrizes vale 24,4 pontos; a parte σ sozinha
recupera 0%, a parte direções 100%, e a parte direções com a mesma energia da parte σ recupera 7%.

**Controle embaralhado** (cosseno médio entre mudanças): real × real **0,253**; embaralhado × embaralhado 0,088;
real × embaralhado **0,034** (13% do real × real).

### O que foi aprendido

1. **A função está na parte que gira direções, não na que reescala σ.** É o resultado causal: transplantar só a
   parte σ não recupera nada do ganho; só a parte direções recupera tudo; e por unidade de energia as direções
   rendem 7% contra 0% dos σ.
2. **O tamanho e o tipo da mudança não dependem de a tarefa precisar dela.** Pele e pets dão praticamente os mesmos
   números em todas as medidas da régua, mas a mudança vale 24 pontos num caso e 1,7 no outro. Com passo e número
   de atualizações fixos, o otimizador move os pesos a mesma distância; o que varia é quanta função cabe nela.
3. **Contra o zero isotrópico, a parte σ é a mais estruturada, mas só no topo do espectro:** os 10 maiores σ mudam
   2,1× o que o ruído de mesma energia muda; o subespaço top-10 gira 1,4× o ruído; top-50 e top-100 ficam no zero.
   Do 11º σ em diante a mudança real é **menor** que a do ruído: o fine-tuning concentra a mudança espectral nos
   maiores σ e poupa os pequenos. A razão "87×" das etapas 1 e 2 fica aposentada: cada medida só faz sentido contra
   o seu zero.
4. **A mudança é em boa parte específica dos rótulos:** só 13% da concordância entre treinos reais sobrevive quando
   um deles usa rótulos embaralhados.
5. **Os dois lados da pergunta apontam para partes diferentes.** A reponderação dos σ é a parte mais **repetível**
   da mudança (concordância 0,31 contra 0,25 das direções, agora com uma medida imune à degenerescência); a parte
   direções é a mais **útil**.

---

## 5. Resposta atual à pergunta da tese, e seus limites

**Neste modelo (DINOv2-Small), neste protocolo (fine-tuning completo, passo 1e-5, 5 épocas), a especialização
acontece girando direções, não reponderando as que já existiam.** A reponderação dos σ é real, estruturada e
repetível, mas funcionalmente marginal.

**O que ainda não dá para afirmar:**

- que um fine-tuning **restrito a σ** (SVDiff, SVF) não funcionaria: aqui só se avaliou a parte σ que o fine-tuning
  completo depositou, que é minúscula (0,24% da energia); métodos que treinam σ diretamente otimizam essa parte com
  mais liberdade;
- que o resultado valha para outros passos de treino e números de épocas (o item 2 sugere que o passo dita o
  tamanho da mudança);
- que valha para outros modelos e para um dataset dermatológico com proveniência conhecida (HAM10000/ISIC).

Esses três pontos são os candidatos à etapa 4.

---

## 6. Lições de método (as que mais custaram)

- **Medir sem controle é contar história.** Cada versão da régua produziu "achados" que o controle seguinte desfez:
  o piso de ruído mal lido (etapa 1), a razão 87× (estrutural), o giro vetor a vetor (degenerescência), e a
  mudança sem efeito funcional (pets). O zero isotrópico e o transplante foram o que separou efeito de artefato.
- **Toda medida de peso precisa vir com a acurácia**, e com a comparação contra o corpo congelado; senão não dá
  para distinguir adaptação de deriva.
- **Um treino mal configurado destrói o modelo** (tentativa 0) e produz mudanças enormes que, medidas por SVD,
  pareceriam "aprendizado".
- **Diário reproduzível, nada sobrescrito.** Cada etapa tem script de reprodução, logs, ambiente registrado (versões,
  GPU, hashes) e os artefatos anteriores preservados; a etapa 3 reconstruiu duas páginas que tinham sido sobrescritas
  na etapa 1 e passou a gerar a página final a partir de uma cópia intocada.
