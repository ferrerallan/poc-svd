# Transfer Learning do Zero

> Diário de aprendizado · Doutorado
>
> A jornada completa de uma tarde: de "o que é um modelo treinado?" até derrubar
> os próprios achados com um experimento de controle.

**8 episódios · 1 modelo quebrado · 5 achados, 4 mortos · RTX 3060**

```
1 ──── 2 ──── 3 ──── 4 ──── 5 ──── 6 ──── 7 ──── 8
   o     calma   pra     o que    384     a      o     o
problema         que    tem dentro pergun-  lupa   erro  piso
                 serve             tas
```

---

## Episódio 1 — O problema de verdade

*onde a POC de imagem revela que era só aquecimento*

> **Você:** "Até aqui eu fiz somente testes com SVD e conceito para poder entender.
> Meu problema real é tentar usar SVD no processo de transfer learning de foundation
> models: descobrir quais são as coisas que mais mudam — os pesos ou os vetores."

O repositório tinha uma POC simples: uma imagem em tons de cinza quebrada em camadas
por SVD. Parecia o trabalho. Era o aquecimento.

A pergunta real é de doutorado: quando um foundation model se especializa numa tarefa
nova, o que muda nos pesos dele?

**A pergunta, em uma frase:** o modelo *reorganiza as ferramentas que já tinha*, ou
*compra ferramenta nova*?

E o produto não é um método novo de adaptação. É um **framework de orientação**: dizer
para quem vem depois onde vale a pena mexer, e economizar semanas de GPU.

**O que ficou.** A pergunta já tem resposta parcial na literatura — FSGAN, SVDiff, SVF
e OFT já apostaram em um lado ou no outro, e cada um funcionou. Então a contribuição
não pode ser "descobrir quem muda mais". Tem que ser **um jeito padronizado de medir**.

---

## Episódio 2 — "Calma"

*onde o aluno corrige o orientador, duas vezes*

A primeira resposta veio com decomposição matricial, tabela de datasets, protocolo de
treino e cinco controles metodológicos. Tudo de uma vez.

> **Você:** "Calma. Memorize isso: estou no início e você precisa usar linguagem
> simples. Nunca me dê um monte de instruções de uma vez."

Mudou o formato. Mas ainda não o suficiente — três conceitos novos numa mensagem curta
continuam sendo três conceitos novos.

> **Você:** "Tá indo muito rápido. Precisa esperar eu confirmar entendimento de cada
> trecho."

**O que ficou.** O pedaço certo é **um conceito por vez**. Explicar, parar, esperar.
Esse episódio não é sobre SVD, mas é o que tornou todos os outros possíveis.

---

## Episódio 3 — "Mas pra que serve isso?"

*onde a pergunta mais básica era a que faltava*

Vieram várias perguntas seguidas, e todas apontavam para o mesmo buraco:

> **Você:** "Não está claro o que você está me propondo nesta POC."
>
> "Mas eu preciso fornecer imagens?"
>
> "Ainda não está claro então o objetivo do transfer learning."

A resposta tinha ficado presa no "como" sem nunca ter dito o "por quê".

**O objetivo, enfim:** transfer learning serve para **não começar do zero**.

|                     | Do zero | Transfer learning |
| ------------------- | ------- | ----------------- |
| Fotos necessárias   | milhões | 3.680             |
| Tempo               | dias    | minutos           |
| Acerto              | ~20%    | ~90%              |

Enxergar é quase tudo o mesmo trabalho, seja para reconhecer um carro ou um Beagle.
Borda é borda, pelo é pelo. O que muda de tarefa para tarefa é só o final.

**O que ficou.** E a sua tese é a pergunta seguinte: **quanto da parte grande e comum
precisou ser mexido?** Todo mundo faz transfer learning e ninguém sabe responder isso
direito.

---

## Episódio 4 — O que tem dentro de um modelo

*onde 85 MB de números encontram 547 bytes de receita*

> **Você:** "Eu ainda estou confuso sobre a composição do que é um modelo treinado.
> O que eu tenho baixado no dinov2-small?"

**A virada:** um modelo **não é um programa**. Não tem código, não tem lógica, não roda.

É um monte de números mais um papelzinho dizendo como organizá-los. Você não comprou um
bolo pronto: recebeu os **ingredientes já pesados** e a **receita**. O cozinheiro
(PyTorch) já morava na sua cozinha.

```
model.safetensors         85 MB    os números (223 tabelas)
config.json              547 B     a receita
preprocessor_config.json 436 B     como preparar a foto
README.md                  3 KB    texto para humano
```

O detalhe que impressiona: a receita inteira cabe em meio KB.

```json
"hidden_size": 384          // a bitola
"num_hidden_layers": 12     // os 12 blocos
"patch_size": 14            // o quadradinho de 14x14
"num_channels": 3           // as 3 cores
```

Esse arquivinho descreve a estrutura inteira. O PyTorch lê, monta a estrutura vazia, e
depois preenche com os 22 milhões de números do `safetensors`. É ali que mora tudo o
que o modelo aprendeu com 142 milhões de fotos.

**O que ficou.** Houve também a confusão entre **223** e **256**: 223 são as tabelas do
modelo (fica no disco, nunca muda), 256 são os pedaços em que a foto é cortada (muda a
cada foto). O manual e o caso. **Sua tese mede o manual.**

---

## Episódio 5 — As 384 perguntas

*onde a foto vira números e o aluno pega duas simplificações no flagra*

O modelo corta a foto em 256 quadradinhos de 14×14 pixels. Para cada um, faz 384
perguntas: "tem risca em pé?", "é escuro embaixo?", "tem muito vermelho?". Cada resposta
é um número.

> **Você:** "384 perguntas. De onde veio este número 384?"

**Orientador:** alguém escolheu. É a **bitola** do modelo: quantos números ele usa para
descrever cada pedacinho. No Small é 384, no Large é 1024.

Depois veio a melhor intuição do dia:

> **Você:** "Estas perguntas são sempre em relação ao restante da imagem?"

**A resposta.** Nessa primeira etapa, não. Imagine **256 pessoas em salas fechadas**,
cada uma com um pedacinho da foto, preenchendo um formulário de 384 campos. Nenhuma sabe
que juntas estão olhando um gato.

Mas o instinto estava certo: no passo seguinte elas **se juntam numa roda e conversam**.
Essa conversa é a **atenção**, e é de onde vem o nome Transformer.

Na roda entram dois personagens a mais, que estavam visíveis na árvore do arquivo desde
o começo:

- **O secretário** (`cls_token`) — uma cadeira a mais, que não recebeu pedaço nenhum da
  foto. Ele só escuta as 12 rodadas e anota. No fim, é a ele que você pergunta "o que era
  essa foto?"
- **Os carimbos de posição** (`position_embeddings`) — sem eles, os 256 pedaços chegam
  como um monte picotado, e ninguém sabe qual veio de onde.

E depois da conversa, cada pedacinho vai pensar sozinho: é o **MLP**, as matrizes `fc1`
e `fc2`.

**Atenção × MLP.** Você vai a uma reunião e **escuta** todo mundo. Sai com informação
solta na cabeça — isso é a atenção. Depois volta para sua mesa e **pensa**: "juntando
tudo o que ouvi, a conclusão é essa" — isso é o MLP. Sem o segundo passo, você só
acumulou fofoca.

### O flagrante

> **Você:** "Estes 384 são realmente números ou arrays?"

A frase "você dá uma foto, ele devolve 384 números" estava simplificada. Rodando de
verdade com a `entrada.png` do próprio repositório:

```
foto original:       (336, 296)
depois de preparada: (1, 3, 224, 224)
saída completa:      (1, 257, 384)   <- uma TABELA
só o secretário:     (384,)          <- aí sim, uma lista
```

257 = 256 quadradinhos + 1 secretário. Cada um entrega a própria folha de 384 campos.

**O que ficou.** E o segundo flagrante: os 384 da entrada **não são os mesmos 384 da
saída**. A quantidade é a mesma, o significado não. Na entrada são riscas e cores;
depois de 12 rodadas, são coisas como "isso faz parte de uma orelha". 384 é o tamanho do
formulário, não o conteúdo dele.

> **Você:** "Ele aprendeu a enxergar coisas mas não sabe o que são?"

**Orientador:** exatamente. Ele sabe **distinguir** e **agrupar**. Não sabe **nomear**.
Juntou todos os Beagles num canto do espaço numérico, só que o canto não tem placa.

---

## Episódio 6 — A lupa

*onde a SVD deixa de ser matemática e vira instrumento*

**O armário.** Cada matriz do modelo é um **armário com 384 ferramentas**. A SVD abre o
armário e mostra duas listas:

- os **vetores singulares** = quais ferramentas existem
- os **valores singulares** = o quanto cada uma é usada

Você abre antes do treino, abre depois, e compara. Se só mudou o uso, mudaram os valores.
Se apareceu ferramenta nova, mudaram os vetores.

> **Você:** "As ferramentas são as perguntas?"

**Orientador:** quase. As perguntas são as **linhas** da tabela. As ferramentas da SVD
são os **temas por trás** das perguntas — porque muitas das 384 perguntas são variações
da mesma coisa.

Rodando na matriz `query` do bloco 0 do DINOv2:

```
sigma_1     4,79        1º valor:       25,8% de tudo
sigma_2     1,91        10 primeiros:   51,9%
sigma_3     1,85        50 primeiros:   82,6%
...                     100 primeiros:  96,2%
sigma_384   0,0000022
```

Metade do que essa camada faz está em 10 direções, de 384 possíveis.

> **Você:** "Me explique o que é o sigma neste caso."

**O equalizador.** Sigma é **o quanto a matriz amplifica naquela direção**. Pense num
equalizador de som com 384 controles deslizantes: o **vetor** diz qual faixa é aquele
controle, o **sigma** diz a que altura ele está.

### O segundo flagrante

> **Você:** "Ok, mas nesta lista sabendo só o sigma e não a direção que está falando."

Correto. O sigma diz **o quanto**, nunca **o quê**. A direção estava no `U`, que ainda
não tinha sido mostrado. Impressa, ela é isto:

```
[-0.0417, 0.0444, 0.0583, -0.0510, -0.0678, -0.0314, ...]

384 números, entre -0,12 e +0,12
soma dos quadrados = 1,0000
```

Sem nome, sem legenda, sem nada. Toda direção da SVD tem comprimento exatamente 1 — é
uma seta padronizada, muda só para onde aponta. A força foi toda extraída para o sigma.

**Voltando ao equalizador:** você sabe a altura de cada controle. Mas **as faixas não
têm etiqueta**. Ninguém escreveu "grave" ou "agudo" embaixo.

### A pergunta mais afiada do dia

> **Você:** "Não dá pra saber os critérios que ele usou."

**O que ficou.** Está certo: ninguém escreveu esses critérios. Mas eles estão **gravados
nos pesos**, e a SVD os separa: os vetores são *quais* critérios, os sigmas são *quanto
pesa* cada um.

Essa objeção é exatamente o motivo de a SVD ser a lente certa para a tese. Ela saiu da
sua desconfiança, não da teoria.

### A armadilha do sinal

Para medir se uma direção girou, mede-se o ângulo entre a seta de antes e a de depois.
Mas uma direção e a mesma direção invertida são, para a matemática, idênticas — a SVD
devolve qualquer uma das duas, sem critério.

**A faixa da estrada.** Você pode dizer que ela vai de norte para sul ou de sul para
norte. É a mesma faixa. Se num dia você anota uma coisa e no outro dia a outra, parece
que a estrada girou 180° — mas quem mudou foi a sua anotação.

Conserto: usar o valor absoluto do cosseno. Uma linha de código. O perigo não é a
dificuldade, é não saber que o problema existe.

---

## Episódio 7 — O experimento que destruiu o modelo

*onde o orientador erra feio e o erro vira a melhor aula*

Montagem: DINOv2-Small, 37 raças de cachorro e gato, Oxford-IIIT Pet, fine-tuning
completo. A checagem de sanidade passou — 2,1% antes de treinar, que é 1 em 37.

E aí:

```
época 1/5   acurácia 4.1%
época 2/5   acurácia 5.2%
época 3/5   acurácia 9.7%
```

**O que deu errado.** A cabeça nasce aleatória, chuta, erra feio e gera um erro enorme.
Esse erro volta para o corpo com força total, **antes de a cabeça aprender a ler**.

É um especialista com 20 anos de estudo colocado para trabalhar com um estagiário no
primeiro dia. O estagiário erra tudo, o chefe culpa o especialista, e o especialista muda
todo o método dele por causa do erro de um novato.

O diagnóstico: congelar o corpo por completo e treinar só uma regressão logística sobre
os 384 números do secretário.

| Situação                          | Acurácia  |
| --------------------------------- | --------- |
| corpo congelado, só a cabeça      | **94,8%** |
| corpo destravado, 3 épocas        | **9,7%**  |

Os 94,8% provaram que o corpo estava intacto e os dados, corretos. **O problema era o
método de treino.** Destravar o corpo tinha piorado o modelo em 85 pontos.

Conserto, duas linhas: congelar o corpo na primeira época e baixar o passo de `5e-5`
para `1e-5`. Resultado:

```
época 1/5  corpo travado   acurácia 91.6%
época 2/5  corpo solto     acurácia 91.9%
época 3/5  corpo solto     acurácia 93.9%
época 4/5  corpo solto     acurácia 93.8%
época 5/5  corpo solto     acurácia 94.2%     1,6 min de treino
```

**Por que esse erro importa para a tese.** Se a SVD tivesse sido medida no modelo
quebrado, os pesos teriam mudado muito — e a conclusão seria besteira, porque aquilo era
**destruição**, não especialização.

Regra que nasceu daqui: **toda medição de peso precisa ser reportada junto com a
acurácia final.**

### Os outros tropeços do caminho

- O servidor de Oxford entregava o dataset a **70 KB/s** — 2 horas de espera. O mesmo
  arquivo no espelho do HuggingFace: **21 MB/s**, 300 vezes mais rápido.
- Um `head -40` matou o script de comparação antes de ele salvar o resultado.
- Um `global` mal colocado quebrou o script de treino na segunda rodada.
- Passar a pasta em vez do arquivo `.safetensors` deu um erro de sistema críptico,
  "No such device".

Nenhum deles é interessante sozinho. Juntos mostram a proporção real do trabalho: o
conceito é uma fração do tempo.

---

## Episódio 8 — O piso de ruído

*onde quatro dos cinco achados morrem, e é a melhor notícia do dia*

Com o modelo treinado, as 72 matrizes foram comparadas antes e depois. Saíram cinco
achados, e eles pareciam bons:

- deslocamento médio das 72 matrizes: **0,52%**
- giro médio das direções principais: **0,96°**
- acurácia final: **94,2%**

A narrativa estava pronta: especializar um foundation model é um ajuste fino, não uma
reforma; o meio da rede muda mais que as pontas; a mudança é quase toda rotação.

Faltava uma coisa só. Treinar **de novo**, mudando apenas a semente aleatória, e medir a
diferença entre os dois modelos treinados. Nenhum aprendizado novo ali, só o acaso do
treino. Esse número é o zero da régua.

|                        | Desloc.    | Giro       | Var. σ     | Concentr. |
| ---------------------- | ---------- | ---------- | ---------- | --------- |
| adaptação (semente 0)  | 0,516%     | 0,964°     | 0,019%     | 3,30      |
| adaptação (semente 1)  | 0,529%     | 0,981°     | 0,019%     | 3,14      |
| **PISO DE RUÍDO**      | **0,651%** | **1,069°** | **0,018%** | **2,16**  |

**O veredito.** O deslocamento entre os dois modelos treinados é **maior** que o
deslocamento causado pelo treino. Razão sinal/ruído: **0,79x**.

Aqueles 0,5% apresentados como "o que a especialização fez" **não são efeito da
especialização**.

| Achado                                | vs ruído  | Situação                  |
| ------------------------------------- | --------- | ------------------------- |
| A mudança é minúscula                 | —         | sobrevive, mas era óbvio  |
| ~~Os valores singulares não mudaram~~ | 1,08x     | empate com o acaso        |
| ~~Quase tudo é rotação~~              | 0,90x     | abaixo do acaso           |
| ~~O meio da rede muda mais~~          | —         | provavelmente ruído       |
| **O MLP se comporta diferente**       | **1,95x** | **sobreviveu**            |

O único sinal claramente acima do acaso:

```
tipo        adaptação    ruído    razão
mlp.fc2        8,11       4,16    1,95x   <-- sobreviveu
mlp.fc1        5,78       4,06    1,42x
query          1,94       1,44    1,35x
dense          1,09       1,01    1,08x
```

E repare: os dois maiores são `fc1` e `fc2` — justamente o MLP, aquela parte onde cada
pedacinho "pensa sozinho" depois da conversa.

**O que ficou.** Sem controle, a medição apontava cinco achados. Com controle, sobra um
— e esse um vale muito mais que os cinco, porque agora tem com o que ser comparado.

Esse é o experimento funcionando como devia. A diferença entre **um resultado** e **um
achado** é exatamente esse segundo treino de um minuto e meio.

---

## Encerramento — o que a jornada ensinou

### Sobre o conteúdo

- Um modelo treinado é **números + receita**, não um programa.
- Ele aprende a **enxergar** sem aprender a **nomear**. Nomear é barato.
- A SVD separa **quais critérios** de **quanto pesa cada um** — e é por isso que ela é a
  lente certa para a pergunta da tese.
- Especializar um foundation model mexe muito pouco nos pesos.

### Sobre o método

- **Medir sem controle é contar história.** O piso de ruído é o que separa resultado de
  achado.
- Um experimento que falha e é diagnosticado ensina mais que um que dá certo de primeira.
- Toda medida de peso precisa vir acompanhada da acurácia — senão não dá para distinguir
  adaptação de destruição.
- As perguntas que pareciam básicas ("de onde veio o 384?", "são números ou arrays?",
  "não dá pra saber os critérios") foram as que mais mudaram o rumo.

### O que vem a seguir

1. **Teste funcional.** Devolver ao modelo só a parte diagonal do deslocamento, depois só
   a rotação, e medir a acurácia de cada um. Transforma geometria em causa.
2. **Domínio distante.** Repetir tudo com imagem médica, e comparar o mapa de
   concentração com o de raça de pet.
3. **A segunda armadilha**, ainda não enfrentada: valores singulares colados tornam as
   direções instáveis — elas giram sozinhas sem nada ter sido aprendido.

---

*Jornada de 28 de setembro de 2026, em uma RTX 3060 12 GB. Todos os números vieram de
execuções reais registradas em `doutorado/experimento1/`. As falas marcadas como suas são
citações do que você escreveu. Nada aqui é estimativa.*
