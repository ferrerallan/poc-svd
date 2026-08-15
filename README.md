# poc-svd

POC de **SVD (Singular Value Decomposition)** aplicada a uma imagem.

A ideia: uma imagem em tons de cinza é só uma matriz de números. A SVD quebra
essa matriz em uma pilha de **camadas** ordenadas por importância. Somando as
primeiras `N` camadas você já recupera quase a imagem inteira — usando bem menos
números do que a original.

## O que o script faz

[svd_imagem.py](svd_imagem.py) lê `entrada.png`, converte para tons de cinza e,
para as primeiras `NUM_LAYERS` camadas:

1. salva **cada camada isolada** em dois formatos:
   - `layer_NN.png` — versão visível, só para enxergar o padrão
   - `layer_NN.npy` — os números crus (já com o peso/sigma embutido), úteis para somar depois
2. salva a **reconstrução** com as `N` camadas somadas em `reconstructed.png`
3. imprime um relatório de compressão no terminal

Quanto maior o `NUM_LAYERS`, mais perto a reconstrução fica do original.

## Requisitos

```bash
pip install numpy pillow
```

## Como rodar

```bash
python svd_imagem.py
```

Formato da saída no terminal (com a `entrada.png` de 336 × 296 e 35 camadas):

```
image:             336 x 296  (99,456 numbers)
layers generated:  35 of 296
reconstruction:    22,155 numbers (22.3% of the original)
image captured:    <depende da imagem>%
everything saved:  saida/
```

Ou seja: nesta configuração a reconstrução guarda só **22,3% dos números** da
imagem original.

## Configuração

As três variáveis ficam no topo de [svd_imagem.py:25-27](svd_imagem.py#L25-L27):

| Variável      | Padrão         | O que é                                             |
| ------------- | -------------- | --------------------------------------------------- |
| `IMAGE_PATH`  | `"entrada.png"` | imagem de entrada (qualquer formato que o Pillow abra) |
| `NUM_LAYERS`  | `35`           | quantas camadas gerar (limitado a `min(altura, largura)`) |
| `OUTPUT_DIR`  | `"saida"`      | pasta onde tudo é salvo                             |

## Estrutura

```
.
├── svd_imagem.py       # o script inteiro
├── entrada.png         # imagem de entrada
├── saida/
│   ├── layer_01.png    # camada 1 isolada (visualização)
│   ├── layer_01.npy    # camada 1 isolada (números crus)
│   ├── ...
│   └── reconstructed.png
└── README.md
```

## Como funciona

A SVD decompõe a matriz da imagem `W` em `W = U · Σ · Vᵀ`. Cada camada `i` é o
produto externo de uma coluna de `U` por uma linha de `Vᵀ`, multiplicado pelo
peso `σᵢ`:

```python
camada_i = σᵢ * np.outer(U[:, i], Vt[i])
```

Os pesos `σ` vêm ordenados do maior para o menor — por isso a camada 1 carrega o
"esqueleto" da imagem e as últimas só acrescentam detalhe fino.

Somar as `N` primeiras camadas é o mesmo que a multiplicação de matrizes truncada:

```python
reconstruida = (U[:, :N] * σ[:N]) @ Vt[:N, :]
```

### Sobre os PNGs de camada

Uma camada isolada tem valores **negativos e positivos**. Para virar imagem, o
script centra o zero no cinza médio (128) e reescala pelo maior valor absoluto
([`to_visible_image`](svd_imagem.py#L55-L62)). Isso é **só para visualização** —
os números reais só existem nos `.npy`.

## Métricas do relatório

- **reconstruction** — cada camada custa `altura + largura + 1` números
  (uma coluna de `U`, uma linha de `Vᵀ` e um `σ`), contra `altura × largura` da
  imagem cheia
- **image captured** — a "energia" retida, `Σσᵢ² (usados) / Σσᵢ² (todos)`
