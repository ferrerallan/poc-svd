# poc-svd: SVD no estudo de transfer learning

Pesquisa de doutorado sobre **o que muda mais quando um modelo de visão é ajustado a uma tarefa nova: os pesos (valores singulares σ) ou as direções (vetores singulares U e V)**. O repositório conta a história dessa pesquisa, etapa por etapa.

**Comece por [`artigo/index.html`](artigo/index.html).** Ele explica os conceitos, resume cada etapa e lista, para cada uma, os scripts usados e os dados produzidos (seção 11).

## As etapas

| Etapa | Pasta | Pergunta |
| --- | --- | --- |
| 0 | [`doutorado/experimento0/`](doutorado/experimento0/) | Uma foto é uma matriz: quantas camadas de SVD bastam para reconhecê-la? |
| 1 | [`doutorado/experimento1/etapa1/`](doutorado/experimento1/etapa1/) | No fine-tuning do DINOv2-Small em raças de pet, mudam mais os σ ou as direções? |
| 2 | [`doutorado/experimento1/etapa2/`](doutorado/experimento1/etapa2/) | Parte da mudança era efeito do weight decay? |
| 3 | [`doutorado/experimento1/etapa3/`](doutorado/experimento1/etapa3/) | E num domínio distante (lesões de pele), com régua corrigida e controles? |
| 4 | [`doutorado/experimento1/etapa4/`](doutorado/experimento1/etapa4/) | Treinar só σ ou só a rotação das direções basta para aprender? |

Cada etapa tem os seus `scripts/`, os seus JSON, os seus `logs/` e uma receita `reproduce_etapaN.sh`. O código usado por mais de uma etapa fica em [`doutorado/experimento1/common/`](doutorado/experimento1/common/). O diário completo está em [`doutorado/experimento1/resultados.html`](doutorado/experimento1/resultados.html).

## Estrutura

```
artigo/        o artigo, as figuras e o verificador (check_repo.py)
doutorado/     as etapas, a lista de leitura e a jornada de aprendizado
learning/      resumos técnicos do que foi aprendido
plannings/     planos e revisões de mudanças no repositório
models/        pesos de todos os treinos (fora do git, cerca de 7 GB)
data/          datasets baixados (fora do git)
```

## Como rodar

Sempre a partir da raiz do repositório.

```bash
python3 doutorado/experimento0/svd_image.py          # etapa 0
bash doutorado/experimento1/etapa3/reproduce_etapa3.sh   # etapas 3 e 4 são idempotentes
python3 artigo/check_repo.py                          # confere rastreabilidade, padrão dos comentários e idioma
```

As receitas das etapas 1 e 2 retreinam os modelos e sobrescrevem `models/`; leia o cabeçalho antes de rodar.

## Convenções

- **Código em inglês, páginas em português.** Chaves dos JSON e nomes de pastas de dados e modelos seguem os nomes originais, porque os dados das etapas são congelados.
- **Padrão dos scripts:** docstring de 3 linhas (`what?`, `why?`, `how?`) em toda função e classe, e corpo de função dividido em regiões numeradas.
- **Nada é sobrescrito:** cada etapa guarda em `base/` cópias congeladas das páginas da anterior.

Requisitos: Python 3.10, PyTorch com CUDA, `transformers`, `safetensors`, `scikit-learn`, `pandas`, `matplotlib`, `Pillow`.
