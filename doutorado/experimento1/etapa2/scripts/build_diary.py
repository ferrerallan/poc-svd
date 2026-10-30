"""
* what? - Builds TWO pages from the JSONs of stages 1 and 2: diario.html (the log book) and resultados.html (stages 1-2).
* why?  - No number on the pages is typed by hand: everything comes from etapa1/, etapa2/ and the training reports.
* how?  - Loads aggregates and reports, computes the page tokens, fills diary_template.html and writes both pages.

diario.html      the log book (the detailed exploration of stage 1 opens in an iframe)
resultados.html  both stages in sequence, with ALL the content open in the page itself
Run from the repository root:  python3 doutorado/experimento1/etapa2/scripts/build_diary.py
Later stages insert their section into a frozen copy of resultados.html (see build_results_etapa3.py).
"""
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
import stages  # noqa: E402

EXPERIMENT = Path("doutorado/experimento1")
HERE = Path(__file__).resolve().parent
KEYS = ["tamanho_mudanca", "variacao_dos_valores", "giro_dos_vetores_graus", "concentracao"]
K = math.pi / 180 * 100   # radians -> %

# Page text of the sections that differ between the two pages (Portuguese: it is page content)
EXP1_HTML = ("<h4>Exploração detalhada da etapa 1: 4 medidas, piso de ruído e variação entre sementes</h4>"
             "<p>As mesmas medidas das versões v1 e v2 listadas acima, agora todas abertas aqui, uma abaixo da outra. Cada medida tem a sua ficha e três mapas: "
             "a adaptação (média dos 10 treinos), o piso de ruído (média de 45 pares de treinos) e a variação entre sementes.</p>"
             "<div class='caixa aviso'><span class='rot'>Cuidado ao ler o piso de ruído</span>"
             "<p>O piso de ruído compara <b>dois treinos entre si</b>. Se os treinos fossem totalmente independentes, a diferença entre eles seria cerca de <b>1,41 vez</b> a de um treino contra o original. "
             "No Deslocamento observa-se <b>[[DESL_RAZAO1]]</b>, coerente com o cosseno de <b>[[COSM1]]</b> entre as mudanças de dois treinos: parte da mudança é compartilhada. "
             "Por isso, nas medidas Deslocamento, Giro e Variação dos pesos, o ruído tende a sair <b>maior</b> que a adaptação sem que isso prove que a adaptação seja só acaso. "
             "A concentração é uma razão, e não sofre desse fator.</p></div><div id='exp1'></div>")
EXP2_HTML = ("<h4>Exploração detalhada da etapa 2: 4 medidas e variação entre sementes</h4>"
             "<p>As mesmas 4 medidas da etapa 1 (as fichas estão em 1.6), com a adaptação (média dos 10 treinos) e a variação entre sementes. "
             "Nesta etapa <b>não</b> foi calculado o piso de ruído (os 45 pares de treinos), porque a comparação principal era com a etapa 1.</p><div id='exp2'></div>")


def load_json(path):
    """
    * what? - Reads a JSON file into Python objects.
    * why?  - The builder reads dozens of JSON files; this keeps each read to one short call.
    * how?  - json.load on an opened file.
    """
    #region 1 - Read the file
    # what? - Parses the JSON.
    # how?  - The file handle is closed by the garbage collector, as in the original script.
    return json.load(open(path))
    #endregion


def num(x, decimals):
    """
    * what? - Formats a number in Brazilian Portuguese style (decimal comma).
    * why?  - The pages are read in Portuguese.
    * how?  - Fixed-point formatting, then the dot is replaced by a comma.
    """
    #region 1 - Format the number
    # what? - Applies the format.
    # how?  - f-string with a dynamic number of decimals.
    return f"{x:.{decimals}f}".replace(".", ",")
    #endregion


def range_text(agg, key):
    """
    * what? - Returns the min-max range of a training field over the 10 seeds as '12,3% a 45,6%'.
    * why?  - The text of the pages quotes accuracy ranges, not every seed.
    * how?  - Collects the field from agg['treino'] and formats the extremes.
    """
    #region 1 - Build the range
    # what? - Minimum and maximum of the field.
    # how?  - Uses num() with one decimal.
    v = [t[key] for t in agg["treino"]]
    return f"{num(min(v), 1)}% a {num(max(v), 1)}%"
    #endregion


def training_table(agg):
    """
    * what? - Builds the HTML table of the 10 trainings of a stage with accuracies and minutes.
    * why?  - Each stage section shows its raw training results.
    * how?  - One row per seed plus an average row of the final accuracy.
    """
    #region 1 - Build the rows
    # what? - Table rows for the seeds.
    # how?  - Cells carry the CSS class n for right-aligned numbers.
    rows = "".join(f"<tr><td class='n'>{t['semente']}</td><td class='n'>{num(t['antes'], 1)}%</td>"
                   f"<td class='n'>{num(t['depois'], 1)}%</td><td class='n'>{num(t['minutos'], 1)}</td></tr>" for t in agg["treino"])
    after = [t["depois"] for t in agg["treino"]]
    mean = sum(after) / len(after)
    #endregion

    #region 2 - Wrap in the table
    # what? - Header, rows and the average row.
    # how?  - The wrapper div makes the table scroll horizontally on narrow screens.
    return ("<div class='rolagem'><table><thead><tr><th class='n'>Semente</th><th class='n'>Acurácia antes</th>"
            "<th class='n'>Acurácia depois</th><th class='n'>Minutos de treino</th></tr></thead><tbody>" + rows +
            f"<tr class='dest'><td><b>média</b></td><td></td><td class='n'><b>{num(mean, 2)}%</b></td><td></td></tr></tbody></table></div>")
    #endregion


def ratio(agg):
    """
    * what? - Returns the direction percentage divided by the weight percentage of a stage.
    * why?  - The first reading of the thesis question was this ratio (later shown to be structural).
    * how?  - Ratio of the global d and p of the aggregate.
    """
    #region 1 - Divide
    # what? - Computes d / p.
    # how?  - Both come from agregado.json.
    return agg["global"]["d"] / agg["global"]["p"]
    #endregion


def count_larger(agg):
    """
    * what? - Counts the matrices whose direction percentage exceeds the weight percentage.
    * why?  - It is the number quoted as 'directions change more in N of 72 matrices'.
    * how?  - Sum over the cells.
    """
    #region 1 - Count
    # what? - Cells with d > p.
    # how?  - Generator expression.
    return sum(1 for c in agg["celulas"] if c["d"] > c["p"])
    #endregion


def repetition_html(rep):
    """
    * what? - Builds the HTML table and the interpretation of the same-seed repetition check (section 2.2).
    * why?  - Shows whether training is deterministic and how far two runs of the same seed are.
    * how?  - Reads repeticao.json fields and chooses the interpretation by the number of identical tensors.
    """
    #region 1 - Choose the interpretation
    # what? - Text for the deterministic and the non-deterministic cases.
    # how?  - Identical tensors equal to the total means bit-for-bit equal models.
    identical, total = rep["tensores_identicos"], rep["tensores_total"]
    if identical == total:
        interp = ("As duas execuções deram <b>exatamente o mesmo modelo</b>, tensor por tensor. Neste ambiente o procedimento é "
                  "<b>determinístico</b>: a mesma semente reproduz o resultado bit a bit.")
    else:
        interp = (f"As duas execuções <b>não</b> deram o mesmo modelo bit a bit ({identical} de {total} tensores idênticos), mas ficaram muito próximas. "
                  f"A diferença entre elas é, em média, <b>{num(100 * rep['diferenca_relativa_a_mudanca_do_treino_media'], 1)}%</b> do tamanho da mudança que o "
                  "fine-tuning causa nas matrizes. O procedimento é reprodutível em parâmetros e resultado, mas não é determinístico (a GPU não é).")
    #endregion

    #region 2 - Build the table
    # what? - Rows with the measures of the check.
    # how?  - Scientific notation for the largest absolute difference.
    return (
        "<div class='rolagem'><table><thead><tr><th>Medida</th><th class='n'>Valor</th></tr></thead><tbody>"
        f"<tr><td>Tensores idênticos entre as duas execuções</td><td class='n'>{identical} de {total}</td></tr>"
        f"<tr><td>Maior diferença absoluta em qualquer número do modelo</td><td class='n'>{rep['diferenca_maxima_absoluta']:.2e}</td></tr>"
        f"<tr><td>Diferença entre as execuções, em % da mudança causada pelo fine-tuning (média das 72 matrizes)</td><td class='n'>{num(100 * rep['diferenca_relativa_a_mudanca_do_treino_media'], 1)}%</td></tr>"
        f"<tr><td>Acurácia final, execução 1 e execução 2</td><td class='n'>{num(rep['acuracia_execucao_1'], 2)}% e {num(rep['acuracia_execucao_2'], 2)}%</td></tr>"
        f"<tr><td>Acurácia por época, execução 1</td><td class='n'>{' · '.join(num(x, 1) for x in rep['acuracias_por_epoca_1'])}</td></tr>"
        f"<tr><td>Acurácia por época, execução 2</td><td class='n'>{' · '.join(num(x, 1) for x in rep['acuracias_por_epoca_2'])}</td></tr>"
        f"</tbody></table></div><p>{interp}</p>")
    #endregion


def layer_agreement_table():
    """
    * what? - Builds the HTML table of agreement between trainings per layer, stage 1 versus stage 2.
    * why?  - Shows where in the network the trainings agree more, for weights and for directions.
    * how?  - Averages cos_pesos and cos_direcoes of each block from the two reproducibility JSONs.
    """
    #region 1 - Average per layer
    # what? - Four lists of 12 values.
    # how?  - One mean per block index for each stage and each measure.
    rp1, rp2 = load_json(stages.REPRODUCIBILITY[1]), load_json(stages.REPRODUCIBILITY[2])

    def per_layer(rp, key):
        """
        * what? - Mean of a key over the matrices of each of the 12 blocks.
        * why?  - The table has one row per layer.
        * how?  - Statistics mean over the entries of the block.
        """
        #region 1 - Average each block
        # what? - List of 12 means.
        # how?  - Filters the entries by block number.
        return [st.mean(x[key] for x in rp if x["bloco"] == b) for b in range(12)]
        #endregion

    l1p, l2p, l1d, l2d = per_layer(rp1, "cos_pesos"), per_layer(rp2, "cos_pesos"), per_layer(rp1, "cos_direcoes"), per_layer(rp2, "cos_direcoes")
    #endregion

    #region 2 - Build the table
    # what? - HTML with one row per layer.
    # how?  - Two decimals, Portuguese number format.
    return ("<div class='rolagem'><table><thead><tr><th class='n'>Layer</th><th class='n'>Pesos, etapa 1</th><th class='n'>Pesos, etapa 2</th>"
            "<th class='n'>Direções, etapa 1</th><th class='n'>Direções, etapa 2</th></tr></thead><tbody>" +
            "".join(f"<tr><td class='n'>{i}</td><td class='n'>{num(l1p[i], 2)}</td><td class='n'>{num(l2p[i], 2)}</td><td class='n'>{num(l1d[i], 2)}</td><td class='n'>{num(l2d[i], 2)}</td></tr>" for i in range(12)) +
            "</tbody></table></div>")
    #endregion


def detail_stage1():
    """
    * what? - Returns the data of the detailed explorer of stage 1: global and per-matrix adaptation and noise.
    * why?  - The page script draws its heat maps from this compact structure.
    * how?  - Reads resultado_10sementes.json and keeps mean and standard deviation of the four measures.
    """
    #region 1 - Read and compact the file
    # what? - Builds the global and per-cell dictionaries.
    # how?  - The nested lambda only keeps media and dp of a statistics entry.
    r = load_json(stages.TEN_SEEDS_JSON)
    pick = lambda d: {"media": d["media"], "dp": d["dp"]}
    cells = [{"b": c["bloco"], "t": c["tipo"], **{k: {"adaptacao": pick(c[k]["adaptacao"]), "ruido": pick(c[k]["ruido"])} for k in KEYS}} for c in r["celulas"]]
    glob = {k: {"adaptacao": pick(r["global"][k]["adaptacao"]), "ruido": pick(r["global"][k]["ruido"])} for k in KEYS}
    return {"n": len(r["sementes"]), "np": len(r["pares"]), "global": glob, "celulas": cells}
    #endregion


def detail_stage2():
    """
    * what? - Returns the data of the detailed explorer of stage 2: adaptation only, no noise pairs.
    * why?  - Stage 2 did not compute the 45 noise pairs, so the structure has no noise entries.
    * how?  - Computes mean and standard deviation over the 10 seeds from the comparison JSONs.
    """
    #region 1 - Load the comparisons
    # what? - Indexes each seed's rows by (block, type).
    # how?  - Seed 0 defines the list of matrices.
    res = {s: {(x["bloco"], x["tipo"]): x for x in load_json(stages.comparison(2, s))} for s in stages.SEEDS}
    keys = sorted(res[0])
    #endregion

    #region 2 - Summarize per matrix and globally
    # what? - Mean and standard deviation of each measure.
    # how?  - Global values are the statistics of the per-seed means over all matrices.
    cells = []
    for (b, t) in keys:
        entry = {"b": b, "t": t}
        for k in KEYS:
            v = [res[s][(b, t)][k] for s in stages.SEEDS]
            entry[k] = {"adaptacao": {"media": st.mean(v), "dp": st.stdev(v)}}
        cells.append(entry)
    glob = {}
    for k in KEYS:
        per_seed = [st.mean(res[s][c][k] for c in keys) for s in stages.SEEDS]
        glob[k] = {"adaptacao": {"media": st.mean(per_seed), "dp": st.stdev(per_seed)}}
    return {"n": len(stages.SEEDS), "np": 0, "global": glob, "celulas": cells}
    #endregion


def round_value(x):
    """
    * what? - Rounds every float inside nested dicts and lists to 5 significant digits.
    * why?  - Keeps the JSON embedded in the page small.
    * how?  - Recurses over dicts and lists; floats go through a 5g format.
    """
    #region 1 - Round recursively
    # what? - Returns a rounded copy.
    # how?  - Non-float scalars are returned untouched.
    if isinstance(x, float):
        return float(f"{x:.5g}")
    if isinstance(x, dict):
        return {k: round_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [round_value(v) for v in x]
    return x
    #endregion


def percent_tokens(prefix, agg, decomp):
    """
    * what? - Builds the page tokens (E1_*, E2_*) that come from one stage aggregate and its decomposition.
    * why?  - The template has placeholders like [[E1_P]] and every number must come from the data.
    * how?  - Formats the global statistics, the decomposition and the accuracy ranges with the prefix.
    """
    #region 1 - Format the tokens
    # what? - Dictionary placeholder name -> text.
    # how?  - The placeholder names are the contract with the template and stay unchanged.
    g = agg["global"]
    return {
        f"{prefix}_P": num(g["p"], 3), f"{prefix}_P2": num(g["p"], 2), f"{prefix}_D": num(g["d"], 2),
        f"{prefix}_D_GRAUS": num(g["d"] / 1.74533, 2), f"{prefix}_RAZAO": num(ratio(agg), 0),
        f"{prefix}_CP": num(g["cp"], 2), f"{prefix}_CD": num(g["cd"], 2), f"{prefix}_POS": str(g["pos"]),
        f"{prefix}_MAIOR72": str(count_larger(agg)),
        f"{prefix}_TOTAL_ABS": num(decomp["total_abs_pct"], 4), f"{prefix}_COMUM_ABS": num(decomp["comum_abs_pct"], 4),
        f"{prefix}_DIF_ABS": num(decomp["diferencial_abs_pct"], 4), f"{prefix}_FRAC_COMUM": num(decomp["fracao_energia_comum_pct"], 1),
        f"{prefix}_FRAC_DIF": num(100 - decomp["fracao_energia_comum_pct"], 1),
        f"{prefix}_MESMO_SENT": num(decomp["mesmo_sentido_pct"], 1), f"{prefix}_COMUM_SINAL": num(decomp["comum_media_com_sinal_pct"], 4),
        f"{prefix}_COMUM_POS": num(decomp["comum_positiva_pct"], 1), f"{prefix}_CONSIST": num(decomp["consistencia_media_de_10_treinos"], 1),
        f"{prefix}_ENCOLH": num(decomp["encolhimento_esperado_weight_decay_pct"], 4),
        f"{prefix}_ACC_ANTES": range_text(agg, "antes"), f"{prefix}_ACC_DEPOIS": range_text(agg, "depois"),
    }
    #endregion


def per_seed(stage):
    """
    * what? - Returns the weight % and direction % of every seed of a stage, averaged over the 72 matrices.
    * why?  - The stage 1 versus stage 2 comparison is made seed by seed with a Welch statistic.
    * how?  - Reads the 10 comparison JSONs; direction turn in degrees is converted to % with K.
    """
    #region 1 - Collect the per-seed means
    # what? - Two lists with one value per seed.
    # how?  - variacao_dos_valores is a fraction, so x100; giro is in degrees, so x K.
    ps, ds = [], []
    for s in stages.SEEDS:
        r = load_json(stages.comparison(stage, s))
        ps.append(100 * st.mean(x["variacao_dos_valores"] for x in r))
        ds.append(K * st.mean(x["giro_dos_vetores_graus"] for x in r))
    return ps, ds
    #endregion


def welch(a, b):
    """
    * what? - Welch t statistic of two samples (difference of means over the combined standard error).
    * why?  - Quantifies whether stage 1 and stage 2 differ beyond the seed-to-seed variation.
    * how?  - Sample variances divided by sample sizes, square-rooted, as the denominator.
    """
    #region 1 - Compute the statistic
    # what? - (mean a - mean b) / standard error.
    # how?  - Unequal variances allowed, which is the Welch form.
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return (st.mean(a) - st.mean(b)) / se
    #endregion


def mean_accuracy(agg):
    """
    * what? - Mean final accuracy over the 10 seeds of a stage.
    * why?  - Quoted in the text as the accuracy of the stage.
    * how?  - Average of the 'depois' fields.
    """
    #region 1 - Average
    # what? - Sum over count.
    # how?  - Plain Python sum.
    return sum(t["depois"] for t in agg["treino"]) / len(agg["treino"])
    #endregion


def data_json(agg):
    """
    * what? - Serializes the per-matrix weight % and direction % of a stage for the page script.
    * why?  - The heat map in the page reads this compact JSON.
    * how?  - Keeps b, t, p, d of each cell and dumps with minimal separators.
    """
    #region 1 - Serialize
    # what? - JSON string.
    # how?  - separators without spaces keep the page small.
    return json.dumps({"celulas": [{"b": c["b"], "t": c["t"], "p": c["p"], "d": c["d"]} for c in agg["celulas"]]}, separators=(",", ":"))
    #endregion


def render_page(mode, base, explorer_js, tokens, e1, e2, det1, det2):
    """
    * what? - Fills the template for one page mode ('diario' or 'resultados') and returns the HTML.
    * why?  - The two pages share the template and differ only in the marked sections.
    * how?  - Substitutes section markers, the explorer script and the [[TOKEN]] placeholders; asserts none is left.
    """
    #region 1 - Apply the mode-specific sections
    # what? - 'resultados' opens the explorers in the page; 'diario' removes the markers.
    # how?  - Regex over the DIARIO_ONLY block, then plain replaces.
    html = base
    if mode == "resultados":
        html = re.sub(r"<!--DIARIO_ONLY_START-->.*?<!--DIARIO_ONLY_END-->", EXP1_HTML, html, flags=re.S)
        html = html.replace("<!--EXP2-->", EXP2_HTML)
        js = explorer_js.replace("__DET1__", json.dumps(round_value(det1), separators=(",", ":"))).replace("__DET2__", json.dumps(round_value(det2), separators=(",", ":")))
        html = html.replace("/*EXPLORADOR_JS*/", js)
        title = "Experimento 1 — resultados das etapas 1 e 2"
        lede = " Esta página reúne, de uma vez, todo o conteúdo das duas etapas em sequência, inclusive a exploração detalhada de cada uma, sem depender de outras páginas."
    else:
        html = html.replace("<!--DIARIO_ONLY_START-->", "").replace("<!--DIARIO_ONLY_END-->", "").replace("<!--EXP2-->", "").replace("/*EXPLORADOR_JS*/", "")
        title, lede = "Experimento 1 — diário de bordo", ""
    #endregion

    #region 2 - Substitute the data and the tokens
    # what? - Fills __D1__, __D2__ and every [[TOKEN]].
    # how?  - Five passes, because some token values contain other tokens.
    html = html.replace("__D1__", data_json(e1)).replace("__D2__", data_json(e2))
    tk = {**tokens, "TITULO": title, "LEDE_EXTRA": lede}
    for _ in range(5):
        for k, v in tk.items():
            html = html.replace(f"[[{k}]]", v)
    #endregion

    #region 3 - Check that nothing is left unfilled
    # what? - Fails loudly on a leftover placeholder or marker.
    # how?  - Two assertions with the offending names in the message.
    missing = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", html)))
    assert not missing, f"tokens without value: {missing}"
    assert "__DET" not in html and "<!--EXP" not in html and "DIARIO_ONLY" not in html, "marker left unreplaced"
    return html
    #endregion


def main():
    """
    * what? - Builds diario.html and resultados.html of stages 1 and 2.
    * why?  - The two pages are the log book of the first two stages; they are regenerated, never edited by hand.
    * how?  - Load data, compute every token, then render the template once per mode and write both pages.
    """
    #region 1 - Load the stage data
    # what? - Aggregates, diagnosis, attempt 0, repetition and environment.
    # how?  - d1 and d2 are the weight decompositions stored inside the aggregates.
    e1, e2 = load_json(stages.STAGE_DIR[1] / "agregado.json"), load_json(stages.STAGE_DIR[2] / "agregado.json")
    diag = load_json(stages.STAGE_DIR[1] / "diagnostico.json")
    t0 = load_json("models/etapa1_tentativa0/relatorio.json")
    rep = load_json(stages.STAGE_DIR[2] / "repeticao.json")
    env = load_json(stages.STAGE_DIR[2] / "ambiente.json")
    d1, d2 = e1["decomposicao"], e2["decomposicao"]
    #endregion

    #region 2 - Build the table and detail blocks
    # what? - HTML blocks and explorer data.
    # how?  - achados_etapa2.html is the findings text written by hand after seeing the numbers.
    repetition = repetition_html(rep)
    findings_file = stages.STAGE_DIR[2] / "achados_etapa2.html"
    findings = findings_file.read_text() if findings_file.exists() else "<p>(achados ainda não escritos)</p>"
    layer_table = layer_agreement_table()
    det1, det2 = detail_stage1(), detail_stage2()
    g1 = det1["global"]["tamanho_mudanca"]
    #endregion

    #region 3 - Compute the stage 1 versus stage 2 statistics
    # what? - Per-seed means, Welch statistics and whole-matrix cosines.
    # how?  - dist_seeds is the mean relative distance between two trainings of stage 2.
    pkg = env["pacotes"]
    p1s, d1s = per_seed(1)
    p2s, d2s = per_seed(2)
    cm1, cm2 = load_json(stages.STAGE_DIR[1] / "cosseno_matriz.json"), load_json(stages.STAGE_DIR[2] / "cosseno_matriz.json")
    dist_seeds = cm2["distancia_relativa_media"]
    extras = {
        "E1_P_FULL": num(st.mean(p1s), 4), "E2_P_FULL": num(st.mean(p2s), 4),
        "E1_D_FULL": num(st.mean(d1s), 3), "E2_D_FULL": num(st.mean(d2s), 3),
        "E1_P_SD": num(st.stdev(p1s), 4), "E2_P_SD": num(st.stdev(p2s), 4),
        "DIF_P": f"{100 * (st.mean(p2s) - st.mean(p1s)) / st.mean(p1s):+.1f}".replace(".", ",").replace("-", "−"),
        "DIF_D": f"{100 * (st.mean(d2s) - st.mean(d1s)) / st.mean(d1s):+.1f}".replace(".", ",").replace("-", "−"),
        "T_P": num(abs(welch(p1s, p2s)), 1), "T_D": num(abs(welch(d1s, d2s)), 1),
        "COSM1": num(cm1["cosseno_medio"], 3), "COSM2": num(cm2["cosseno_medio"], 3),
        "BASE_COMUM": num(100 / 10, 1),
        "ACC1_MED": num(mean_accuracy(e1), 2), "ACC2_MED": num(mean_accuracy(e2), 2),
        "REP_REL": num(100 * rep["diferenca_relativa_a_mudanca_do_treino_media"], 0), "REP_IDENT": str(rep["tensores_identicos"]),
        "DIST_SEMENTES": num(100 * dist_seeds, 0),
        "REP_FRACAO": num(100 * (rep["diferenca_relativa_a_mudanca_do_treino_media"] / dist_seeds) ** 2, 1),
    }
    #endregion

    #region 4 - Assemble the token dictionary
    # what? - Every placeholder of the template with its text.
    # how?  - The AMB_* tokens describe the environment recorded in stage 2.
    tokens = {**percent_tokens("E1", e1, d1), **percent_tokens("E2", e2, d2), **extras,
              "COLADOS": str(e1["colados"]), "PARES_TOP": str(e1["pares_top"]),
              "T0_EPOCAS": " → ".join(num(x, 1) + "%" for x in t0["acuracias_por_epoca"]),
              "DIAG_REEXEC": num(diag["acuracia_corpo_congelado"], 1) + "%",
              "TAB_E1": training_table(e1), "TAB_E2": training_table(e2),
              "REPETICAO_HTML": repetition, "TAB_LAYER": layer_table, "ACHADOS2_HTML": findings,
              "AMB_PYTHON": env["python"], "AMB_TORCH": pkg["torch"], "AMB_TV": pkg["torchvision"],
              "AMB_TRANSFORMERS": pkg["transformers"], "AMB_NUMPY": pkg["numpy"], "AMB_PIL": pkg["PIL"], "AMB_SKLEARN": pkg["sklearn"],
              "AMB_GPU": env["gpu"], "AMB_DRIVER": env["driver_nvidia"], "AMB_VRAM": env["vram_total"],
              "AMB_COMMIT": env["git_commit"][:10], "AMB_NAOVERS": ", ".join(x.replace("?? ", "") for x in env["git_arquivos_nao_versionados"]) or "nenhum",
              "AMB_SHA_MODELO": env["sha256_modelo_original"][:16] + "…",
              "AMB_SHA_DADOS": " · ".join(f"<code>{k}</code> {v[:12]}…" for k, v in env["sha256_dados"].items())}

    tokens["DESL_RAZAO1"] = num(g1["ruido"]["media"] / g1["adaptacao"]["media"], 2)
    c1, c2 = det1["global"]["concentracao"]["adaptacao"], det2["global"]["concentracao"]["adaptacao"]
    tokens.update({"CONC1": num(c1["media"], 2), "CONC2": num(c2["media"], 2), "CONC1_SD": num(c1["dp"], 2), "CONC2_SD": num(c2["dp"], 2)})
    #endregion

    #region 5 - Render and write both pages
    # what? - One page per mode.
    # how?  - The template and the explorer script sit next to this file.
    base = (HERE / "diary_template.html").read_text()
    explorer_js = (HERE / "explorer_template.js").read_text()
    for mode, file in (("diario", "diario.html"), ("resultados", "resultados.html")):
        page = render_page(mode, base, explorer_js, tokens, e1, e2, det1, det2)
        (EXPERIMENT / file).write_text(page)
        print(f"{file}: {len(page.encode()) / 1024:.1f} KB")
    #endregion


if __name__ == "__main__":
    main()
