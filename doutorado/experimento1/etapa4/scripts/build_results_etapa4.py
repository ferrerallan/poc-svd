"""
* what? - STAGE 4: inserts the stage 4 section into doutorado/experimento1/resultados.html.
* why?  - The page is the log book of the experiment; each stage adds its section to a frozen copy of the previous page.
* how?  - Reads the stage 4 JSONs and training reports, fills etapa4_template.html and splices it into the frozen base page.

Base: etapa4/base/resultados_ate_etapa3.html (never edited). Everything comes from the JSONs of etapa4/ and the relatorio.json
of the trainings. Idempotent. Page regeneration chain:
  etapa2/scripts/build_diary.py (stages 1-2) -> etapa3/scripts/build_results_etapa3.py -> etapa4/scripts/build_results_etapa4.py
Run from the repository root:  python3 doutorado/experimento1/etapa4/scripts/build_results_etapa4.py
"""
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa4 as C  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ["completo", "so_rotacao", "so_sigma", "congelado_72"]
LABEL = {"completo": "completo", "so_rotacao": "só rotação", "so_sigma": "só σ", "congelado_72": "72 congeladas"}
NOTE = {"completo": " (etapa 3)", "so_rotacao": "", "so_sigma": "", "congelado_72": ""}


def load_json(path):
    """
    * what? - Reads a JSON file into Python objects.
    * why?  - The builder reads dozens of JSON files.
    * how?  - json.load on an opened file.
    """
    #region 1 - Read the file
    # what? - Parses the JSON.
    # how?  - The handle is closed by the garbage collector.
    return json.load(open(path))
    #endregion


def num(x, decimals):
    """
    * what? - Formats a number in Brazilian Portuguese style (decimal comma).
    * why?  - The page is read in Portuguese.
    * how?  - Fixed-point formatting, then the dot is replaced by a comma.
    """
    #region 1 - Format the number
    # what? - Applies the format.
    # how?  - f-string with a dynamic number of decimals.
    return f"{x:.{decimals}f}".replace(".", ",")
    #endregion


def thousands(x):
    """
    * what? - Formats an integer with dots as thousand separators (12.345).
    * why?  - Parameter counts are quoted in Portuguese style.
    * how?  - Comma grouping, then commas become dots.
    """
    #region 1 - Format the integer
    # what? - Applies the grouping.
    # how?  - f-string with the , format.
    return f"{x:,}".replace(",", ".")
    #endregion


def paired(a, b):
    """
    * what? - Returns the mean paired difference, its t statistic and the number of seeds where a is larger.
    * why?  - Variants share seeds, so the comparison is paired by seed.
    * how?  - Differences by seed; t = mean / (sd / sqrt(n)); infinity when the sd is zero.
    """
    #region 1 - Compute the paired statistics
    # what? - (mean difference, t, positive count).
    # how?  - Same formulas as the original script.
    d = [x - y for x, y in zip(a, b)]
    return st.mean(d), st.mean(d) / (st.stdev(d) / math.sqrt(len(d))) if st.stdev(d) > 0 else float("inf"), sum(x > 0 for x in d)
    #endregion


def step_text(PAS, variant):
    """
    * what? - Builds the sentence that lists the tested steps of a variant and the chosen one.
    * why?  - Section 4.1 summarizes the pilots in prose.
    * how?  - Sorts the pilots by lr and joins their validation accuracies.
    """
    #region 1 - Build the sentence
    # what? - Text with steps, accuracies and the choice.
    # how?  - The chosen step is shown in bold.
    ps = sorted(PAS[variant]["pilotos"], key=lambda x: x["lr"])
    chosen = PAS[variant]["lr"]
    steps = ", ".join(p["lr_str"] for p in ps)
    values = ", ".join(num(p["validacao_final"], 1) + "%" for p in ps)
    return f"{LABEL[variant]}: passos {steps} deram {values} na validação; escolhido <b>{chosen}</b>."
    #endregion


def training_table(R, variant):
    """
    * what? - Builds the HTML table of the 10 trainings of one variant with an average row.
    * why?  - Section 4.2 shows the raw results of every variant.
    * how?  - One row per seed from the relatorio.json plus mean and standard deviation of the final accuracy.
    """
    #region 1 - Build the rows
    # what? - Rows and summary values.
    # how?  - Cells carry the CSS class n for right-aligned numbers.
    rs = R[variant]
    rows = "".join(f"<tr><td class='n'>{r['semente']}</td><td class='n'>{num(r['acuracia_antes'], 1)}%</td><td class='n'>{num(r['acuracia_treino_por_epoca'][-1], 1)}%</td><td class='n'>{num(r['acuracia_depois'], 1)}%</td><td class='n'>{num(r['minutos_de_treino'], 1)}</td></tr>" for r in rs)
    after = [r["acuracia_depois"] for r in rs]
    #endregion

    #region 2 - Wrap in the table
    # what? - Title, header, rows and the average row.
    # how?  - The wrapper div scrolls horizontally on narrow screens.
    return (f"<h4>{LABEL[variant]}</h4><div class='rolagem'><table><thead><tr><th class='n'>Semente</th><th class='n'>Teste, antes</th><th class='n'>Treino, última época</th><th class='n'>Teste, depois</th><th class='n'>Minutos</th></tr></thead><tbody>" + rows +
            f"<tr class='dest'><td><b>média</b></td><td></td><td class='n'>{num(st.mean(r['acuracia_treino_por_epoca'][-1] for r in rs), 1)}%</td><td class='n'><b>{num(st.mean(after), 2)}%</b> ± {num(st.stdev(after), 2)}</td><td></td></tr></tbody></table></div>")
    #endregion


def build_tokens(MED, PAS, R):
    """
    * what? - Builds every [[TOKEN]] of the stage 4 template from the measures, the chosen steps and the reports.
    * why?  - No number of the page is typed by hand.
    * how?  - One region per section (4.0 to 4.5); the token names are the contract with etapa4_template.html.
    """
    #region 1 - Section 4.0: parameter counts
    # what? - Tokens P_SIG, P_ROT and P_RESTO.
    # how?  - Counts come from the training reports of seed 0.
    tok = {}
    tok["P_SIG"] = thousands(R["so_sigma"][0]["params_restritos"])
    pr = R["so_rotacao"][0]
    tok["P_ROT"] = f"{pr.get('params_restritos_efetivos', pr['params_restritos']):,} efetivos ({pr['params_restritos']:,} armazenados)".replace(",", ".")
    tok["P_RESTO"] = thousands(R["so_sigma"][0]["params_resto_corpo"])
    #endregion

    #region 2 - Section 4.1: step choice
    # what? - The pilots table and its text.
    # how?  - The chosen row is highlighted with the CSS class dest.
    rows = ""
    for variant in ("so_sigma", "so_rotacao"):
        for p in PAS[variant]["pilotos"]:
            chosen = p["lr_str"] == PAS[variant]["lr"]
            rows += (f"<tr{' class=dest' if chosen else ''}><td>{LABEL[variant]}</td><td class='n'>{p['lr_str']}</td><td class='n'>{' · '.join(num(x, 1) for x in p['validacao_por_epoca'])}</td>"
                     f"<td class='n'>{num(p['validacao_final'], 1)}%</td><td class='n'>{num(p['treino_final'], 1)}%</td><td>{'escolhido' if chosen else ''}</td></tr>")
    tok["TAB_PASSO"] = ("<div class='rolagem'><table><thead><tr><th>Variante</th><th class='n'>Passo</th><th class='n'>Validação por época</th><th class='n'>Validação final</th><th class='n'>Treino final</th><th></th></tr></thead><tbody>" + rows + "</tbody></table></div>")
    tok["LR_SIG"], tok["LR_ROT"] = str(PAS['so_sigma']['lr']), str(PAS['so_rotacao']['lr'])
    tok["TXT_PASSO"] = step_text(PAS, "so_sigma") + " " + step_text(PAS, "so_rotacao") + " O passo escolhido de cada variante foi usado nas 10 sementes com o conjunto de treino inteiro e avaliação no teste."
    #endregion

    #region 3 - Section 4.2: trainings
    # what? - One table per restricted variant.
    # how?  - C.VARIANTS is the list of variants trained in this stage.
    tok["TAB_TREINOS"] = "".join(training_table(R, v) for v in C.VARIANTS)
    #endregion

    #region 4 - Section 4.3: summary table and paired tests
    # what? - Accuracy per variant, share of the gain recovered and paired statistics.
    # how?  - rec(v) is the share of (complete - frozen) recovered by variant v.
    acc = {v: [r["acuracia_depois"] for r in R[v]] for v in ORDER}
    frozen, complete = st.mean(acc["congelado_72"]), st.mean(acc["completo"])

    def rec(v):
        """
        * what? - Share of the gain (complete minus frozen) that variant v recovers, in percent.
        * why?  - The headline of stage 4 is how much each restricted freedom recovers.
        * how?  - (mean accuracy of v - frozen) / (complete - frozen).
        """
        #region 1 - Compute the share
        # what? - Percentage.
        # how?  - Uses the mean accuracies of the 10 seeds.
        return 100 * (st.mean(acc[v]) - frozen) / (complete - frozen)
        #endregion

    P72 = {"completo": 21_233_664, "congelado_72": 0, "so_sigma": R["so_sigma"][0]["params_restritos"], "so_rotacao": pr.get("params_restritos_efetivos", pr["params_restritos"])}
    rows = "".join(f"<tr{' class=dest' if v == 'so_sigma' else ''}><td><code>{v}</code>, {LABEL[v]}{NOTE[v]}</td><td class='n'>{thousands(P72[v])}</td><td class='n'>{num(st.mean(acc[v]), 2)}% ± {num(st.stdev(acc[v]), 2)}</td>"
                   f"<td class='n'>{num(rec(v), 0)}%</td><td class='n'>{num(st.mean(r['acuracia_treino_por_epoca'][-1] for r in R[v]), 1)}%</td><td class='n'>{num(MED[v]['desloc_pct']['media'], 3)}%</td></tr>" for v in ORDER)
    tok["TAB_RESUMO"] = ("<div class='rolagem'><table><thead><tr><th>Variante</th><th class='n'>Parâmetros livres nas 72</th><th class='n'>Acurácia de teste (10 sementes)</th><th class='n'>Do ganho completo − congeladas</th><th class='n'>Acerto no treino</th><th class='n'>Deslocamento das 72</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
    dsc, tsc, psc = paired(acc["so_sigma"], acc["congelado_72"])
    drs, trs, prs = paired(acc["so_rotacao"], acc["so_sigma"])
    dcr, tcr, pcr = paired(acc["completo"], acc["so_rotacao"])
    tok["TXT_RESUMO"] = (f"Com as 72 matrizes congeladas, o resto do corpo e a cabeça chegam a <b>{num(frozen, 1)}%</b>; o fine-tuning completo chega a <b>{num(complete, 1)}%</b> ({num(complete - frozen, 1)} pontos de ganho atribuíveis às 72 matrizes). "
                         f"<b>Só σ</b> chega a {num(st.mean(acc['so_sigma']), 1)}%: recupera <b>{num(rec('so_sigma'), 0)}%</b> desse ganho com {P72['so_sigma']:,} parâmetros (0,13% dos 21,2 M); contra as congeladas, a diferença pareada por semente é {num(dsc, 1).replace('-', '−')} pontos (t = {num(tsc, 1)}, maior em {psc}/10 sementes). "
                         f"<b>Só rotação</b> chega a {num(st.mean(acc['so_rotacao']), 1)}%: recupera <b>{num(rec('so_rotacao'), 0)}%</b>; contra só σ, {num(drs, 1).replace('-', '−')} pontos (t = {num(trs, 1)}, maior em {prs}/10). O completo supera a rotação em {num(dcr, 1).replace('-', '−')} pontos (t = {num(tcr, 1)}).").replace(",000,", ".000.").replace("27,648", "27.648")
    #endregion

    #region 5 - Measures table
    # what? - What each variant did to the 72 matrices.
    # how?  - Values come from medidas_variantes.json.
    head = "<th>Variante</th><th class='n'>Deslocamento</th><th class='n'>|Δσ/σ| top-10</th><th class='n'>|Δσ/σ| todos</th><th class='n'>|Δσ/σ| máximo</th><th class='n'>Subespaço top-10</th><th class='n'>Subespaço top-50</th><th class='n'>Direções preservadas</th>"
    rows = "".join(f"<tr><td>{LABEL[v]}{NOTE[v]}</td><td class='n'>{num(MED[v]['desloc_pct']['media'], 3)}%</td><td class='n'>{num(MED[v]['sigma_top10_abs_pct']['media'], 3)}%</td><td class='n'>{num(MED[v]['sigma_todos_abs_pct']['media'], 3)}%</td>"
                   f"<td class='n'>{num(MED[v]['sigma_max_abs_pct']['media'], 2)}%</td><td class='n'>{num(MED[v]['sub10_graus']['media'], 2)}°</td><td class='n'>{num(MED[v]['sub50_graus']['media'], 2)}°</td><td class='n'>{num(MED[v]['direcoes_preservadas']['media'], 4)}</td></tr>" for v in ORDER)
    tok["TAB_MEDIDAS"] = "<div class='rolagem'><table><thead><tr>" + head + "</tr></thead><tbody>" + rows + "</tbody></table></div>"
    #endregion

    #region 6 - Numbers used by the findings
    # what? - The A_* tokens quoted in achados_etapa4.html.
    # how?  - Values come from the measures, the paired tests and the reports.
    tok.update({"A_CONG": num(frozen, 1), "A_COMP": num(complete, 1), "A_SIG": num(st.mean(acc["so_sigma"]), 1), "A_ROT": num(st.mean(acc["so_rotacao"]), 1),
                "A_SIG_REC": num(rec("so_sigma"), 0), "A_ROT_REC": num(rec("so_rotacao"), 0), "A_GANHO": num(complete - frozen, 1),
                "A_SIG_T": num(tsc, 1), "A_SIG_D": num(dsc, 1), "A_SIG_P": str(psc), "A_RS_D": num(drs, 1), "A_RS_T": num(trs, 1), "A_CR_D": num(dcr, 1), "A_CR_T": num(tcr, 1),
                "A_SIG_S10": num(MED["so_sigma"]["sigma_top10_abs_pct"]["media"], 2), "A_SIG_SALL": num(MED["so_sigma"]["sigma_todos_abs_pct"]["media"], 2), "A_SIG_SMAX": num(MED["so_sigma"]["sigma_max_abs_pct"]["media"], 1),
                "A_COMP_S10": num(MED["completo"]["sigma_top10_abs_pct"]["media"], 3), "A_SIG_DESL": num(MED["so_sigma"]["desloc_pct"]["media"], 2), "A_COMP_DESL": num(MED["completo"]["desloc_pct"]["media"], 2),
                "A_ROT_DESL": num(MED["so_rotacao"]["desloc_pct"]["media"], 2), "A_ROT_SUB10": num(MED["so_rotacao"]["sub10_graus"]["media"], 2), "A_COMP_SUB10": num(MED["completo"]["sub10_graus"]["media"], 2),
                "A_ROT_SALL": num(MED["so_rotacao"]["sigma_todos_abs_pct"]["media"], 4), "A_SIG_PRES": num(MED["so_sigma"]["direcoes_preservadas"]["media"], 4),
                "A_SIG_TR": num(st.mean(r["acuracia_treino_por_epoca"][-1] for r in R["so_sigma"]), 1), "A_ROT_TR": num(st.mean(r["acuracia_treino_por_epoca"][-1] for r in R["so_rotacao"]), 1),
                "A_COMP_TR": num(st.mean(r["acuracia_treino_por_epoca"][-1] for r in R["completo"]), 1), "A_CONG_TR": num(st.mean(r["acuracia_treino_por_epoca"][-1] for r in R["congelado_72"]), 1),
                "A_P_SIG": thousands(P72["so_sigma"]), "A_P_ROT": thousands(P72["so_rotacao"]), "A_LR_SIG": str(PAS['so_sigma']['lr']), "A_LR_ROT": str(PAS['so_rotacao']['lr']),
                "A_SIG_DP": num(st.stdev(acc["so_sigma"]), 2), "A_ROT_DP": num(st.stdev(acc["so_rotacao"]), 2), "A_SIG_SUB10": num(MED["so_sigma"]["sub10_graus"]["media"], 1),
                "A_SIG_D2": num(complete - st.mean(acc["so_sigma"]), 1)})
    tok["ACHADOS4"] = (C.STAGE4 / "achados_etapa4.html").read_text() if (C.STAGE4 / "achados_etapa4.html").exists() else "<p>(achados ainda não escritos)</p>"
    tok["PROBLEMAS4"] = (C.STAGE4 / "problemas_etapa4.html").read_text() if (C.STAGE4 / "problemas_etapa4.html").exists() else "<p>(ainda não escrito)</p>"
    return tok
    #endregion


def main():
    """
    * what? - Builds the stage 4 fragment and splices it into resultados.html, also writing the stage-only page.
    * why?  - Keeps the log book of the experiment regenerable from data, never edited by hand.
    * how?  - Load the data, build the tokens, fill the template, then insert the fragment, the navigation and the file list into the base page.
    """
    #region 1 - Load the stage 4 data
    # what? - Measures, chosen steps, environment and training reports.
    # how?  - R maps each variant to its 10 reports; 'completo' reuses the stage 3 reports.
    MED = load_json(C.STAGE4 / "medidas_variantes.json")
    PAS = load_json(C.STAGE4 / "passos_escolhidos.json")
    AMB = load_json(C.STAGE4 / "ambiente.json")
    R = {v: [load_json(C.MODELS / f"{v}_s{s}" / "relatorio.json") for s in C.SEEDS] for v in C.VARIANTS}
    R["completo"] = [load_json(C.full_model(s).parent / "relatorio.json") for s in C.SEEDS]
    #endregion

    #region 2 - Fill the stage 4 template
    # what? - The fragment with every token replaced.
    # how?  - Four passes, because some token values contain other tokens; leftovers abort.
    tok = build_tokens(MED, PAS, R)
    frag = (HERE / "etapa4_template.html").read_text()
    for _ in range(4):
        for k, v in tok.items():
            frag = frag.replace(f"[[{k}]]", v)
    missing = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", frag)))
    assert not missing, f"tokens without value: {missing}"
    #endregion

    #region 3 - Splice into the frozen base page
    # what? - Inserts the fragment, the navigation, the overview row, the title and the file list.
    # how?  - Each replace targets text of the frozen base; the asserts guard against a base that is not the expected one.
    base = (C.STAGE4 / "base/resultados_ate_etapa3.html").read_text()
    assert '<section class="sec" id="anexo">' in base and "ETAPA 4 (inserida" not in base
    html = base.replace('<section class="sec" id="anexo">', frag + '\n<section class="sec" id="anexo">', 1)
    html = html.replace('<li><a href="#anexo" class="e">Anexo</a></li>',
                        '<li><a href="#etapa4" class="e">Etapa 4</a></li><li><a href="#e4-justificativa">4.0 justificativa</a></li><li><a href="#e4-passo">4.1 passo</a></li><li><a href="#e4-treinos">4.2 treinos</a></li><li><a href="#e4-resultados">4.3 resultados</a></li><li><a href="#e4-achados">4.4 achados</a></li><li><a href="#e4-problemas">4.5 problemas</a></li><li><a href="#anexo" class="e">Anexo</a></li>', 1)
    html = html.replace('<td><code>bash doutorado/experimento1/etapa3/reproduzir_etapa3.sh</code></td></tr>',
                        '<td><code>bash doutorado/experimento1/etapa3/reproduzir_etapa3.sh</code></td></tr>'
                        f'<tr><td><b>4</b></td><td>fine-tuning <b>restrito</b>: só σ, só rotação, 72 congeladas (mesmo domínio e protocolo da etapa 3)</td><td>A etapa 3 só avaliou a parte σ que o treino livre depositou; a etapa 4 mede o que cada liberdade sozinha consegue: só σ recupera {tok["A_SIG_REC"]}% do ganho, só rotação {tok["A_ROT_REC"]}%.</td><td><code>bash doutorado/experimento1/etapa4/reproduzir_etapa4.sh</code></td></tr>', 1)
    html = html.replace("todo o conteúdo das três etapas em sequência", "todo o conteúdo das quatro etapas em sequência", 1)
    html = html.replace("<title>Experimento 1 — resultados das etapas 1, 2 e 3</title>", "<title>Experimento 1 — resultados das etapas 1 a 4</title>", 1)
    html = html.replace("  <p>Esta página (<code>diario.html</code>) é gerada por",
                        "  <h4>Etapa 4</h4><pre><code>doutorado/experimento1/etapa4/\n  reproduzir_etapa4.sh  config_etapa4.py  treinar_etapa4.py  escolher_passo.py  medir_variantes.py\n  construir_resultados_etapa4.py  etapa4_template.html  achados_etapa4.html  problemas_etapa4.html\n  base/resultados_ate_etapa3.html (copia intocada)  logs/  passos_escolhidos.json  medidas_variantes.json  ambiente.json\nmodels/etapa4/  piloto_so_sigma_lr*/  piloto_so_rotacao_lr*/  congelado_72_s0..9/  so_sigma_s0..9/  so_rotacao_s0..9/</code></pre>\n"
                        f"  <p>Ambiente da etapa 4: Python {AMB['python']}, torch {AMB['pacotes']['torch']}, transformers {AMB['pacotes']['transformers']}, {AMB['gpu']}, commit <code>{AMB['git_commit'][:10]}</code>.</p>\n"
                        "  <p>Esta página (<code>diario.html</code>) é gerada por", 1)
    missing_page = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", html)))
    assert not missing_page, missing_page
    #endregion

    #region 4 - Write the pages
    # what? - Saves resultados.html and the stage-only page.
    # how?  - The stage-only page is the fragment before splicing.
    (C.EXPERIMENT / "resultados.html").write_text(html)
    (C.STAGE4 / "resultados_etapa4_somente.html").write_text(frag)
    print(f"resultados.html: {len(html.encode())/1024:.1f} KB | stage 4 inserted ({len(frag.encode())/1024:.1f} KB)")
    #endregion


if __name__ == "__main__":
    main()
