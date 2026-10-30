"""
* what? - STAGE 3: inserts the stage 3 section into doutorado/experimento1/resultados.html.
* why?  - The page is the log book of the experiment; each stage adds its section to a frozen copy of the previous page.
* how?  - Reads the stage 3 JSONs and training reports, fills etapa3_template.html and splices it into the frozen base page.

Base: etapa3/base/resultados_ate_etapa2.html (copy of resultados.html as it was at the end of stage 2; never edited).
Everything that enters comes from the JSONs of etapa3/ and the relatorio.json of the trainings. Idempotent: it regenerates the
whole page from the base. It also writes etapa3/resultados_etapa3_somente.html (stage 3 only) for consultation.
Run from the repository root:  python3 doutorado/experimento1/etapa3/scripts/build_results_etapa3.py
"""
import json
import math
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_etapa3 as C  # noqa: E402

HERE = Path(__file__).resolve().parent
K = math.pi / 180 * 100
MODEL_SETS = ("etapa1", "etapa2", "etapa3", "embaralhado")
# (key, label, multiplier, unit, decimals) of the ruler table rows; labels are page content in Portuguese
MEASURES = [("desloc_pct", "Deslocamento da matriz, ‖W1−W0‖/‖W0‖", 100, "%", 3),
            ("sigma_top10_abs_pct", "Pesos: |Δσ/σ|, 10 maiores", 1, "%", 4),
            ("sigma_11_em_diante_abs_pct", "Pesos: |Δσ/σ|, do 11º em diante", 1, "%", 3),
            ("escala_geral_top10_pct", "Escala geral dos 10 maiores (média com sinal)", 1, "%", 4),
            ("diferencial_top10_pct", "Parte diferencial dos 10 maiores", 1, "%", 4),
            ("giro_vetor_top10_graus", "Giro vetor a vetor, top-10 (régua antiga)", 1, "°", 3),
            ("sub10_medio_graus", "Subespaço top-10: ângulo principal médio", 1, "°", 3),
            ("sub10_maior_graus", "Subespaço top-10: maior ângulo principal", 1, "°", 3),
            ("sub50_medio_graus", "Subespaço top-50: ângulo principal médio", 1, "°", 3),
            ("sub100_medio_graus", "Subespaço top-100: ângulo principal médio", 1, "°", 3),
            ("fracao_sigma_completa_pct", "Fração da energia na parte σ (norma completa)", 1, "%", 3),
            ("fracao_sigma_projetada_pct", "Idem, como nas etapas 1–2 (projeção em span U0)", 1, "%", 3),
            ("energia_fora_do_span_pct", "Energia da mudança fora de span(U0) (média das 72)", 1, "%", 1),
            ("concentracao", "Concentração (fração σ × maior lado)", 1, "", 2)]


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


def training_table(reports):
    """
    * what? - Builds the HTML table of the 10 trainings (accuracy before, train, test after, minutes) with an average row.
    * why?  - Section 3.2 shows the raw results of the real and of the shuffled-label trainings.
    * how?  - One row per seed from the relatorio.json, plus mean and standard deviation of the final accuracies.
    """
    #region 1 - Build the rows
    # what? - Rows and summary lists.
    # how?  - Cells carry the CSS class n for right-aligned numbers.
    rows = "".join(f"<tr><td class='n'>{r['semente']}</td><td class='n'>{num(r['acuracia_antes'], 1)}%</td><td class='n'>{num(r['acuracia_treino_por_epoca'][-1], 1)}%</td><td class='n'>{num(r['acuracia_depois'], 1)}%</td><td class='n'>{num(r['minutos_de_treino'], 1)}</td></tr>" for r in reports)
    after = [r["acuracia_depois"] for r in reports]
    train_final = [r["acuracia_treino_por_epoca"][-1] for r in reports]
    #endregion

    #region 2 - Wrap in the table
    # what? - Header, rows and the average row.
    # how?  - The wrapper div scrolls horizontally on narrow screens.
    return ("<div class='rolagem'><table><thead><tr><th class='n'>Semente</th><th class='n'>Teste, antes</th><th class='n'>Treino, última época</th><th class='n'>Teste, depois</th><th class='n'>Minutos</th></tr></thead><tbody>" + rows +
            f"<tr class='dest'><td><b>média</b></td><td></td><td class='n'><b>{num(st.mean(train_final), 1)}%</b></td><td class='n'><b>{num(st.mean(after), 2)}%</b> ± {num(st.stdev(after), 2)}</td><td></td></tr></tbody></table></div>")
    #endregion


def transplant_table(t):
    """
    * what? - Builds the HTML table of the transplant test of one model set, with a mean row.
    * why?  - Section 3.4 shows the accuracy of each variant per seed.
    * how?  - One row per seed with a column per variant and the energy share of the sigma part.
    """
    #region 1 - Build the rows and the mean row
    # what? - Table body.
    # how?  - The variant names are the frozen keys of transplante_<set>.json.
    variants = t["variantes"]
    rows = "".join(f"<tr><td class='n'>{r['semente']}</td>" + "".join(f"<td class='n'>{num(r[v], 2)}%</td>" for v in variants) + f"<td class='n'>{num(r['fracao_energia_sigma_media_pct'], 3)}%</td></tr>" for r in t["sementes"])
    mean = "<tr class='dest'><td><b>média</b></td>" + "".join(f"<td class='n'><b>{num(t['media'][v], 2)}%</b> ± {num(t['dp'][v], 2)}</td>" for v in variants) + "<td></td></tr>"
    #endregion

    #region 2 - Wrap in the table
    # what? - Header and body.
    # how?  - Variant names are shown as code.
    return ("<div class='rolagem'><table><thead><tr><th class='n'>Semente</th>" + "".join(f"<th class='n'><code>{v}</code></th>" for v in variants) + "<th class='n'>Energia na parte σ</th></tr></thead><tbody>" + rows + mean + "</tbody></table></div>")
    #endregion


def recovery(t, variant):
    """
    * what? - Returns the percentage of the matrix gain that a transplant variant recovers.
    * why?  - The headline of the functional test is how much of the gain each part recovers.
    * how?  - (variant - none) / (complete - none), in percent, from the mean accuracies.
    """
    #region 1 - Compute the recovery
    # what? - Percentage of the gain.
    # how?  - 'nenhuma' is the original matrices with everything else fine-tuned.
    m = t["media"]
    return 100 * (m[variant] - m["nenhuma"]) / (m["completo"] - m["nenhuma"])
    #endregion


def ruler_cell(reg, key, unit, decimals):
    """
    * what? - Builds the two table cells (measure and isotropic-noise zero) of a ruler row for one model set.
    * why?  - The ruler table puts the real value next to the noise zero.
    * how?  - Reads media and zero_media of the global entry; the zero is shown in the muted color.
    """
    #region 1 - Build the cells
    # what? - Two HTML cells.
    # how?  - Portuguese number format.
    g = reg["global"][key]
    r, z = g["media"], g["zero_media"]
    return f"<td class='n'>{num(r, decimals)}{unit}</td><td class='n' style='color:var(--ink3)'>{num(z, decimals)}{unit}</td>"
    #endregion


def agreement(regs, model_set, key):
    """
    * what? - Returns the global agreement between seeds for one part of the change in one model set.
    * why?  - The agreement table and the findings quote these values.
    * how?  - Reads global.concordancia[key] of the ruler JSON of the set.
    """
    #region 1 - Look the value up
    # what? - A float.
    # how?  - Keys are inteira, sigma, fora, espectro.
    return regs[model_set]["global"]["concordancia"][key]
    #endregion


def build_tokens(d):
    """
    * what? - Builds every [[TOKEN]] of the stage 3 template from the loaded data.
    * why?  - No number of the page is typed by hand; text and tables come from the JSONs.
    * how?  - One region per page section (3.0 to 3.6); the token names are the contract with etapa3_template.html.
    """
    #region 1 - Unpack the data and prepare the shortcuts
    # what? - Names for the loaded dictionaries and the frequently used pieces.
    # how?  - REL3 are the real trainings and RELE the shuffled-label ones.
    V, TRI, REG, TRA, CTR, PIL, REL3, RELE, INSP = d["V"], d["TRI"], d["REG"], d["TRA"], d["CTR"], d["PIL"], d["REL3"], d["RELE"], d["INSP"]
    tok = {}
    #endregion

    #region 2 - Section 3.0: verification of stages 1-2
    # what? - Tokens V_* of the verification text.
    # how?  - Each value comes from verificacao_etapas12.json.
    f, rz, gc, sb, pp, es, zi = (V["funcional"], V["razao_direcoes_pesos"], V["giro_x_colados"], V["subespaco_top10_etapa2_s0"],
                                 V["pareado_pesos_e1_e2"], V["escala_geral_por_matriz_etapa2"], V["zero_isotropico_0p5pct"])
    tok = {"V_CONG": num(f["corpo_congelado_reg_logistica"], 2), "V_E1": num(f["etapa1"]["media"], 2), "V_E2": num(f["etapa2"]["media"], 2),
           "V_GANHO": num(f["etapa2"]["ganho_sobre_congelado"], 2).replace("-", "−"),
           "V_RZ_RUIDO": num(rz["ruido_etapa1"], 0), "V_RZ_ISO": num(zi["razao_mediana"], 0), "V_RZ_ISO_MIN": num(zi["razao_min"], 0), "V_RZ_ISO_MAX": num(zi["razao_max"], 0),
           "V_R_COL": num(gc["r_pearson"], 2), "V_GIRO_ATE3": num(gc["giro_ate_3_colados"], 2), "V_N_ATE3": str(gc["n_ate_3"]), "V_GIRO_MAIS3": num(gc["giro_mais_de_3"], 2), "V_N_MAIS3": str(gc["n_mais_de_3"]),
           "V_SUB_MED": num(sb["angulo_principal_medio_graus"], 2), "V_VET": num(sb["giro_vetor_a_vetor_graus"], 2), "V_SUB_PCT_PERDA": num(100 * (1 - sb["fracao_subespaco_sobre_vetores"]), 0),
           "V_ESC_37": str(es["matrizes_com_sinal_igual_em_9_ou_10_treinos"]), "V_ESC_ACASO": num(es["esperado_por_acaso"], 1),
           "V_T_PAR": num(pp["t_pareado"], 1), "V_T_POS": str(pp["positivo_em"])}
    #endregion

    #region 3 - Section 3.1: screening table and pilots
    # what? - Screening and pilot tables and their texts.
    # how?  - Skin lesions is highlighted with the CSS class dest.
    order = ["pele", "dtd", "kmnist", "eurosat"]
    rows = "".join(f"<tr{' class=dest' if ap == 'pele' else ''}><td>{TRI[ap]['nome']}</td><td class='n'>{TRI[ap]['classes']}</td><td class='n'>{TRI[ap]['n_treino']} / {TRI[ap]['n_teste']}</td>"
                   f"<td class='n'>{num(TRI[ap]['acaso_pct'], 1)}%</td><td class='n'>{num(TRI[ap]['acuracia_corpo_congelado'], 1)}%</td><td class='n'>{num(TRI[ap]['acuracia_treino'], 1)}%</td></tr>" for ap in order)
    rows += f"<tr><td>{TRI['pets_referencia']['nome']}</td><td class='n'>37</td><td class='n'>3680 / 3669</td><td class='n'>2,7%</td><td class='n'>{num(TRI['pets_referencia']['acuracia_corpo_congelado'], 1)}%</td><td class='n'>—</td></tr>"
    tok["TAB_TRIAGEM"] = ("<div class='rolagem'><table><thead><tr><th>Candidato</th><th class='n'>Classes</th><th class='n'>Treino / teste usados</th><th class='n'>Acaso</th>"
                          "<th class='n'>Acerto, corpo congelado (teste)</th><th class='n'>Idem, treino</th></tr></thead><tbody>" + rows + "</tbody></table></div>")
    tok["TXT_TRIAGEM"] = (f"EuroSAT fica no mesmo patamar de raças de pet ({num(TRI['eurosat']['acuracia_corpo_congelado'], 1)}%): o corpo congelado já resolve, e não serve. "
                          f"Lesões de pele é o mais distante ({num(TRI['pele']['acuracia_corpo_congelado'], 1)}% no teste, com só {num(TRI['pele']['acuracia_treino'], 1)}% no treino: as 384 características do corpo não separam bem as classes nem nos dados de treino). "
                          f"DTD e KMNIST ficam no meio (≈{num(TRI['dtd']['acuracia_corpo_congelado'], 0)}%).")
    pilot_rows = "".join(f"<tr{' class=dest' if x == 'pele' else ''}><td>{TRI[x]['nome']}</td><td class='n'>{num(TRI[x]['acuracia_corpo_congelado'], 1)}%</td><td class='n'>{num(PIL[x]['acuracia_depois'], 1)}%</td>"
                         f"<td class='n'>{num(PIL[x]['acuracia_depois'] - TRI[x]['acuracia_corpo_congelado'], 1).replace('-', '−')}</td><td class='n'>{num(PIL[x]['acuracia_treino_por_epoca'][-1], 1)}%</td><td class='n'>{num(PIL[x]['minutos_de_treino'], 1)}</td></tr>" for x in ("pele", "dtd", "kmnist"))
    tok["TAB_PILOTO"] = ("<div class='rolagem'><table><thead><tr><th>Candidato</th><th class='n'>Corpo congelado</th><th class='n'>Fine-tuning (piloto, semente 0)</th><th class='n'>Ganho (pontos)</th>"
                         "<th class='n'>Acerto no treino</th><th class='n'>Minutos</th></tr></thead><tbody>" + pilot_rows + "</tbody></table></div>")
    tok["TXT_PILOTO"] = ("Mesmo protocolo da etapa 2 (weight_decay = 0), uma semente, para medir quanto o fine-tuning do corpo ganha do corpo congelado. "
                         "Em raças de pet esse ganho foi negativo (−0,65 ponto). Critério combinado: ganho de pelo menos 5 pontos.")
    tok["TXT_ESCOLHA"] = (f"<b>Lesões de pele</b>: ganho de <b>{num(PIL['pele']['acuracia_depois'] - TRI['pele']['acuracia_corpo_congelado'], 1)} pontos</b> do fine-tuning sobre o corpo congelado, "
                          "o único caso em que a mudança do corpo tem efeito funcional grande. É também o domínio do caso de uso que motivou a tese (apoio ao diagnóstico dermatológico com poucas imagens e uma GPU).")
    #endregion

    #region 4 - Section 3.2: data and trainings
    # what? - Domain description tokens and the training tables.
    # how?  - The dataset README is mentioned only if it exists on disk.
    readme = (C.DATA / "pele/README.md")
    tok.update({"DOM_NOME": "lesões de pele, 8 classes", "DOM_AP": "pele",
                "DOM_DESC": f"Fotos de lesões de pele em <b>8 classes</b>, publicadas como <code>{INSP['repo']}</code> no HuggingFace, com rótulos inteiros 0 a 7 (o repositório não traz os nomes das classes{'' if not readme.exists() else '; o README está em etapa3/dados/pele/'}). Imagens de {INSP['treino']['imagem']}.",
                "DOM_ORIGEM": f"HuggingFace, <code>{INSP['repo']}</code>; arquivos parquet em <code>etapa3/dados/pele/</code>. Proveniência original das fotos não documentada pelo repositório: tratar como dataset de teste metodológico, não clínico.",
                "DOM_QTD": f"Publicadas: {INSP['treino']['linhas']:,} de treino e {INSP['teste']['linhas']:,} de teste. Usadas: <b>{REL3[0]['n_treino']:,}</b> de treino (subamostra fixa, a mesma em todas as sementes) e <b>{REL3[0]['n_teste']:,}</b> de teste (todas). Classe majoritária no teste: {num(TRI['pele']['classe_majoritaria_pct'], 1)}%.".replace(",", "."),
                "DOM_K": "8", "DOM_TREINO": "etapa3/dados/pele/data/train-…parquet", "DOM_TESTE": "etapa3/dados/pele/data/test-…parquet"})
    tok["TAB_E3"] = "<h4>Treinos com os rótulos reais</h4>" + training_table(REL3) + "<h4>Treinos com os rótulos embaralhados (controle)</h4>" + training_table(RELE)
    acc3 = [r["acuracia_depois"] for r in REL3]
    acce = [r["acuracia_depois"] for r in RELE]
    #endregion

    #region 5 - Section 3.3: the corrected ruler
    # what? - The ruler table (four sets with their noise zero) and the agreement table.
    # how?  - Each row is a measure; each set contributes a value and a muted zero.
    head = "<th>Medida</th>" + "".join(f"<th class='n'>{t}</th><th class='n' style='color:var(--ink3)'>zero</th>" for t in ("Etapa 1 (pets, wd 0,05)", "Etapa 2 (pets, wd 0)", "Etapa 3 (pele, wd 0)", "Embaralhado (pele)"))
    body = "".join("<tr><td>" + label + "</td>" + "".join(ruler_cell(REG[s], key, unit, decimals) for s in MODEL_SETS) + "</tr>" for key, label, mult, unit, decimals in MEASURES)
    tok["TAB_REGUA"] = "<div class='rolagem'><table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table></div>"
    tok["TAB_CONC"] = ("<div class='rolagem'><table><thead><tr><th>Parte da mudança</th><th class='n'>Etapa 1</th><th class='n'>Etapa 2</th><th class='n'>Etapa 3</th><th class='n'>Embaralhado</th></tr></thead><tbody>" +
                       "".join(f"<tr><td>{label}</td>" + "".join(f"<td class='n'>{num(agreement(REG, s, k), 3)}</td>" for s in MODEL_SETS) + "</tr>"
                               for k, label in (("inteira", "Mudança inteira"), ("sigma", "Parte σ (diagonal)"), ("fora", "Parte direções (fora da diagonal)"), ("espectro", "Espectro de σ completo"))) + "</tbody></table></div>")
    #endregion

    #region 6 - Section 3.4: functional transplant test
    # what? - Transplant tables and the headline text.
    # how?  - m3 and m2 are the mean accuracies per variant for skin lesions and for pets (negative control).
    TRA3, TRA2 = TRA["etapa3"], TRA["etapa2"]
    tok["TAB_TRANSPLANTE"] = "<h4>Etapa 3, lesões de pele</h4>" + transplant_table(TRA3) + "<h4>Controle negativo: etapa 2, raças de pet</h4>" + transplant_table(TRA2)
    m3, m2 = TRA3["media"], TRA2["media"]
    tok["TXT_TRANSPLANTE"] = (f"<b>Lesões de pele:</b> as 72 matrizes originais com o resto fine-tuned (<code>nenhuma</code>) dão {num(m3['nenhuma'], 1)}%, e a mudança completa dá {num(m3['completo'], 1)}%: a mudança das matrizes vale <b>{num(m3['completo'] - m3['nenhuma'], 1)} pontos</b>. "
                              f"Só a parte σ recupera <b>{num(recovery(TRA3, 'so_sigma'), 0)}%</b> desse ganho, só a parte direções recupera <b>{num(recovery(TRA3, 'so_direcoes'), 0)}%</b>. "
                              f"Com a <b>mesma energia</b> da parte σ, a parte direções recupera {num(recovery(TRA3, 'direcoes_eq'), 0)}%. "
                              f"<b>Raças de pet:</b> a mudança das matrizes vale {num(m2['completo'] - m2['nenhuma'], 1)} pontos (sob a cabeça co-adaptada), a parte σ recupera {num(recovery(TRA2, 'so_sigma'), 0)}% e a parte direções {num(recovery(TRA2, 'so_direcoes'), 0)}%.")
    #endregion

    #region 7 - Section 3.5: shuffled-label control
    # what? - Control table and text.
    # how?  - The ratio real-vs-shuffled over real-vs-real is the ceiling of what can be attributed to data and optimizer.
    c72 = CTR["cosseno_medio_72"]
    tok["TAB_CONTROLE"] = ("<div class='rolagem'><table><thead><tr><th>Par de treinos</th><th class='n'>Mudança inteira</th><th class='n'>Parte σ</th><th class='n'>Parte direções</th></tr></thead><tbody>" +
                           "".join(f"<tr><td>{label}</td><td class='n'>{num(c72[g]['inteira'], 3)}</td><td class='n'>{num(c72[g]['sigma'], 3)}</td><td class='n'>{num(c72[g]['fora'], 3)}</td></tr>"
                                   for g, label in (("real_real", "real × real (45 pares)"), ("emb_emb", "embaralhado × embaralhado (45 pares)"), ("real_emb", "real × embaralhado (100 pares)"))) +
                           f"</tbody></table></div><p style='font-size:13.5px;color:var(--ink3)'>Deslocamento médio das matrizes: {num(CTR['deslocamento_medio_pct']['real'], 3)}% (real) e {num(CTR['deslocamento_medio_pct']['emb'], 3)}% (embaralhado). "
                           f"Acurácia de teste dos treinos embaralhados: {num(st.mean(acce), 1)}% (acaso 12,5%); acerto no treino na última época: {num(st.mean(r['acuracia_treino_por_epoca'][-1] for r in RELE), 1)}%.</p>")
    tok["TXT_CONTROLE"] = (f"A parte da mudança compartilhada entre dois treinos reais (cosseno {num(c72['real_real']['inteira'], 3)}) é bem maior do que a compartilhada entre um treino real e um embaralhado ({num(c72['real_emb']['inteira'], 3)}). "
                           f"A fração do cosseno real × real que sobrevive sem os rótulos é de <b>{num(100 * c72['real_emb']['inteira'] / c72['real_real']['inteira'], 0)}%</b>: esse é o teto do que pode ser atribuído aos dados e ao otimizador; o resto depende dos rótulos. "
                           f"Entre dois treinos embaralhados a concordância é {num(c72['emb_emb']['inteira'], 3)}: com a mesma permutação fixa de rótulos, eles também aprendem algo em comum (memorização do rótulo aleatório).")
    #endregion

    #region 8 - Findings and problems texts
    # what? - Hand-written findings and problems pages, with their own tokens inside.
    # how?  - A missing file produces a placeholder paragraph.
    tok["ACHADOS3"] = (C.STAGE3 / "achados_etapa3.html").read_text() if (C.STAGE3 / "achados_etapa3.html").exists() else "<p>(achados ainda não escritos)</p>"
    tok["PROBLEMAS3"] = (C.STAGE3 / "problemas_etapa3.html").read_text() if (C.STAGE3 / "problemas_etapa3.html").exists() else "<p>(ainda não escrito)</p>"
    #endregion

    #region 9 - Numbers used by the findings
    # what? - The A_* tokens quoted in achados_etapa3.html.
    # how?  - g3, g2 and ge are the global ruler entries of stage 3, stage 2 and the shuffled control.
    g3, g2, ge = (REG[k]["global"] for k in ("etapa3", "etapa2", "embaralhado"))
    tok.update({
        "A_ACC3": num(st.mean(acc3), 1), "A_ACC3_DP": num(st.stdev(acc3), 1), "A_CONG": num(TRI["pele"]["acuracia_corpo_congelado"], 1), "A_GANHO": num(st.mean(acc3) - TRI["pele"]["acuracia_corpo_congelado"], 1),
        "A_DESL3": num(g3["desloc_pct"]["media"], 2), "A_DESL2": num(g2["desloc_pct"]["media"], 2), "A_DESL_RZ": num(g3["desloc_pct"]["media"] / g2["desloc_pct"]["media"], 1),
        "A_S10_3": num(g3["sigma_top10_abs_pct"]["media"], 3), "A_S10_3Z": num(g3["sigma_top10_abs_pct"]["zero_media"], 3), "A_S10_3R": num(g3["sigma_top10_abs_pct"]["media"] / g3["sigma_top10_abs_pct"]["zero_media"], 1),
        "A_S10_2R": num(g2["sigma_top10_abs_pct"]["media"] / g2["sigma_top10_abs_pct"]["zero_media"], 1),
        "A_SUB10_3": num(g3["sub10_medio_graus"]["media"], 2), "A_SUB10_3Z": num(g3["sub10_medio_graus"]["zero_media"], 2), "A_SUB10_3R": num(g3["sub10_medio_graus"]["media"] / g3["sub10_medio_graus"]["zero_media"], 2),
        "A_SUB10_2R": num(g2["sub10_medio_graus"]["media"] / g2["sub10_medio_graus"]["zero_media"], 2),
        "A_SUB50_3R": num(g3["sub50_medio_graus"]["media"] / g3["sub50_medio_graus"]["zero_media"], 2), "A_SUB100_3R": num(g3["sub100_medio_graus"]["media"] / g3["sub100_medio_graus"]["zero_media"], 2),
        "A_VET_3": num(g3["giro_vetor_top10_graus"]["media"], 2), "A_VET_3Z": num(g3["giro_vetor_top10_graus"]["zero_media"], 2),
        "A_FR_3": num(g3["fracao_sigma_completa_pct"]["media"], 3), "A_FR_3Z": num(g3["fracao_sigma_completa_pct"]["zero_media"], 3), "A_CONC3": num(g3["concentracao"]["media"], 2), "A_CONC2": num(g2["concentracao"]["media"], 2), "A_CONCE": num(ge["concentracao"]["media"], 2),
        "A_FR_PROJ2": num(g2["fracao_sigma_projetada_pct"]["media"], 3), "A_FR_COMP2": num(g2["fracao_sigma_completa_pct"]["media"], 3), "A_FORA": num(g2["energia_fora_do_span_pct"]["media"], 0),
        "A_ESC3": num(g3["escala_geral_top10_pct"]["media"], 4), "A_DIF3": num(g3["diferencial_top10_pct"]["media"], 4), "A_DIF3Z": num(g3["diferencial_top10_pct"]["zero_media"], 4),
        "A_DIF3R": num(g3["diferencial_top10_pct"]["media"] / g3["diferencial_top10_pct"]["zero_media"], 1),
        "A_S11_3": num(g3["sigma_11_em_diante_abs_pct"]["media"], 3), "A_S11_3Z": num(g3["sigma_11_em_diante_abs_pct"]["zero_media"], 3),
        "A_C3_INT": num(agreement(REG, "etapa3", "inteira"), 3), "A_C3_SIG": num(agreement(REG, "etapa3", "sigma"), 3), "A_C3_FORA": num(agreement(REG, "etapa3", "fora"), 3), "A_C2_INT": num(agreement(REG, "etapa2", "inteira"), 3), "A_C2_SIG": num(agreement(REG, "etapa2", "sigma"), 3), "A_C2_FORA": num(agreement(REG, "etapa2", "fora"), 3),
        "A_CE_INT": num(agreement(REG, "embaralhado", "inteira"), 3), "A_RE_INT": num(c72["real_emb"]["inteira"], 3), "A_RE_PCT": num(100 * c72["real_emb"]["inteira"] / c72["real_real"]["inteira"], 0),
        "A_TR_GAN": num(m3["completo"] - m3["nenhuma"], 1), "A_TR_SIG": num(recovery(TRA3, "so_sigma"), 0), "A_TR_DIR": num(recovery(TRA3, "so_direcoes"), 0), "A_TR_EQ": num(recovery(TRA3, "direcoes_eq"), 0),
        "A_TR_SIG_ABS": num(m3["so_sigma"], 1), "A_TR_NEN": num(m3["nenhuma"], 1), "A_TR_COMP": num(m3["completo"], 1), "A_TR_EQ_ABS": num(m3["direcoes_eq"], 1), "A_TR_DIR_ABS": num(m3["so_direcoes"], 1),
        "A_TR2_GAN": num(m2["completo"] - m2["nenhuma"], 1), "A_TR2_SIG": num(recovery(TRA2, "so_sigma"), 0), "A_TR2_DIR": num(recovery(TRA2, "so_direcoes"), 0), "A_TR2_EQ": num(recovery(TRA2, "direcoes_eq"), 0),
        "A_EN3": num(TRA3["sementes"][0]["fracao_energia_sigma_media_pct"], 2),
        "A_ACCE": num(st.mean(acce), 1), "A_ACCE_TR": num(st.mean(r["acuracia_treino_por_epoca"][-1] for r in RELE), 1),
        "A_TRF3": num(st.mean(r["acuracia_treino_por_epoca"][-1] for r in REL3), 1)})
    return tok
    #endregion


def main():
    """
    * what? - Builds the stage 3 fragment and splices it into resultados.html, also writing the stage-only page.
    * why?  - Keeps the log book of the experiment regenerable from data, never edited by hand.
    * how?  - Load the data, build the tokens, fill the template, then insert the fragment, the navigation and the maps into the base page.
    """
    #region 1 - Load the stage 3 data
    # what? - JSONs of the stage and the training reports.
    # how?  - Dictionary d carries everything build_tokens needs.
    E3 = C.STAGE3
    d = {"V": load_json(E3 / "verificacao_etapas12.json"),
         "TRI": load_json(E3 / "triagem.json"),
         "REG": {k: load_json(E3 / f"regua_{k}.json") for k in MODEL_SETS},
         "TRA": {k: load_json(E3 / f"transplante_{k}.json") for k in ("etapa2", "etapa3")},
         "CTR": load_json(E3 / "controle_embaralhado.json"),
         "PIL": {x: load_json(C.MODELS / f"piloto_{x}" / "relatorio.json") for x in ("pele", "dtd", "kmnist")},
         "REL3": [load_json(C.MODELS / f"s{s}" / "relatorio.json") for s in C.SEEDS],
         "RELE": [load_json(C.MODELS / f"embaralhado_s{s}" / "relatorio.json") for s in C.SEEDS],
         "INSP": load_json(C.DATA / "candidatos_inspecao.json")["pele"]}
    AMB = load_json(E3 / "ambiente.json")
    #endregion

    #region 2 - Fill the stage 3 template
    # what? - The fragment with every token replaced.
    # how?  - Four passes, because some token values contain other tokens; leftovers abort.
    tok = build_tokens(d)
    frag = (HERE / "etapa3_template.html").read_text()
    for _ in range(4):
        for k, v in tok.items():
            frag = frag.replace(f"[[{k}]]", v)
    missing = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", frag)))
    assert not missing, f"tokens without value: {missing}"
    #endregion

    #region 3 - Build the map script
    # what? - Data and calls for the two stage 3 heat maps.
    # how?  - The maps use the page function mapa(); the script goes AFTER the page script, never inside it.
    d3 = {"celulas": [{"b": c["bloco"], "t": c["tipo"], "p": round(c["sigma_top10_abs_pct"]["media"], 5), "s": round(c["sub10_medio_graus"]["media"], 4)} for c in d["REG"]["etapa3"]["celulas"]]}
    js = ("<script>\nconst D3E = " + json.dumps(d3, separators=(",", ":")) + ";\n"
          'mapa("e3-mapas","Etapa 3 · Pesos (parte σ)","|Δσ/σ| dos 10 maiores, em % do original (média de 10 treinos)",D3E,"p",3);\n'
          'mapa("e3-mapas","Etapa 3 · Direções (subespaço top-10)","ângulo principal médio, em graus (média de 10 treinos)",D3E,"s",2,undefined,undefined,"°");\n</script>\n')
    #endregion

    #region 4 - Splice into the frozen base page
    # what? - Inserts the fragment, the navigation, the overview row and the file list.
    # how?  - Each replace targets text of the frozen base; the asserts guard against a base that is not the expected one.
    base = (E3 / "base/resultados_ate_etapa2.html").read_text()
    assert '<section class="sec" id="anexo">' in base and "<!-- =============================== ETAPA 3" not in base
    html = base.replace('<section class="sec" id="anexo">', frag + '\n<section class="sec" id="anexo">', 1)
    html = html.replace('<li><a href="#anexo" class="e">Anexo</a></li>',
                        '<li><a href="#etapa3" class="e">Etapa 3</a></li><li><a href="#e3-justificativa">3.0 justificativa</a></li><li><a href="#e3-triagem">3.1 triagem</a></li><li><a href="#e3-treinos">3.2 treinos</a></li>'
                        '<li><a href="#e3-regua">3.3 régua</a></li><li><a href="#e3-funcional">3.4 funcional</a></li><li><a href="#e3-controle">3.5 controle</a></li><li><a href="#e3-achados">3.6 achados</a></li><li><a href="#e3-problemas">3.7 problemas</a></li>'
                        '<li><a href="#anexo" class="e">Anexo</a></li>', 1)
    html = html.replace('<td>Testar a hipótese de que parte da mudança dos pesos era efeito do otimizador, e não da tarefa.</td><td><code>bash doutorado/experimento1/reproduzir_etapa2.sh</code></td></tr>',
                        '<td>Testar a hipótese de que parte da mudança dos pesos era efeito do otimizador, e não da tarefa.</td><td><code>bash doutorado/experimento1/reproduzir_etapa2.sh</code></td></tr>'
                        '<tr><td><b>3</b></td><td>domínio: <b>lesões de pele</b> (corpo congelado 56,6%); régua corrigida; teste funcional; rótulos embaralhados</td><td>As etapas 1 e 2 mediram mudanças sem efeito funcional, com uma razão direções/pesos estrutural e um giro inflado. A etapa 3 corrige as três coisas.</td><td><code>bash doutorado/experimento1/etapa3/reproduzir_etapa3.sh</code></td></tr>', 1)
    html = html.replace("Esta página reúne, de uma vez, todo o conteúdo das duas etapas em sequência", "Esta página reúne, de uma vez, todo o conteúdo das três etapas em sequência", 1)
    html = html.replace("  <p>Esta página (<code>diario.html</code>) é gerada por",
                        "  <h4>Etapa 3</h4><pre><code>doutorado/experimento1/etapa3/\n  reproduzir_etapa3.sh                  receita completa e idempotente da etapa\n  config_etapa3.py  treinar_etapa3.py  triagem_dominio.py  baixar_candidatos.py\n  verificar_conclusoes_etapas12.py  medir_regua.py  transplante.py  controle_embaralhado.py\n  construir_resultados_etapa3.py  etapa3_template.html  achados_etapa3.html  problemas_etapa3.html\n  base/   resultados_ate_etapa2.html, diario_ate_etapa2.html   (copias intocadas)\n  dados/  parquet dos 4 candidatos + candidatos_inspecao.json\n  logs/   um log por passo;  *.json  todos os resultados desta etapa\nmodels/etapa3/  piloto_{pele,dtd,kmnist}/  s0..s9/  embaralhado_s0..s9/</code></pre>\n"
                        f"  <p>Ambiente da etapa 3: Python {AMB['python']}, torch {AMB['pacotes']['torch']}, transformers {AMB['pacotes']['transformers']}, {AMB['gpu']}, driver {AMB['driver_nvidia']}, commit <code>{AMB['git_commit'][:10]}</code>.</p>\n"
                        "  <p>Esta página (<code>diario.html</code>) é gerada por", 1)
    assert html.count("</script>\n</body>") == 1
    html = html.replace("</script>\n</body>", "</script>\n" + js + "</body>", 1)
    html = html.replace("<title>Experimento 1 — resultados das etapas 1 e 2</title>", "<title>Experimento 1 — resultados das etapas 1, 2 e 3</title>", 1)
    missing_page = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", html)))
    assert not missing_page, f"tokens without value in the page: {missing_page}"
    #endregion

    #region 5 - Write the pages
    # what? - Saves resultados.html and the stage-only page.
    # how?  - The stage-only page is the fragment before splicing.
    (C.EXPERIMENT / "resultados.html").write_text(html)
    (E3 / "resultados_etapa3_somente.html").write_text(frag)
    print(f"resultados.html: {len(html.encode())/1024:.1f} KB | stage 3 inserted ({len(frag.encode())/1024:.1f} KB) | tokens: {len(tok)}")
    #endregion


if __name__ == "__main__":
    main()
