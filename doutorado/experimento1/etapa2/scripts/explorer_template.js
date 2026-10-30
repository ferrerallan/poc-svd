// ---------------- exploracao detalhada: as 4 medidas, uma abaixo da outra, tudo aberto
const DET1 = __DET1__, DET2 = __DET2__;
const MEDIDAS = {
  concentracao:{rot:"Concentração", ch:"concentracao", f:1, un:"", c:1, cd:2,
    mede:"Se a mudança da matriz foi mais do tipo <b>alterar os pesos das direções</b> (valores singulares) do que o acaso daria.",
    calc:"Projeta a mudança ΔW = W1 − W0 na base da SVD da matriz original (M = U0ᵀ·ΔW·V0). A fração da energia de M (soma dos quadrados) que está na <b>diagonal</b> é a parte &laquo;valores singulares&raquo;. A medida é essa fração multiplicada pelo maior lado da matriz.",
    ref:"<b>1</b> = indistinguível de uma mudança aleatória. Quanto maior, mais da mudança está na diagonal.",
    nao:"Não mede o tamanho da mudança nem o giro das direções. Um valor alto pode corresponder a uma fração pequena da mudança total: a maior parte da energia continua fora da diagonal."},
  giro:{rot:"Giro", ch:"giro_dos_vetores_graus", f:1, un:"°", c:2, cd:2,
    mede:"Quanto as <b>10 direções principais</b> da matriz giraram.",
    calc:"Para cada uma das 10 maiores direções (vetores singulares à esquerda, U), o ângulo entre a versão original e a de depois (arco cosseno do valor absoluto do cosseno). O valor é a média dos 10 ângulos.",
    ref:"<b>0°</b> = não girou. <b>90°</b> = ficou perpendicular à original.",
    nao:"Não mede mudança de pesos. Quando dois valores singulares vizinhos são quase iguais, as direções correspondentes ficam mal definidas e podem girar sozinhas, o que infla esta medida."},
  deslocamento:{rot:"Deslocamento", ch:"tamanho_mudanca", f:100, un:"%", c:2, cd:3,
    mede:"O <b>tamanho total</b> da mudança da matriz.",
    calc:"‖W1 − W0‖ ÷ ‖W0‖, com a norma de Frobenius (raiz da soma dos quadrados de todos os números), em porcentagem.",
    ref:"<b>0%</b> = a matriz não mudou.",
    nao:"Não distingue mudança de pesos de mudança de direções."},
  variacao:{rot:"Variação dos pesos", ch:"variacao_dos_valores", f:100, un:"%", c:3, cd:4,
    mede:"Quanto mudaram os <b>pesos</b> (valores singulares) das 10 direções principais.",
    calc:"Para cada um dos 10 maiores valores singulares, |σ depois − σ antes| ÷ σ antes, comparando o 1º com o 1º, o 2º com o 2º e assim por diante. O valor é a média dos 10, em porcentagem.",
    ref:"<b>0%</b> = nenhum peso mudou.",
    nao:"Só olha os 10 maiores valores singulares, não a matriz inteira. Não mede giro."}
};
function explorar(contId, DET, temRuido, comFicha){
  const cont = document.getElementById(contId);
  for(const [id, m] of Object.entries(MEDIDAS)){
    const k = m.ch, f = m.f, un = m.un, idM = contId + "-" + id, g = DET.global[k];
    const sec = document.createElement("div"); sec.className = "medida";
    let h = `<h4 id="${idM}">${m.rot}</h4>`;
    if(comFicha){
      h += `<dl class="fichadl"><dt>O que mede</dt><dd>${m.mede}</dd><dt>Como é calculada</dt><dd>${m.calc}</dd><dt>Referência</dt><dd>${m.ref}</dd><dt>O que não mede</dt><dd>${m.nao}</dd></dl>`;
    } else {
      h += `<p style="font-size:13.5px;color:var(--ink3)">Mesma medida da etapa 1: ${m.mede}</p>`;
    }
    h += `<div class="resumo"><span>Média das 72 matrizes. Adaptação: <b>${fmt(g.adaptacao.media*f,m.c)}${un}</b> ± ${fmt(g.adaptacao.dp*f,m.cd)} (entre ${DET.n} treinos)</span>` +
         (temRuido ? `<span>Piso de ruído: <b>${fmt(g.ruido.media*f,m.c)}${un}</b> ± ${fmt(g.ruido.dp*f,m.cd)} (entre ${DET.np} pares)</span>` : ``) + `</div>`;
    h += `<div class="mapas" id="${idM}-mapas"></div>`;
    sec.innerHTML = h; cont.appendChild(sec);
    const A = c => c[k].adaptacao.media*f, R = c => c[k].ruido.media*f, S = c => c[k].adaptacao.dp*f;
    const mk = fn => ({celulas: DET.celulas.map(c => ({b:c.b, t:c.t, v:fn(c)}))});
    const vAR = DET.celulas.flatMap(c => temRuido ? [A(c), R(c)] : [A(c)]), vS = DET.celulas.map(S);
    const mnA = Math.min(...vAR), mxA = Math.max(...vAR), mnS = Math.min(...vS), mxS = Math.max(...vS);
    mapa(idM+"-mapas", "Adaptação", `original → treino, média de ${DET.n} treinos`, mk(A), "v", m.c, mnA, mxA, un);
    if(temRuido) mapa(idM+"-mapas", "Piso de ruído", `treino i → treino j, média de ${DET.np} pares`, mk(R), "v", m.c, mnA, mxA, un);
    mapa(idM+"-mapas", "Variação entre sementes", `desvio padrão da adaptação entre os ${DET.n} treinos`, mk(S), "v", m.cd, mnS, mxS, un);
  }
}
explorar("exp1", DET1, true, true);
explorar("exp2", DET2, false, false);
