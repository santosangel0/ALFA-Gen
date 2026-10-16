"""
================================================================================
PROJETO ALFA-GEN: APLICAÇÃO STREAMLIT - PARÁFRASE INTELIGENTE & CLASSIFICAÇÃO
================================================================================
Ferramenta para professores criarem provas sem repetição de textos ou frases,
substituindo termos por seus sinônimos mais comuns em língua portuguesa,
preservando fórmulas matemáticas e monitorando o nível de ensino com modelos ONNX.
================================================================================
"""

import os
import sys
from pathlib import Path
import glob
import zipfile
import random
import pandas as pd
import numpy as np
import streamlit as st

# Garantir importações modulares a partir do diretório do script e da raiz
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from extrator_features import ExtratorFeatures, NOMES_FEATURES_ORDENADAS, ROTULOS_FEATURES
from classificador_onnx import ClassificadorONNX, MAPA_CLASSES
from modificador_sinonimos import ModificadorSinonimos

# Configuração da Página
st.set_page_config(
    page_title="ALFA-Gen: Assistente de Provas & Nível de Ensino",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS simples e limpa
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #0d6efd;
        margin-bottom: 10px;
    }
    .badge-status-ok {
        background-color: #d1e7dd;
        color: #0f5132;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-status-warn {
        background-color: #fff3cd;
        color: #664d03;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .stTextArea textarea {
        font-size: 14px;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CARREGAMENTO EM CACHE DOS RECURSOS (SPACY, NLTK, MODELOS ONNX, METADADOS)
# ==============================================================================
@st.cache_resource(show_spinner="Carregando modelos de IA e pipeline linguístico...")
def carregar_componentes():
    extrator = ExtratorFeatures()
    classificador = ClassificadorONNX(diretorio_modelos="modelos")
    modificador = ModificadorSinonimos()
    return extrator, classificador, modificador


@st.cache_data(show_spinner="Carregando banco de textos...")
def carregar_banco_dados():
    caminho_csv = "dados/dataset_multivariado_features.csv"
    if os.path.exists(caminho_csv):
        df = pd.read_csv(caminho_csv)
    else:
        df = pd.DataFrame()

    zips = glob.glob("dados/*.zip")
    zip_path = zips[0] if zips else None
    return df, zip_path


def ler_texto_do_zip(zip_path: str, caminho_relativo: str) -> str:
    if not zip_path or not os.path.exists(zip_path):
        return ""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            raw = z.read(caminho_relativo)
            try:
                return raw.decode("utf-8")
            except UnicodeDecodeError:
                return raw.decode("latin-1", errors="ignore")
    except Exception as e:
        return f"Erro ao ler arquivo: {e}"


# Inicializar instâncias
extrator, classificador, modificador = carregar_componentes()
df_banco, zip_path = carregar_banco_dados()


# ==============================================================================
# BARRA LATERAL (SIDEBAR): CONTROLES E SELEÇÃO DE TEXTO
# ==============================================================================
st.sidebar.title("📚 ALFA-Gen")
st.sidebar.caption("Avaliação Linguística e Filtragem Automática — Generator")
st.sidebar.markdown("---")

modo_entrada = st.sidebar.radio(
    "Fonte do Texto:",
    ["Banco de Dados ALFA-Gen", "Texto Livre / Colar Próprio"],
    index=0
)

texto_selecionado = ""
classe_real_banco = "Não especificado (Texto Livre)"
nome_arquivo_banco = ""

if modo_entrada == "Banco de Dados ALFA-Gen" and not df_banco.empty:
    st.sidebar.subheader("Filtrar Textos do Acervo")
    
    niveis_disponiveis = ["Todos"] + sorted(df_banco["nivel_completo"].dropna().unique().tolist())
    nivel_filtro = st.sidebar.selectbox("Filtrar por Nível Oficial:", niveis_disponiveis)
    
    df_filtrado = df_banco if nivel_filtro == "Todos" else df_banco[df_banco["nivel_completo"] == nivel_filtro]
    
    # Botão de sorteio
    if st.sidebar.button("🎲 Sortear Texto Aleatório"):
        st.session_state["random_idx"] = random.randint(0, len(df_filtrado) - 1)
        
    idx_padrao = st.session_state.get("random_idx", 0)
    if idx_padrao >= len(df_filtrado):
        idx_padrao = 0

    lista_opcoes = df_filtrado.apply(
        lambda r: f"[{r['id_arquivo']}] - {r['nivel_completo']}", axis=1
    ).tolist()

    escolha = st.sidebar.selectbox(
        "Selecione o Texto:",
        options=range(len(df_filtrado)),
        format_func=lambda i: lista_opcoes[i],
        index=idx_padrao
    )

    item_selecionado = df_filtrado.iloc[escolha]
    nome_arquivo_banco = item_selecionado["id_arquivo"]
    classe_real_banco = item_selecionado["nivel_completo"]
    caminho_rel = item_selecionado["caminho_relativo"]

    texto_selecionado = ler_texto_do_zip(zip_path, caminho_rel)

st.sidebar.markdown("---")
st.sidebar.subheader("Configurações do Modelo")
modelos_disp = classificador.obter_modelos_disponiveis()
modelo_escolhido = st.sidebar.selectbox(
    "Modelo Classificador ONNX:",
    options=modelos_disp,
    index=modelos_disp.index("LightGBM") if "LightGBM" in modelos_disp else 0,
    help="O LightGBM é o modelo de melhor acurácia (72,8% global, 92,2% ROC-AUC)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros de Sinônimos")
intensidade = st.sidebar.slider(
    "Intensidade de Substituição:",
    min_value=0.10,
    max_value=0.80,
    value=0.40,
    step=0.05,
    help="Fração de termos elegíveis (substantivos, adjetivos, verbos) a substituir."
)

apenas_mais_frequentes = st.sidebar.checkbox(
    "Priorizar Sinônimo Mais Comum na Língua Portuguesa",
    value=True,
    help="Seleciona a alternativa com o maior índice de frequência Zipf em português."
)

preservar_formulas = st.sidebar.checkbox(
    "Preservar Fórmulas Matemáticas e Termos Técnicos",
    value=True,
    help="Mantém equações LaTeX, notações, células de planilhas e termos disciplinares intactos."
)

semente_var = st.sidebar.number_input("Semente de Variação (Seed):", min_value=1, max_value=9999, value=42)


# ==============================================================================
# CORPO PRINCIPAL DA APLICAÇÃO
# ==============================================================================
st.title("📝 ALFA-Gen: Avaliação Linguística & Filtragem Automática")
st.markdown(
    "Assistente do Professor: **Geração de versões alternativas de questões e textos para avaliações** "
    "substituindo palavras por seus sinônimos mais comuns em português, **preservando o rigor de fórmulas** "
    "e monitorando em tempo real se a **complexidade pedagógica** (Fundamental I, II, Médio ou Superior) se manteve estável."
)

# Caixa de Entrada do Texto
st.subheader("1. Texto de Entrada")

col_info1, col_info2 = st.columns([3, 1])
with col_info1:
    if modo_entrada == "Banco de Dados ALFA-Gen":
        st.info(f"📂 **Arquivo do Banco:** `{nome_arquivo_banco}` | **Nível Registrado:** **{classe_real_banco}**")
with col_info2:
    if modo_entrada == "Banco de Dados ALFA-Gen":
        st.caption(f"Total no Acervo: {len(df_banco)} textos")

texto_input = st.text_area(
    "Texto para Paráfrase (você pode editar livremente se desejar):",
    value=texto_selecionado if texto_selecionado else "O menino encontrou uma bela moradia perto da floresta antiga.",
    height=180
)

# Botão de Ação
botao_executar = st.button("🚀 Modificar Texto & Avaliar Classificação", type="primary", use_container_width=True)

if botao_executar or "resultado_execucao" in st.session_state:
    if botao_executar:
        with st.spinner("Modificando texto e executando inferência no modelo ONNX..."):
            # 1. Modificar texto com sinônimos
            res_mod = modificador.modificar_texto(
                texto=texto_input,
                intensidade=intensidade,
                apenas_mais_frequentes=apenas_mais_frequentes,
                semente=semente_var
            )

            # 2. Extrair features de ambos os textos
            feats_orig = extrator.extrair_features(texto_input)
            feats_mod = extrator.extrair_features(res_mod["texto_modificado"])

            # 3. Classificar ambos os textos
            pred_orig = classificador.prever(feats_orig, modelo_escolhido)
            pred_mod = classificador.prever(feats_mod, modelo_escolhido)

            st.session_state["resultado_execucao"] = {
                "res_mod": res_mod,
                "feats_orig": feats_orig,
                "feats_mod": feats_mod,
                "pred_orig": pred_orig,
                "pred_mod": pred_mod,
                "classe_real": classe_real_banco
            }

    # Recuperar resultados
    dados_res = st.session_state["resultado_execucao"]
    res_mod = dados_res["res_mod"]
    pred_orig = dados_res["pred_orig"]
    pred_mod = dados_res["pred_mod"]
    feats_orig = dados_res["feats_orig"]
    feats_mod = dados_res["feats_mod"]
    classe_real = dados_res["classe_real"]

    nivel_orig = pred_orig["classe_nome"]
    nivel_mod = pred_mod["classe_nome"]
    mudou_nivel = (nivel_orig != nivel_mod)

    st.markdown("---")
    st.subheader("2. Diagnóstico Pedagógico do Nível de Ensino")

    # Alerta em Destaque
    if not mudou_nivel:
        st.success(
            f"✅ **NÍVEL PEDAGÓGICO MANTIDO:** O texto modificado permaneceu classificado como **{nivel_mod}** "
            f"(Confiança de {pred_mod['confianca']*100:.1f}%). A prova mantém a adequação para o público-alvo pretendido."
        )
    else:
        st.warning(
            f"⚠️ **MUDANÇA DE CATEGORIA DETECTADA:** "
            f"O texto original foi classificado como **{nivel_orig}**, mas após as modificações léxicas "
            f"ele migrou para **{nivel_mod}** (Confiança de {pred_mod['confianca']*100:.1f}%). "
            f"Avalie se a alteração de vocabulário tornou o texto mais simples ou mais complexo do que o desejado!"
        )

    # Cards comparativos
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        st.metric(
            label="Nível no Banco (Gabarito)",
            value=classe_real.replace("Ensino ", "")
        )
    with col_c2:
        st.metric(
            label="Predição Texto Original",
            value=nivel_orig.replace("Ensino ", ""),
            delta=f"{pred_orig['confianca']*100:.1f}% conf."
        )
    with col_c3:
        st.metric(
            label="Predição Texto Modificado",
            value=nivel_mod.replace("Ensino ", ""),
            delta=f"{pred_mod['confianca']*100:.1f}% conf.",
            delta_color="normal" if not mudou_nivel else "inverse"
        )
    with col_c4:
        st.metric(
            label="Palavras Substituídas",
            value=f"{res_mod['total_modificacoes']}",
            delta=f"{res_mod['taxa_substituicao_pct']}% substituídas"
        )

    # Probabilidades detalhadas
    st.markdown("##### Distribuição de Probabilidades por Nível:")
    df_probs = pd.DataFrame({
        "Nível de Ensino": list(pred_orig["probabilidades"].keys()),
        "Texto Original (%)": [round(v * 100, 1) for v in pred_orig["probabilidades"].values()],
        "Texto Modificado (%)": [round(v * 100, 1) for v in pred_mod["probabilidades"].values()]
    })
    st.dataframe(
        df_probs.style.highlight_max(axis=0, subset=["Texto Original (%)", "Texto Modificado (%)"], color="#d1e7dd"),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")
    st.subheader("3. Comparativo Lado a Lado dos Textos")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("**📖 Texto Original:**")
        st.text_area(
            "Original:",
            value=res_mod["texto_original"],
            height=260,
            disabled=True,
            label_visibility="collapsed"
        )
        st.caption(f"Fórmulas identificadas e protegidas: {res_mod['formulas_preservadas']}")

    with col_t2:
        st.markdown("**✏️ Novo Texto Modificado (Pronto para Uso em Prova):**")
        st.text_area(
            "Modificado:",
            value=res_mod["texto_modificado"],
            height=260,
            label_visibility="collapsed"
        )
        st.caption("Você pode copiar o texto acima diretamente para o seu documento de prova.")

    # Tabela de Modificações Realizadas
    st.markdown("---")
    st.subheader("4. Dicionário de Sinônimos Aplicados nesta Versão")

    if res_mod["modificacoes"]:
        df_mod = pd.DataFrame(res_mod["modificacoes"])
        col_names = {
            "original": "Palavra Original",
            "substituto": "Termo Adotado",
            "pos": "Classe Gramatical",
            "freq_original": "Freq. Original (Zipf)",
            "freq_substituto": "Freq. Adotado (Zipf)",
            "ganho_frequencia": "Ganho Frequência",
            "motivo": "Justificativa / Concordância"
        }
        df_mod = df_mod.rename(columns={k: v for k, v in col_names.items() if k in df_mod.columns})
        st.dataframe(df_mod, use_container_width=True, hide_index=True)
    else:
        st.info("Nenhuma palavra foi substituída com as configurações atuais de intensidade e frequência mínima.")

    # Detalhamento das 15 Features Linguísticas
    with st.expander("🔍 Ver Impacto Detalhado nas 15 Features Linguísticas (Métricas de Legibilidade)"):
        st.markdown(
            "Abaixo estão os valores exatos das 15 métricas morfossintáticas e lexicais calculadas para ambos os textos, "
            "as mesmas utilizadas pelo modelo classificador ONNX:"
        )

        linhas_feat = []
        for feat in NOMES_FEATURES_ORDENADAS:
            v_orig = feats_orig[feat]
            v_mod = feats_mod[feat]
            delta = v_mod - v_orig
            pct = (delta / v_orig * 100.0) if v_orig != 0 else 0.0

            linhas_feat.append({
                "Atributo Linguístico": ROTULOS_FEATURES[feat],
                "Original": round(v_orig, 4),
                "Modificado": round(v_mod, 4),
                "Diferença (Δ)": round(delta, 4),
                "Variação (%)": f"{pct:+.1f}%" if v_orig != 0 else "-"
            })

        df_feats = pd.DataFrame(linhas_feat)
        st.dataframe(df_feats, use_container_width=True, hide_index=True)
