"""
================================================================================
PROJETO ALFA-GEN: EXTRAÇÃO DO LOTE 3 DE FEATURES LINGUÍSTICAS (5 FEATURES)
================================================================================
Features direcionadas para resolver a confusão Fundamental II vs. Médio e Médio vs. Superior:
  1. Citações e Referenciação Acadêmica (citacao_academica_prop) -> dados/features_citacao_academica.csv
  2. Sufixos Abstratos e Nominalizações (nominalizacao_prop) -> dados/features_nominalizacao.csv
  3. Conectivos Argumentativos e Conclusivos (conectivos_argumentativos_prop) -> dados/features_conectivos_argumentativos.csv
  4. Pontuação Estruturante Complexa (pont_estruturante_prop) -> dados/features_pont_estruturante.csv
  5. Densidade de Pessoalidade / 1ª Pessoa (pessoalidade_prop) -> dados/features_pessoalidade.csv

Processamento:
  - Utiliza spaCy (pt_core_news_sm) e expressões regulares nos 2.076 arquivos do acervo
================================================================================
"""

import os
import re
import unicodedata
import zipfile
import glob
import pandas as pd
import spacy
from tqdm import tqdm


def normalizar_texto(texto: str) -> str:
    """Remove diacríticos e converte para minúsculas para matching consistente."""
    texto_norm = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto_norm if not unicodedata.combining(c)).lower()


# Padrões regex para citações acadêmicas e referências formais
RE_CITACAO_ANO = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
RE_CITACAO_PARENTETICA = re.compile(r"\([A-ZÀ-Ú][a-zà-úA-ZÀ-Ú\s\.,;&\-]+,?\s*(?:19|20)\d{2}[a-z]?\)")
RE_CITACAO_TERMOS = re.compile(r"\b(?:et\s+al\.?|apud|ibid\.?|op\.\s*cit\.?|in\s+memoriam)\b", re.IGNORECASE)
RE_PAGINACAO = re.compile(r"\b(?:p\.|pp\.|págs?\.|p[aá]gina)\s*\d+\b", re.IGNORECASE)
RE_SECAO_NUMERADA = re.compile(r"^(?:[1-9]\.[0-9](?:\.[0-9])?)\s+[A-ZÀ-Ú]", re.MULTILINE)

# Padrões regex para nominalizações com sufixos abstratos em português (palavras >= 6 caracteres)
RE_NOMINALIZACAO = re.compile(
    r"\b[a-z]{3,}(?:cao|coes|sao|soes|xao|xoes|mento|mentos|dade|dades|ismo|ismos|encia|encias|ancia|ancias|tude|tudes)\b"
)

# Conectivos argumentativos e propositivos formais (normalizados sem acento)
CONECTIVOS_EXPRESSOES = [
    # Conclusivos e Consecutivos formais
    "portanto", "por conseguinte", "por consequencia", "dessa forma", "desse modo",
    "deste modo", "desta forma", "em suma", "assim sendo", "haja vista",
    "diante disso", "ante o exposto", "em virtude disso", "sendo assim",
    # Aditivos e de continuidade argumentativa
    "ademais", "outrossim", "alem disso", "alem do mais", "somado a isso",
    # Opositivos formais
    "contudo", "todavia", "entretanto", "no entanto", "nao obstante",
    # Fórmulas de tese / proposta de intervenção (ENEM / Dissertação)
    "faz-se necessario", "faz-se mister", "torna-se evidente", "torna-se mister",
    "cabe ao", "cabe a", "e primordial", "e mister", "deve-se", "urgente que"
]

# Pronomes de 1ª pessoa (pessoalidade)
PRONOMES_1P = {
    # Singular
    "eu", "me", "mim", "comigo", "meu", "meus", "minha", "minhas",
    # Plural
    "nos", "conosco", "nosso", "nossos", "nossa", "nossas"
}


def main():
    print("=" * 80)
    print("EXTRAÇÃO DO LOTE 3: 5 NOVAS FEATURES LINGUÍSTICAS (2.076 TEXTOS)")
    print("=" * 80)

    print("Carregando spaCy pt_core_news_sm...")
    nlp = spacy.load("pt_core_news_sm", disable=["ner"])

    zips = glob.glob("dados/*.zip")
    if not zips:
        raise FileNotFoundError("Nenhum arquivo .zip encontrado em dados/")
    zip_path = zips[0]
    print(f"[INFO] Processando arquivo zip: {zip_path}")

    def extrair_nivel(caminho: str) -> tuple:
        if "1_Ensino_Fundamental_I" in caminho:
            return (1, "Ensino Fundamental I")
        elif "2_Ensino_Fundamental_II" in caminho:
            return (2, "Ensino Fundamental II")
        elif "3_Ensino_Medio" in caminho:
            return (3, "Ensino Médio")
        elif "4_Ensino_Superior" in caminho:
            return (4, "Ensino Superior")
        return (0, "Desconhecido")

    registros_meta = []
    textos_conteudo = []

    with zipfile.ZipFile(zip_path, "r") as z:
        arquivos_txt = sorted([
            f for f in z.namelist()
            if not f.startswith("__MACOSX") and f.endswith(".txt") and "dataset_caed/" in f
        ])
        print(f"[INFO] Total de textos a processar: {len(arquivos_txt)}")

        for arq in arquivos_txt:
            raw = z.read(arq)
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:
                txt = raw.decode("latin-1", errors="ignore")

            ordem, nivel_comp = extrair_nivel(arq)
            registros_meta.append({
                "caminho_relativo": arq,
                "id_arquivo": os.path.basename(arq),
                "nivel_ordem": ordem,
                "nivel_completo": nivel_comp,
            })
            textos_conteudo.append(txt)

    total_docs = len(registros_meta)
    print(f"Total de arquivos mapeados: {total_docs}")
    assert total_docs == 2076, f"Esperado 2.076 arquivos, encontrado {total_docs}!"

    dados_citacao = []
    dados_nominalizacao = []
    dados_conectivos = []
    dados_pont_estruturante = []
    dados_pessoalidade = []

    print("\nExecutando pipeline em lote spaCy (nlp.pipe)...")
    docs = nlp.pipe(textos_conteudo, batch_size=64)

    for idx, doc in enumerate(tqdm(docs, total=total_docs, desc="Processando lote 3")):
        meta = registros_meta[idx]
        caminho = meta["caminho_relativo"]
        id_arq = meta["id_arquivo"]
        nivel_ord = meta["nivel_ordem"]
        nivel_comp = meta["nivel_completo"]
        texto_raw = textos_conteudo[idx]
        texto_norm = normalizar_texto(texto_raw)

        tokens_palavras = [t for t in doc if t.is_alpha and not t.is_space]
        total_palavras = max(len(tokens_palavras), 1)

        sentencas = list(doc.sents)
        total_sentencas = max(len(sentencas), 1)

        # ---------------------------------------------------------------------
        # 1. Citações e Referenciação Acadêmica (citacao_academica_prop)
        # ---------------------------------------------------------------------
        # Conta anos entre parênteses, citações autor-ano, termos et al., paginação e seções numeradas
        cit_parenteses = len(RE_CITACAO_PARENTETICA.findall(texto_raw))
        cit_termos = len(RE_CITACAO_TERMOS.findall(texto_raw))
        cit_paginacao = len(RE_PAGINACAO.findall(texto_raw))
        cit_secao = len(RE_SECAO_NUMERADA.findall(texto_raw))
        # Anos gerais entre parênteses: e.g. (2018)
        anos_parenteses = len(re.findall(r"\((?:19|20)\d{2}[a-z]?\)", texto_raw))

        total_marcadores_citacao = (
            cit_parenteses + cit_termos + cit_paginacao + cit_secao + anos_parenteses
        )
        citacao_academica_prop = total_marcadores_citacao / total_palavras
        citacao_academica_por_mil = (total_marcadores_citacao / total_palavras) * 1000.0

        dados_citacao.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "citacao_academica_qtd": total_marcadores_citacao,
            "citacao_academica_prop": citacao_academica_prop,
            "citacao_academica_por_mil": citacao_academica_por_mil,
        })

        # ---------------------------------------------------------------------
        # 2. Sufixos Abstratos e Nominalizações (nominalizacao_prop)
        # ---------------------------------------------------------------------
        # Identifica substantivos com sufixos abstratos em tokens normalizados
        tokens_norm = [normalizar_texto(t.text) for t in tokens_palavras]
        nominalizacoes_encontradas = [
            t for t in tokens_norm if RE_NOMINALIZACAO.match(t)
        ]
        total_nominalizacoes = len(nominalizacoes_encontradas)
        nominalizacao_prop = total_nominalizacoes / total_palavras

        dados_nominalizacao.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "nominalizacao_qtd": total_nominalizacoes,
            "nominalizacao_prop": nominalizacao_prop,
        })

        # ---------------------------------------------------------------------
        # 3. Conectivos Argumentativos e Conclusivos (conectivos_argumentativos_prop)
        # ---------------------------------------------------------------------
        total_conectivos = 0
        for expr in CONECTIVOS_EXPRESSOES:
            # Busca com boundary
            padrao = r"\b" + re.escape(expr) + r"\b"
            total_conectivos += len(re.findall(padrao, texto_norm))

        conectivos_argumentativos_prop = total_conectivos / total_palavras
        conectivos_argumentativos_por_mil = (total_conectivos / total_palavras) * 1000.0

        dados_conectivos.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "conectivos_argumentativos_qtd": total_conectivos,
            "conectivos_argumentativos_prop": conectivos_argumentativos_prop,
            "conectivos_argumentativos_por_mil": conectivos_argumentativos_por_mil,
        })

        # ---------------------------------------------------------------------
        # 4. Pontuação Estruturante Complexa (pont_estruturante_prop)
        # ---------------------------------------------------------------------
        # Conta ponto e vírgula (;), dois pontos (:), parênteses abertos e travessões
        qtd_ponto_virgula = texto_raw.count(";")
        qtd_dois_pontos = texto_raw.count(":")
        qtd_parenteses = texto_raw.count("(")
        qtd_travessoes = len(re.findall(r"(?:—|--)", texto_raw))

        total_estruturante = qtd_ponto_virgula + qtd_dois_pontos + qtd_parenteses + qtd_travessoes
        pont_estruturante_prop = total_estruturante / total_palavras
        pont_estruturante_por_sentenca = total_estruturante / total_sentencas

        dados_pont_estruturante.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "total_sentencas": total_sentencas,
            "ponto_virgula_qtd": qtd_ponto_virgula,
            "dois_pontos_qtd": qtd_dois_pontos,
            "parenteses_qtd": qtd_parenteses,
            "travessoes_qtd": qtd_travessoes,
            "pont_estruturante_qtd": total_estruturante,
            "pont_estruturante_prop": pont_estruturante_prop,
            "pont_estruturante_por_sentenca": pont_estruturante_por_sentenca,
        })

        # ---------------------------------------------------------------------
        # 5. Densidade de Pessoalidade / 1ª Pessoa (pessoalidade_prop)
        # ---------------------------------------------------------------------
        tokens_1p = [t for t in tokens_norm if t in PRONOMES_1P]
        total_pessoalidade = len(tokens_1p)
        pessoalidade_prop = total_pessoalidade / total_palavras

        dados_pessoalidade.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "pessoalidade_qtd": total_pessoalidade,
            "pessoalidade_prop": pessoalidade_prop,
        })

    # Salvar DataFrames
    print("\nSalvando DataFrames em dados/...")
    pd.DataFrame(dados_citacao).to_csv("dados/features_citacao_academica.csv", index=False)
    print("Salvo: dados/features_citacao_academica.csv")

    pd.DataFrame(dados_nominalizacao).to_csv("dados/features_nominalizacao.csv", index=False)
    print("Salvo: dados/features_nominalizacao.csv")

    pd.DataFrame(dados_conectivos).to_csv("dados/features_conectivos_argumentativos.csv", index=False)
    print("Salvo: dados/features_conectivos_argumentativos.csv")

    pd.DataFrame(dados_pont_estruturante).to_csv("dados/features_pont_estruturante.csv", index=False)
    print("Salvo: dados/features_pont_estruturante.csv")

    pd.DataFrame(dados_pessoalidade).to_csv("dados/features_pessoalidade.csv", index=False)
    print("Salvo: dados/features_pessoalidade.csv")

    print("\n[SUCESSO] Todas as 5 novas features do lote 3 foram extraídas com sucesso!")


if __name__ == "__main__":
    main()
