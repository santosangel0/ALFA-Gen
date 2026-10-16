"""
================================================================================
PROJETO ALFA-GEN: EXTRAÇÃO DAS 5 NOVAS FEATURES LINGUÍSTICAS
================================================================================
Features extraídas para os 2.076 textos:
  1. Razão Nominal (Substantivos / Verbos) -> dados/features_razao_nominal.csv
  2. Conjunções Coordenativas (CCONJ) e Razão CCONJ/SCONJ -> dados/features_coordenativas.csv
  3. Palavras Polissílabas (>= 4 sílabas) e Sílabas por Palavra -> dados/features_polissilabos.csv
  4. Densidade de Vírgulas por Sentença -> dados/features_virgulas.csv
  5. Metadiscurso e Léxico Acadêmico -> dados/features_metadiscurso.csv

Processamento:
  - Utiliza spaCy (pt_core_news_sm) e Pyphen (pt_BR) em lote (nlp.pipe)
  - Processa os 2.076 arquivos originais do dataset_caed
================================================================================
"""

import os
import re
import unicodedata
import pandas as pd
import spacy
import pyphen
from tqdm import tqdm


def normalizar_texto(texto: str) -> str:
    """Remove diacríticos/acentos e converte para minúsculas para matching léxico."""
    texto_norm = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto_norm if not unicodedata.combining(c)).lower()


# Vocabulário formal de metadiscurso acadêmico (lemas / raízes normalizadas)
METADISCURSO_TERMOS = {
    "autor", "autores", "autora", "autoras",
    "estudo", "estudos", "pesquisa", "pesquisas", "pesquisador", "pesquisadores",
    "metodologia", "metodologias", "metodologico", "metodologica", "metodologicos", "metodologicas",
    "dado", "dados", "analise", "analises", "analisar", "analisou", "analisando",
    "conclusao", "conclusoes", "concluir", "conclui-se", "concluiu-se",
    "tabela", "tabelas", "figura", "figuras", "grafico", "graficos", "quadro", "quadros",
    "capitulo", "capitulos", "artigo", "artigos", "secao", "secoes",
    "evidencia", "evidencias", "hipotese", "hipoteses",
    "resultado", "resultados", "investigacao", "investigacoes",
    "conceito", "conceitos", "teoria", "teorias", "teorico", "teorica", "teoricos", "teoricas",
    "bibliografia", "referencia", "referencias", "citacao", "citacoes",
    "observa-se", "constata-se", "verifica-se", "discute-se", "infere-se", "nota-se",
    "proposta", "propostas", "experimento", "experimentos", "amostra", "amostras",
    "cientifico", "cientifica", "cientificos", "cientificas", "academico", "academica"
}


def main():
    print("=" * 80)
    print("EXTRAÇÃO DAS 5 NOVAS FEATURES LINGUÍSTICAS (2.076 TEXTOS)")
    print("=" * 80)
    
    # Carregar modelo spaCy
    print("Carregando modelo spaCy pt_core_news_sm...")
    nlp = spacy.load("pt_core_news_sm", disable=["ner"])
    
    # Carregar hifenizador Pyphen para pt_BR
    print("Inicializando hifenizador Pyphen pt_BR...")
    dic_pyphen = pyphen.Pyphen(lang="pt_BR")
    
    import glob
    import zipfile
    
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
    
    with zipfile.ZipFile(zip_path, 'r') as z:
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
                "nivel_completo": nivel_comp
            })
            textos_conteudo.append(txt)
                
    total_docs = len(registros_meta)
    print(f"Total de arquivos mapeados: {total_docs}")
    assert total_docs == 2076, f"Esperado 2.076 arquivos, encontrado {total_docs}!"
    
    # Estruturas para cada feature
    dados_razao_nominal = []
    dados_coordenativas = []
    dados_polissilabos = []
    dados_virgulas = []
    dados_metadiscurso = []
    
    print("\nExecutando pipeline em lote spaCy (nlp.pipe)...")
    docs = nlp.pipe(textos_conteudo, batch_size=64)
    
    for idx, doc in enumerate(tqdm(docs, total=total_docs, desc="Processando textos")):
        meta = registros_meta[idx]
        caminho = meta["caminho_relativo"]
        id_arq = meta["id_arquivo"]
        nivel_ord = meta["nivel_ordem"]
        nivel_comp = meta["nivel_completo"]
        texto_raw = textos_conteudo[idx]
        
        # Palavras alfabéticas
        tokens_palavras = [t for t in doc if t.is_alpha and not t.is_space]
        total_palavras = max(len(tokens_palavras), 1)
        
        # Sentenças
        sentencas = list(doc.sents)
        total_sentencas = max(len(sentencas), 1)
        
        # -------------------------------------------------------------------------
        # 1. Razão Nominal (Substantivos / Verbos)
        # -------------------------------------------------------------------------
        substantivos = [t for t in doc if t.pos_ == "NOUN"]
        proprios = [t for t in doc if t.pos_ == "PROPN"]
        verbos = [t for t in doc if t.pos_ in ("VERB", "AUX")]
        
        subst_qtd = len(substantivos)
        verbos_qtd = len(verbos)
        
        subst_prop = subst_qtd / total_palavras
        verbos_prop = verbos_qtd / total_palavras
        razao_nominal = subst_qtd / max(verbos_qtd, 1)
        # Índice nominal ampliado (Substantivos + Nomes Próprios) / Verbos
        razao_nominal_ampla = (subst_qtd + len(proprios)) / max(verbos_qtd, 1)
        
        dados_razao_nominal.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "substantivos_qtd": subst_qtd,
            "verbos_qtd": verbos_qtd,
            "substantivos_prop": subst_prop,
            "verbos_prop": verbos_prop,
            "razao_nominal": razao_nominal,
            "razao_nominal_ampla": razao_nominal_ampla
        })
        
        # -------------------------------------------------------------------------
        # 2. Conjunções Coordenativas (CCONJ) e Razão CCONJ/SCONJ
        # -------------------------------------------------------------------------
        cconjs = [t.text.lower() for t in doc if t.pos_ == "CCONJ"]
        sconjs = [t.text.lower() for t in doc if t.pos_ == "SCONJ"]
        
        cconj_qtd = len(cconjs)
        sconj_qtd = len(sconjs)
        cconj_densidade = cconj_qtd / total_palavras
        
        # Razão com suavização de Laplace para evitar divisão por zero
        razao_cconj_sconj = (cconj_qtd + 0.5) / (sconj_qtd + 0.5)
        
        amostra_cconj = ", ".join(cconjs[:5]) if cconjs else "nenhuma"
        
        dados_coordenativas.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "cconj_qtd": cconj_qtd,
            "sconj_qtd": sconj_qtd,
            "cconj_densidade": cconj_densidade,
            "razao_cconj_sconj": razao_cconj_sconj,
            "amostra_cconj": amostra_cconj
        })
        
        # -------------------------------------------------------------------------
        # 3. Palavras Polissílabas (>= 4 sílabas) e Sílabas por Palavra
        # -------------------------------------------------------------------------
        polissilabos = []
        silabas_total = 0
        
        for t in tokens_palavras:
            palavra_limpa = t.text.lower()
            hifenizada = dic_pyphen.inserted(palavra_limpa)
            if hifenizada:
                partes = hifenizada.split("-")
                n_silabas = len(partes)
            else:
                n_silabas = 1
                
            silabas_total += n_silabas
            if n_silabas >= 4:
                polissilabos.append(palavra_limpa)
                
        polissilabos_qtd = len(polissilabos)
        prop_polissilabos = polissilabos_qtd / total_palavras
        silabas_por_palavra = silabas_total / total_palavras
        amostra_poli = ", ".join(polissilabos[:5]) if polissilabos else "nenhum"
        
        dados_polissilabos.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "silabas_total": silabas_total,
            "silabas_por_palavra": silabas_por_palavra,
            "polissilabos_qtd": polissilabos_qtd,
            "prop_polissilabos": prop_polissilabos,
            "amostra_polissilabos": amostra_poli
        })
        
        # -------------------------------------------------------------------------
        # 4. Densidade de Vírgulas por Sentença
        # -------------------------------------------------------------------------
        virgulas_qtd = texto_raw.count(",")
        virgulas_por_sentenca = virgulas_qtd / total_sentencas
        virgulas_prop = virgulas_qtd / total_palavras
        
        dados_virgulas.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "total_sentencas": total_sentencas,
            "virgulas_qtd": virgulas_qtd,
            "virgulas_por_sentenca": virgulas_por_sentenca,
            "virgulas_prop": virgulas_prop
        })
        
        # -------------------------------------------------------------------------
        # 5. Metadiscurso e Léxico Acadêmico
        # -------------------------------------------------------------------------
        termos_encontrados = []
        for t in tokens_palavras:
            tok_norm = normalizar_texto(t.text)
            if tok_norm in METADISCURSO_TERMOS:
                termos_encontrados.append(t.text.lower())
                
        metadiscurso_qtd = len(termos_encontrados)
        metadiscurso_prop = metadiscurso_qtd / total_palavras
        amostra_meta = ", ".join(termos_encontrados[:5]) if termos_encontrados else "nenhum"
        
        dados_metadiscurso.append({
            "caminho_relativo": caminho,
            "id_arquivo": id_arq,
            "nivel_ordem": nivel_ord,
            "nivel_completo": nivel_comp,
            "total_palavras": total_palavras,
            "metadiscurso_qtd": metadiscurso_qtd,
            "metadiscurso_prop": metadiscurso_prop,
            "amostra_metadiscurso": amostra_meta
        })

    # Salvar cada dataset em dados/
    os.makedirs("dados", exist_ok=True)
    
    print("\nSalvando arquivos CSV individuais em dados/...")
    
    df_rn = pd.DataFrame(dados_razao_nominal)
    df_rn.to_csv("dados/features_razao_nominal.csv", index=False)
    print("Salvo: dados/features_razao_nominal.csv")
    
    df_cc = pd.DataFrame(dados_coordenativas)
    df_cc.to_csv("dados/features_coordenativas.csv", index=False)
    print("Salvo: dados/features_coordenativas.csv")
    
    df_pl = pd.DataFrame(dados_polissilabos)
    df_pl.to_csv("dados/features_polissilabos.csv", index=False)
    print("Salvo: dados/features_polissilabos.csv")
    
    df_vg = pd.DataFrame(dados_virgulas)
    df_vg.to_csv("dados/features_virgulas.csv", index=False)
    print("Salvo: dados/features_virgulas.csv")
    
    df_md = pd.DataFrame(dados_metadiscurso)
    df_md.to_csv("dados/features_metadiscurso.csv", index=False)
    print("Salvo: dados/features_metadiscurso.csv")
    
    print("\n" + "=" * 80)
    print("TODAS AS 5 NOVAS FEATURES FORAM EXTRAÍDAS E SALVAS COM SUCESSO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
