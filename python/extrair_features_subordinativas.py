#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto ALFA-Gen: Extração da Feature de Densidade de Conjunções Subordinativas
"""

import os
import glob
import re
import zipfile
import pandas as pd
import spacy

# Carregar modelo spaCy desabilitando NER e Lemmatizer para máxima performance
nlp = spacy.load("pt_core_news_sm", disable=["ner", "lemmatizer"])

def extrair_nivel(caminho: str) -> tuple:
    if "1_Ensino_Fundamental_I" in caminho:
        return (1, "Fundamental I", "Ensino Fundamental I")
    elif "2_Ensino_Fundamental_II" in caminho:
        return (2, "Fundamental II", "Ensino Fundamental II")
    elif "3_Ensino_Medio" in caminho:
        return (3, "Ensino Médio", "Ensino Médio")
    elif "4_Ensino_Superior" in caminho:
        return (4, "Ensino Superior", "Ensino Superior")
    return (0, "Desconhecido", "Desconhecido")

def processar_dataset():
    zips = glob.glob("dados/*.zip")
    if not zips:
        raise FileNotFoundError("Nenhum arquivo .zip encontrado em dados/")
    zip_path = zips[0]
    print(f"[INFO] Processando arquivo zip: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        arquivos_txt = [
            f for f in z.namelist() 
            if not f.startswith("__MACOSX") and f.endswith(".txt") and "dataset_caed/" in f
        ]
        print(f"[INFO] Total de textos a processar: {len(arquivos_txt)}")
        
        # Carregar todos os textos
        textos_brutos = []
        metadados = []
        for arq in arquivos_txt:
            raw = z.read(arq)
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:
                txt = raw.decode("latin-1", errors="ignore")
            
            ordem, nivel_simp, nivel_comp = extrair_nivel(arq)
            textos_brutos.append(txt)
            metadados.append({
                "id_arquivo": os.path.basename(arq),
                "caminho_relativo": arq,
                "nivel_ordem": ordem,
                "nivel": nivel_simp,
                "nivel_completo": nivel_comp
            })

    print("[INFO] Executando pipeline spaCy (POS-tagging) em lote com nlp.pipe...")
    registros = []
    
    # Processar com nlp.pipe
    for idx, doc in enumerate(nlp.pipe(textos_brutos, batch_size=64), start=0):
        if (idx + 1) % 500 == 0 or (idx + 1) == len(textos_brutos):
            print(f"  -> Concluído {idx + 1}/{len(textos_brutos)} textos...")
            
        meta = metadados[idx]
        
        # Filtrar palavras alfabéticas válidas
        palavras = [t for t in doc if t.is_alpha and not t.is_space]
        total_palavras = len(palavras) if len(palavras) > 0 else 1
        
        # Total de sentenças identificadas pelo spaCy
        sentencas = list(doc.sents)
        total_sentencas = len(sentencas) if len(sentencas) > 0 else 1
        
        # Identificar conjunções subordinativas (SCONJ no padrão Universal Dependencies)
        sconjs = [t.text.lower() for t in doc if t.pos_ == "SCONJ"]
        n_sconj = len(sconjs)
        
        # Densidade sobre o total de palavras
        sconj_densidade = n_sconj / total_palavras
        # Proporção por sentença
        sconj_por_sentenca = n_sconj / total_sentencas
        
        amostra_ex = ", ".join(list(set(sconjs))[:5])
        
        registros.append({
            "id_arquivo": meta["id_arquivo"],
            "caminho_relativo": meta["caminho_relativo"],
            "nivel_ordem": meta["nivel_ordem"],
            "nivel": meta["nivel"],
            "nivel_completo": meta["nivel_completo"],
            "total_palavras": total_palavras,
            "total_sentencas": total_sentencas,
            "sconj_qtd": n_sconj,
            "sconj_densidade": sconj_densidade,
            "sconj_por_sentenca": sconj_por_sentenca,
            "amostra_sconj": amostra_ex
        })

    df = pd.DataFrame(registros)
    output_csv = "dados/features_subordinativas.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[SUCESSO] Dataset de Conjunções Subordinativas salvo em: {output_csv}")
    
    print("\nResumo das Médias por Nível de Ensino:")
    print(df.groupby("nivel_completo")[["sconj_qtd", "sconj_densidade", "sconj_por_sentenca"]].mean())
    print("\nResumo das Medianas por Nível de Ensino:")
    print(df.groupby("nivel_completo")[["sconj_qtd", "sconj_densidade", "sconj_por_sentenca"]].median())
    return df

if __name__ == "__main__":
    processar_dataset()
