#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto ALFA-Gen: Extração da Feature de Comprimento Médio da Sentença (ASL)
"""

import os
import glob
import re
import zipfile
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

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
        
        registros = []
        for idx, arq in enumerate(arquivos_txt, start=1):
            if idx % 500 == 0 or idx == len(arquivos_txt):
                print(f"  -> Processando texto {idx}/{len(arquivos_txt)}...")
                
            conteudo_raw = z.read(arq)
            try:
                texto = conteudo_raw.decode("utf-8")
            except UnicodeDecodeError:
                texto = conteudo_raw.decode("latin-1", errors="ignore")
                
            ordem, nivel_simp, nivel_comp = extrair_nivel(arq)
            
            # Tokenização de sentenças
            sentencas = sent_tokenize(texto, language='portuguese')
            total_sentencas = len(sentencas) if len(sentencas) > 0 else 1
            
            # Tokenização de palavras
            tokens = word_tokenize(texto, language='portuguese')
            palavras = [t for t in tokens if re.search(r"^[a-záéíóúâêôãõàüç]+$", t.lower())]
            total_palavras = len(palavras) if len(palavras) > 0 else 1
            total_caracteres = sum(len(p) for p in palavras)
            
            # ASL (Average Sentence Length) em palavras por sentença
            asl = total_palavras / total_sentencas
            # ASL em caracteres por sentença
            asl_chars = total_caracteres / total_sentencas
            
            registros.append({
                "id_arquivo": os.path.basename(arq),
                "caminho_relativo": arq,
                "nivel_ordem": ordem,
                "nivel": nivel_simp,
                "nivel_completo": nivel_comp,
                "total_palavras": total_palavras,
                "total_sentencas": total_sentencas,
                "asl": asl,
                "asl_chars": asl_chars
            })
            
    df = pd.DataFrame(registros)
    output_csv = "dados/features_asl.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[SUCESSO] Dataset salvo em: {output_csv}")
    
    print("\nResumo das Médias por Nível de Ensino:")
    print(df.groupby("nivel_completo")[["total_palavras", "total_sentencas", "asl", "asl_chars"]].mean())
    print("\nResumo das Medianas por Nível de Ensino:")
    print(df.groupby("nivel_completo")[["asl", "total_sentencas"]].median())
    return df

if __name__ == "__main__":
    processar_dataset()
