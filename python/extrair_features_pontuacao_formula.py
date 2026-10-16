#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto ALFA-Gen: Extração das Features de Pontuação Expressiva ('!' + '?') e Palavra 'formula'
"""

import os
import glob
import re
import zipfile
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def contar_pontuacoes_especiais(texto: str) -> tuple:
    """
    Conta ocorrências de '!' e '?' conforme as regras:
    - Sequências consecutivas como '????' ou '!!!!' contam como apenas 1 ocorrência.
    - Sequências mistas como '?!' ou '!?' contam como 1 exclamação e 1 interrogação.
    """
    clusters = re.findall(r"[?!]+", texto)
    n_excl = sum(1 for c in clusters if "!" in c)
    n_interr = sum(1 for c in clusters if "?" in c)
    return n_excl, n_interr

def contar_palavra_formula(texto: str) -> int:
    """
    Conta ocorrências da palavra 'formula' (incluindo 'fórmula', 'fórmulas' e tokens LaTeX 'formula_1').
    """
    matches = re.findall(r"\b[fF][óÓoO]rmula\w*", texto)
    return len(matches)

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
            
            # Tokenização para total de palavras
            tokens = word_tokenize(texto, language='portuguese')
            palavras = [t for t in tokens if re.search(r"^[a-záéíóúâêôãõàüç]+$", t.lower())]
            total_palavras = len(palavras) if len(palavras) > 0 else 1
            
            # Contagens individuais de pontuação
            n_excl, n_interr = contar_pontuacoes_especiais(texto)
            # Feature unificada de pontuação expressiva/dialógica (! + ?)
            n_pont_exp = n_excl + n_interr
            
            # Contagem da palavra formula
            n_formula = contar_palavra_formula(texto)
            
            # Frequências relativas (proporção sobre total de palavras)
            prop_excl = n_excl / total_palavras
            prop_interr = n_interr / total_palavras
            prop_pont_exp = n_pont_exp / total_palavras
            prop_formula = n_formula / total_palavras
            
            registros.append({
                "id_arquivo": os.path.basename(arq),
                "caminho_relativo": arq,
                "nivel_ordem": ordem,
                "nivel": nivel_simp,
                "nivel_completo": nivel_comp,
                "total_palavras": total_palavras,
                # Pontuação Expressiva Unificada (! + ?)
                "pont_exp_qtd": n_pont_exp,
                "pont_exp_prop": prop_pont_exp,
                # Componentes individuais para referência
                "excl_qtd": n_excl,
                "excl_prop": prop_excl,
                "interr_qtd": n_interr,
                "interr_prop": prop_interr,
                # Palavra 'formula'
                "formula_qtd": n_formula,
                "formula_prop": prop_formula
            })
            
    df = pd.DataFrame(registros)
    output_csv = "dados/features_pontuacao_formula.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[SUCESSO] Dataset salvo em: {output_csv}")
    
    print("\nResumo das Médias da Pontuação Expressiva (! + ?) e Fórmula por Nível:")
    print(df.groupby("nivel_completo")[["pont_exp_qtd", "pont_exp_prop", "formula_qtd", "formula_prop"]].mean())
    return df

if __name__ == "__main__":
    processar_dataset()
