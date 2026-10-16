import sys
sys.path.insert(0, ".")
import glob
import zipfile
import pandas as pd
from extrator_features import ExtratorFeatures
from classificador_onnx import ClassificadorONNX
from modificador_sinonimos import ModificadorSinonimos

extrator = ExtratorFeatures()
classificador = ClassificadorONNX()
modificador = ModificadorSinonimos()

zip_path = glob.glob("dados/*.zip")[0]

exemplos = [
    ("Fundamental I", "dataset_caed/1_Ensino_Fundamental_I/195_753.txt"),
    ("Ensino Médio", "dataset_caed/3_Ensino_Medio/96_824.txt"),
    ("Com Fórmula", "dataset_caed/3_Ensino_Medio/170_1531.txt")
]

with zipfile.ZipFile(zip_path, 'r') as z:
    for nome_exemplo, caminho in exemplos:
        print("\n" + "="*80)
        print(f"TESTE: {nome_exemplo} ({caminho})")
        print("="*80)
        
        texto_orig = z.read(caminho).decode("utf-8")
        print(f"Primeiros 200 caracteres originais:\n{texto_orig[:200]}...\n")
        
        # 1. Classificar Original
        feats_orig = extrator.extrair_features(texto_orig)
        pred_orig = classificador.prever(feats_orig, "LightGBM")
        print(f"Predição Original (LightGBM): {pred_orig['classe_nome']} (Confiança: {pred_orig['confianca']*100:.1f}%)")
        print(f"Probabilidades: {pred_orig['probabilidades']}")
        
        # 2. Modificar Texto
        res_mod = modificador.modificar_texto(texto_orig, intensidade=0.5, apenas_mais_frequentes=True)
        texto_mod = res_mod["texto_modificado"]
        print(f"\nTotal de palavras substituídas: {res_mod['total_modificacoes']} (Taxa: {res_mod['taxa_substituicao_pct']}%)")
        print(f"Fórmulas protegidas/preservadas: {res_mod['formulas_preservadas']}")
        print("Primeiras 5 substituições:")
        for m in res_mod["modificacoes"][:5]:
            print(f"  {m['original']} -> {m['substituto']} (Zipf: {m['freq_original']} -> {m['freq_substituto']})")
            
        print(f"\nPrimeiros 200 caracteres modificados:\n{texto_mod[:200]}...\n")
        
        # 3. Classificar Modificado
        feats_mod = extrator.extrair_features(texto_mod)
        pred_mod = classificador.prever(feats_mod, "LightGBM")
        print(f"Predição Modificado (LightGBM): {pred_mod['classe_nome']} (Confiança: {pred_mod['confianca']*100:.1f}%)")
        print(f"Probabilidades: {pred_mod['probabilidades']}")
        
        # 4. Avaliar Mudança de Categoria
        mudou = pred_orig['classe_nome'] != pred_mod['classe_nome']
        print(f"\n-> Houve mudança de nível? {'SIM, ATENÇÃO! [' + pred_orig['classe_nome'] + ' -> ' + pred_mod['classe_nome'] + ']' if mudou else 'NÃO, Nível Mantido [' + pred_orig['classe_nome'] + ']'}")
