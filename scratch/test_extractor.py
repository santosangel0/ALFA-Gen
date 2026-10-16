import sys
sys.path.insert(0, ".")
import glob
import zipfile
import pandas as pd
from extrator_features import ExtratorFeatures

extrator = ExtratorFeatures()

# Ler primeiro arquivo do zip
zip_path = glob.glob("dados/*.zip")[0]
with zipfile.ZipFile(zip_path, 'r') as z:
    txt = z.read("dataset_caed/3_Ensino_Medio/96_824.txt").decode("utf-8")

feats = extrator.extrair_features(txt)
print("Features extraídas para 96_824.txt:")
for k, v in feats.items():
    print(f"  {k}: {v:.6f}")

# Comparar com CSV salvo
df = pd.read_csv("dados/dataset_multivariado_features.csv")
row = df[df["id_arquivo"] == "96_824.txt"].iloc[0]
print("\nComparação com dataset_multivariado_features.csv:")
for k in feats.keys():
    v_orig = float(row[k])
    v_calc = float(feats[k])
    diff = abs(v_orig - v_calc)
    print(f"  {k:30s} -> Orig: {v_orig:.5f} | Calc: {v_calc:.5f} | Diff: {diff:.5f}")
