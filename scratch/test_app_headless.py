import sys
sys.path.insert(0, ".")
import os

print("Testando imports e integridade de app.py...")
try:
    import app
    print("app.py importado com sucesso!")
except Exception as e:
    print(f"Erro ao importar app.py: {e}")
    sys.exit(1)
