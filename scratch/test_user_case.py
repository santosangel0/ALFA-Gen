import sys
sys.path.insert(0, ".")
from modificador_sinonimos import ModificadorSinonimos

mod = ModificadorSinonimos()

frases = [
    "A qualidade do texto foi elogiada pelos professores.",
    "A qualidade principal do produto chamou atenção.",
    "Na qualidade de representante, fiz o discurso.",
    "Pela qualidade apresentada, a nota foi máxima.",
    "Uma qualidade marcante do aluno.",
    "Esta qualidade é muito importante.",
    "O trabalho foi concluído com sucesso."
]

print("=" * 80)
print("TESTE DE CONCORDÂNCIA NOMINAL DE ARTIGOS E SUBSTANTIVOS")
print("=" * 80)

for frase in frases:
    res = mod.modificar_texto(frase, intensidade=1.0, apenas_mais_frequentes=True, semente=42)
    print(f"\nOriginal : {frase}")
    print(f"Modificado: {res['texto_modificado']}")
    print("Modificações realizadas:")
    for m in res["modificacoes"]:
        motivo = m.get('motivo', '')
        print(f"  [{m['pos']}] '{m['original']}' -> '{m['substituto']}' ({motivo})")
