ADJ_IRREGULAR = {
    "bom": "boa", "bons": "boas",
    "boa": "bom", "boas": "bons",
    "mau": "má", "maus": "más",
    "má": "mau", "más": "maus"
}

def ajustar_adjetivo_genero(adj_orig: str, genero_alvo: str) -> str:
    lower = adj_orig.lower()
    
    # Irregulares
    if lower in ADJ_IRREGULAR:
        subst = ADJ_IRREGULAR[lower]
        # Verificar se está no sentido correto
        if genero_alvo == "Masc" and lower in ("boa", "boas", "má", "más"):
            return aplicar_casing(adj_orig, subst)
        elif genero_alvo == "Fem" and lower in ("bom", "bons", "mau", "maus"):
            return aplicar_casing(adj_orig, subst)
        return adj_orig

    # Uniformes em gênero (não mudam: grande, forte, fácil, difícil, superior, melhor, etc.)
    if lower.endswith(("e", "es", "l", "is", "m", "ns", "z", "zes", "or", "ores")):
        return adj_orig

    res = lower
    if genero_alvo == "Masc":
        if lower.endswith("as"):
            res = lower[:-2] + "os"
        elif lower.endswith("a") and not lower.endswith(("ia", "ma")):
            res = lower[:-1] + "o"
    elif genero_alvo == "Fem":
        if lower.endswith("os"):
            res = lower[:-2] + "as"
        elif lower.endswith("o"):
            res = lower[:-1] + "a"

    return aplicar_casing(adj_orig, res)


def aplicar_casing(orig: str, novo: str) -> str:
    if orig.isupper() and len(orig) > 1:
        return novo.upper()
    elif orig.istitle():
        return novo.capitalize()
    return novo


# Testes
exemplos_adj = [
    ("boa", "Masc", "bom"),
    ("alta", "Masc", "alto"),
    ("primeira", "Masc", "primeiro"),
    ("perfeita", "Masc", "perfeito"),
    ("grandes", "Masc", "grandes"), # Uniforme
    ("excelente", "Masc", "excelente"), # Uniforme
    ("alto", "Fem", "alta"),
    ("bom", "Fem", "boa"),
    ("primeiros", "Fem", "primeiras")
]

for orig, gen, esp in exemplos_adj:
    res = ajustar_adjetivo_genero(orig, gen)
    print(f"{orig:10s} -> {gen:5s} => {res:10s} (Correto? {res == esp})")
