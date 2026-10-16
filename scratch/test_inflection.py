def ajustar_inflexao(palavra_orig: str, sinonimo: str, pos: str) -> str:
    """
    Ajusta a flexão de gênero, número e tempo verbal do sinônimo
    para coincidir com a palavra original em português.
    """
    if not sinonimo or not palavra_orig:
        return sinonimo
        
    orig_lower = palavra_orig.lower()
    sin_lower = sinonimo.lower()
    
    # 1. Ajuste para Substantivos e Adjetivos (Gênero e Número)
    if pos in ("NOUN", "ADJ"):
        res = sin_lower
        # Feminino singular
        if orig_lower.endswith("a") and not orig_lower.endswith("ma") and not orig_lower.endswith("ia"):
            if res.endswith("o"):
                res = res[:-1] + "a"
            elif res.endswith("or"):
                res = res + "a"
            elif res.endswith("ês"):
                res = res[:-2] + "esa"
                
        # Plural
        if orig_lower.endswith("s") and not res.endswith("s"):
            if orig_lower.endswith("as") and res.endswith("o"):
                res = res[:-1] + "as"
            elif orig_lower.endswith("as") and res.endswith("a"):
                res = res + "s"
            elif res.endswith(("a", "e", "i", "o", "u")):
                res = res + "s"
            elif res.endswith(("r", "s", "z")):
                res = res + "es"
            elif res.endswith("m"):
                res = res[:-1] + "ns"
            elif res.endswith("ão"):
                res = res[:-2] + "ões"
        return aplicar_capitalizacao(palavra_orig, res)

    # 2. Ajuste para Verbos
    if pos == "VERB" and sin_lower.endswith(("ar", "er", "ir")):
        radical_sin = sin_lower[:-2]
        vogal_tema = sin_lower[-2] # 'a', 'e' ou 'i'
        
        # Gerúndio (-ando, -endo, -indo)
        if orig_lower.endswith("ando") or orig_lower.endswith("endo") or orig_lower.endswith("indo"):
            res = radical_sin + ("ando" if vogal_tema == "a" else ("endo" if vogal_tema == "e" else "indo"))
            return aplicar_capitalizacao(palavra_orig, res)
            
        # Particípio (-ado, -ido)
        if orig_lower.endswith("ado") or orig_lower.endswith("ido"):
            res = radical_sin + ("ado" if vogal_tema == "a" else "ido")
            return aplicar_capitalizacao(palavra_orig, res)
            
        # Pretérito Perfeito 3s (-ou, -eu, -iu)
        if orig_lower.endswith("ou") or orig_lower.endswith("eu") or orig_lower.endswith("iu"):
            res = radical_sin + ("ou" if vogal_tema == "a" else ("eu" if vogal_tema == "e" else "iu"))
            return aplicar_capitalizacao(palavra_orig, res)
            
        # Presente do Indicativo 3s (-a, -e)
        if orig_lower.endswith("a") and vogal_tema == "a":
            res = radical_sin + "a"
            return aplicar_capitalizacao(palavra_orig, res)
        if orig_lower.endswith("e") and vogal_tema in ("e", "i"):
            res = radical_sin + "e"
            return aplicar_capitalizacao(palavra_orig, res)
            
        # Pretérito Imperfeito 3s (-ava, -ia)
        if orig_lower.endswith("ava") or orig_lower.endswith("ia"):
            res = radical_sin + ("ava" if vogal_tema == "a" else "ia")
            return aplicar_capitalizacao(palavra_orig, res)
            
        # Plurais do Pretérito Perfeito (-aram, -eram, -iram)
        if orig_lower.endswith("aram") or orig_lower.endswith("eram") or orig_lower.endswith("iram"):
            res = radical_sin + ("aram" if vogal_tema == "a" else ("eram" if vogal_tema == "e" else "iram"))
            return aplicar_capitalizacao(palavra_orig, res)
            
        # Futuro do Presente 3s (-ará, -erá, -irá)
        if orig_lower.endswith("ará") or orig_lower.endswith("erá") or orig_lower.endswith("irá"):
            res = radical_sin + ("ará" if vogal_tema == "a" else ("erá" if vogal_tema == "e" else "irá"))
            return aplicar_capitalizacao(palavra_orig, res)

    return aplicar_capitalizacao(palavra_orig, sin_lower)


def aplicar_capitalizacao(original: str, substituto: str) -> str:
    """Mantém a caixa alta/baixa do termo original."""
    if original.isupper() and len(original) > 1:
        return substituto.upper()
    elif original.istitle():
        return substituto.capitalize()
    return substituto


# Testes
exemplos = [
    ("bonita", "lindo", "ADJ"),
    ("casas", "moradia", "NOUN"),
    ("encontrou", "achar", "VERB"),
    ("procuravam", "buscar", "VERB"),
    ("estudará", "analisar", "VERB"),
    ("CRIANÇAS", "menino", "NOUN"),
    ("florestas", "mata", "NOUN")
]

for orig, sin, pos in exemplos:
    ajustado = ajustar_inflexao(orig, sin, pos)
    print(f"{orig} + {sin} ({pos}) -> {ajustado}")
