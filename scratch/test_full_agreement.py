import spacy

nlp = spacy.load("pt_core_news_sm")

MAPA_GENERO_DET = {
    # Feminino -> Masculino
    "a": "o", "as": "os",
    "uma": "um", "umas": "uns",
    "da": "do", "das": "dos",
    "na": "no", "nas": "nos",
    "à": "ao", "às": "aos",
    "pela": "pelo", "pelas": "pelos",
    "numa": "num", "numas": "nuns",
    "duma": "dum", "dumas": "duns",
    "esta": "este", "estas": "estes",
    "essa": "esse", "essas": "esses",
    "aquela": "aquele", "aquelas": "aqueles",
    "desta": "deste", "destas": "destes",
    "dessa": "desse", "dessas": "desses",
    "daquela": "daquele", "daquelas": "daqueles",
    "nesta": "neste", "nestas": "nestes",
    "nessa": "nesse", "nessas": "nesses",
    "naquela": "naquele", "naquelas": "naqueles",
    "minha": "meu", "minhas": "meus",
    "tua": "teu", "tuas": "teus",
    "sua": "seu", "suas": "seus",
    "nossa": "nosso", "nossas": "nossos",
    "outra": "outro", "outras": "outros",
    "toda": "todo", "todas": "todos",
    "alguma": "algum", "algumas": "alguns",
    "nenhuma": "nenhum", "nenhumas": "nenhuns",
    "mesma": "mesmo", "mesmas": "mesmos",
    "própria": "próprio", "próprias": "próprios",
}
MAPA_MASC_FEM = {v: k for k, v in MAPA_GENERO_DET.items()}


def detectar_genero(palavra: str, nlp_model) -> str:
    """Detecta gênero gramatical ('Masc' ou 'Fem')."""
    doc_p = nlp_model(palavra)
    if doc_p:
        gen = doc_p[0].morph.get("Gender")
        if gen:
            return gen[0]
    p_lower = palavra.lower()
    if p_lower.endswith(("a", "as", "ção", "ções", "são", "sões", "dade", "dades")):
        return "Fem"
    return "Masc"


def ajustar_artigo_determnante(artigo_orig: str, genero_alvo: str) -> str:
    lower = artigo_orig.lower()
    substituto = artigo_orig
    
    if genero_alvo == "Masc" and lower in MAPA_GENERO_DET:
        substituto = MAPA_GENERO_DET[lower]
    elif genero_alvo == "Fem" and lower in MAPA_MASC_FEM:
        substituto = MAPA_MASC_FEM[lower]
        
    if artigo_orig.isupper() and len(artigo_orig) > 1:
        return substituto.upper()
    elif artigo_orig.istitle():
        return substituto.capitalize()
    return substituto


# Simular teste
doc = nlp("A qualidade do texto foi excelente.")
tokens_res = [t.text for t in doc]

# Suponha que trocamos 'qualidade' (token 1) por 'atributo'
tok_noun = doc[1]
subst = "atributo"
gen_subst = detectar_genero(subst, nlp)

print(f"Substantivo original: {tok_noun.text} -> Novo: {subst} (Gênero alvo: {gen_subst})")

# Procurar determinantes
dets_para_ajustar = []
for child in tok_noun.children:
    if child.i < tok_noun.i and (child.dep_ in ("det", "case") or child.pos_ in ("DET", "ADP")):
        dets_para_ajustar.append(child)

if not dets_para_ajustar and tok_noun.i > 0:
    prev = doc[tok_noun.i - 1]
    if prev.pos_ in ("DET", "ADP") or prev.lower_ in MAPA_GENERO_DET or prev.lower_ in MAPA_MASC_FEM:
        dets_para_ajustar.append(prev)

for det in dets_para_ajustar:
    det_ajustado = ajustar_artigo_determnante(det.text, gen_subst)
    print(f"  Ajustando artigo no índice {det.i}: '{det.text}' -> '{det_ajustado}'")
    tokens_res[det.i] = det_ajustado

tokens_res[tok_noun.i] = subst

print("Resultado final montado:")
print(" ".join(tokens_res))
