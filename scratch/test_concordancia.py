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

# Criar mapeamento inverso: Masculino -> Feminino
MAPA_MASC_FEM = {v: k for k, v in MAPA_GENERO_DET.items()}


def ajustar_artigo(artigo_orig: str, genero_alvo: str) -> str:
    lower = artigo_orig.lower()
    substituto = artigo_orig
    
    if genero_alvo == "Masc" and lower in MAPA_GENERO_DET:
        substituto = MAPA_GENERO_DET[lower]
    elif genero_alvo == "Fem" and lower in MAPA_MASC_FEM:
        substituto = MAPA_MASC_FEM[lower]
        
    # Preservar capitalização
    if artigo_orig.isupper():
        return substituto.upper()
    elif artigo_orig.istitle():
        return substituto.capitalize()
    return substituto


# Testes
exemplos = [
    ("A", "Masc", "O"),
    ("a", "Masc", "o"),
    ("da", "Masc", "do"),
    ("na", "Masc", "no"),
    ("Pela", "Masc", "Pelo"),
    ("Uma", "Masc", "Um"),
    ("esta", "Masc", "este"),
    ("sua", "Masc", "seu"),
    ("O", "Fem", "A"),
    ("do", "Fem", "da"),
    ("no", "Fem", "na"),
    ("este", "Fem", "esta")
]

for orig, gen, esperado in exemplos:
    res = ajustar_artigo(orig, gen)
    print(f"{orig:8s} -> {gen:5s} => {res:8s} (Correto? {res == esperado})")
