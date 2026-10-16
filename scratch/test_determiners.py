import spacy

nlp = spacy.load("pt_core_news_sm")

frases = [
    "A qualidade do texto foi avaliada.",
    "Com a qualidade dos alunos, vencemos.",
    "Uma qualidade importante do trabalho.",
    "Na qualidade de professor, falo hoje.",
    "Pela qualidade do serviço, pagamos mais.",
    "Essa qualidade chamou a atenção.",
    "O atributo do texto foi avaliado."
]

for frase in frases:
    doc = nlp(frase)
    for token in doc:
        if token.pos_ == "NOUN":
            # Determinantes filhos ou imediatamente anteriores
            dets = [child for child in token.children if child.pos_ in ("DET", "ADP") or child.dep_ in ("det", "case")]
            print(f"Frase: '{frase}'")
            print(f"  Substantivo: {token.text} (pos={token.pos_}, dep={token.dep_})")
            print(f"  Children: {[(c.text, c.pos_, c.dep_) for c in token.children]}")
            # Olhar token anterior imediato
            if token.i > 0:
                prev_tok = doc[token.i - 1]
                print(f"  Token anterior: {prev_tok.text} (pos={prev_tok.pos_}, dep={prev_tok.dep_})")
            print("-" * 50)
