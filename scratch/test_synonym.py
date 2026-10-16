import spacy
import wordfreq
from nltk.corpus import wordnet as wn

nlp = spacy.load('pt_core_news_sm')
doc = nlp('O menino encontrou uma residência bonita perto da floresta.')

spacy_to_wn = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV}

for token in doc:
    if token.pos_ in spacy_to_wn:
        wn_pos = spacy_to_wn[token.pos_]
        synsets = wn.synsets(token.lemma_, pos=wn_pos, lang='por')
        synonyms = set()
        for s in synsets:
            for l in s.lemmas('por'):
                cand = l.name().lower().replace('_', ' ')
                if ' ' not in cand and cand != token.lemma_.lower() and cand != token.text.lower():
                    synonyms.add(cand)
        ranked = sorted([(cand, wordfreq.zipf_frequency(cand, 'pt')) for cand in synonyms], key=lambda x: -x[1])
        print(f'{token.text} ({token.pos_}, lemma={token.lemma_}) -> {ranked[:3]}')
