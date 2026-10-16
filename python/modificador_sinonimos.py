"""
================================================================================
PROJETO ALFA-GEN: MOTOR DE MODIFICAÇÃO DE TEXTOS POR SINÔNIMOS FREQUENTES
================================================================================
Substitui termos lexicais por seus sinônimos mais comuns (maior frequência Zipf
na Língua Portuguesa), preservando estritamente:
  - Fórmulas matemáticas e notações em LaTeX ($...$, formula_1, equações)
  - Notações de planilhas e células (=A2+1, "a5")
  - Números, porcentagens, valores monetários e unidades de medida
  - Nomes próprios (PROPN) e entidades nomeadas
  - Estrutura de tópicos, itens de questões ((A), 1., etc.) e pontuação
================================================================================
"""

import re
import random
from typing import Dict, List, Tuple, Any, Optional
import spacy
import wordfreq
from nltk.corpus import wordnet as wn

# Mapeamento spaCy POS -> WordNet POS
SPACY_TO_WN = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ": wn.ADJ,
    "ADV": wn.ADV
}

# Tabela Bidirecional de Concordância Nominal de Artigos, Contrações e Demonstrativos
MAPA_GENERO_DET = {
    # Artigos Definidos
    "a": "o", "as": "os",
    # Artigos Indefinidos
    "uma": "um", "umas": "uns",
    # Contrações de + artigo
    "da": "do", "das": "dos",
    # Contrações em + artigo
    "na": "no", "nas": "nos",
    # Contrações a + artigo
    "à": "ao", "às": "aos",
    # Contrações por + artigo
    "pela": "pelo", "pelas": "pelos",
    # Contrações em/de + indefinido
    "numa": "num", "numas": "nuns",
    "duma": "dum", "dumas": "duns",
    # Demonstrativos Diretos
    "esta": "este", "estas": "estes",
    "essa": "esse", "essas": "esses",
    "aquela": "aquele", "aquelas": "aqueles",
    # Demonstrativos com de
    "desta": "deste", "destas": "destes",
    "dessa": "desse", "dessas": "desses",
    "daquela": "daquele", "daquelas": "daqueles",
    # Demonstrativos com em
    "nesta": "neste", "nestas": "nestes",
    "nessa": "nesse", "nessas": "nesses",
    "naquela": "naquele", "naquelas": "naqueles",
    # Possessivos
    "minha": "meu", "minhas": "meus",
    "tua": "teu", "tuas": "teus",
    "sua": "seu", "suas": "seus",
    "nossa": "nosso", "nossas": "nossos",
    # Quantificadores e Indefinidos
    "outra": "outro", "outras": "outros",
    "toda": "todo", "todas": "todos",
    "alguma": "algum", "algumas": "alguns",
    "nenhuma": "nenhum", "nenhumas": "nenhuns",
    "mesma": "mesmo", "mesmas": "mesmos",
    "própria": "próprio", "próprias": "próprios",
}

# Inverso (Masculino -> Feminino)
MAPA_MASC_FEM = {v: k for k, v in MAPA_GENERO_DET.items()}

# Adjetivos com formas irregulares na flexão de gênero
ADJ_IRREGULARES = {
    "bom": "boa", "bons": "boas",
    "boa": "bom", "boas": "bons",
    "mau": "má", "maus": "más",
    "má": "mau", "más": "maus"
}

# Termos Técnicos, Científicos e Disciplinares Estritamente Preservados em Provas
# Garante que enunciados de Matemática, Física, Química e Biologia mantenham o rigor conceitual
TERMOS_TECNICOS_PRESERVADOS = {
    # Matemática e Geometria
    "matriz", "matrizes", "coluna", "colunas", "linha", "linhas", "vetor", "vetores",
    "equacao", "equacoes", "formula", "formulas", "funcao", "funcoes", "grafico", "graficos",
    "variavel", "variaveis", "triangulo", "triangulos", "vertice", "vertices", "angulo", "angulos",
    "aresta", "arestas", "face", "faces", "raio", "diametro", "cateto", "catetos", "hipotenusa",
    "logaritmo", "exponencial", "derivada", "integral", "limite", "polinomio", "fracao", "fracoes",
    "denominador", "numerador", "coeficiente", "grau", "seno", "cosseno", "tangente",
    "mediana", "media", "moda", "probabilidade", "estatistica", "porcentagem", "ordem",
    # Física e Química
    "massa", "volume", "densidade", "velocidade", "aceleracao", "forca", "energia", "potencia",
    "pressao", "temperatura", "tensao", "corrente", "voltagem", "mol", "atomo", "atomos",
    "molecula", "moleculas", "eletron", "eletrons", "proton", "protons", "neutron", "neutrons",
    "reacao", "solucao", "soluto", "solvente", "acido", "base", "oxido", "sal",
    # Biologia
    "celula", "celulas", "tecido", "tecidos", "orgao", "orgaos", "gene", "genes", "dna", "rna",
    "cromossomo", "cromossomos", "fotossintese", "osmose", "mitose", "meiose", "evolucao"
}

# Dicionário Curado Complementar de Sinônimos Comuns em Língua Portuguesa
# Garante alta qualidade e riqueza lexical especialmente no contexto educacional/provas
SINONIMOS_CURADOS_PT = {
    # Substantivos Educacionais e Cotidianos
    "aluno": ["estudante", "discente", "aprendiz"],
    "estudante": ["aluno", "discente", "aprendiz"],
    "professor": ["docente", "educador", "mestre", "instrutor"],
    "docente": ["professor", "educador", "instrutor"],
    "escola": ["colégio", "instituição", "educandário"],
    "colégio": ["escola", "instituição"],
    "criança": ["menino", "garoto", "infante"],
    "garoto": ["menino", "criança", "jovem"],
    "menino": ["garoto", "criança", "jovem"],
    "coisa": ["objeto", "elemento", "item", "aspecto"],
    "coisas": ["itens", "elementos", "aspectos", "fatores"],
    "gente": ["pessoas", "indivíduos", "humanos"],
    "fato": ["acontecimento", "evento", "ocorrência", "episódio"],
    "casa": ["moradia", "lar", "residência", "habitação"],
    "residência": ["casa", "moradia", "lar", "habitação"],
    "moradia": ["casa", "lar", "residência"],
    "lar": ["casa", "moradia", "residência"],
    "cidade": ["município", "metrópole", "povoado"],
    "município": ["cidade", "localidade"],
    "livro": ["obra", "texto", "volume", "publicação"],
    "texto": ["artigo", "escrito", "obra", "documento"],
    "exemplo": ["modelo", "ilustração", "padrão", "caso"],
    "trabalho": ["atividade", "tarefa", "ocupação", "ofício"],
    "tarefa": ["trabalho", "atividade", "dever"],
    "problema": ["dificuldade", "questão", "impasse", "obstáculo"],
    "questão": ["problema", "pergunta", "assunto", "tema"],
    "tema": ["assunto", "tópico", "questão", "matéria"],
    "assunto": ["tema", "tópico", "questão", "conteúdo"],
    "recurso": ["meio", "ferramenta", "mecanismo", "dispositivo"],
    "método": ["processo", "procedimento", "técnica", "maneira"],
    "processo": ["método", "procedimento", "curso", "andamento"],
    "resultado": ["efeito", "consequência", "produto", "desfecho"],
    "consequência": ["resultado", "efeito", "desfecho", "repercussão"],
    "diferença": ["distinção", "variação", "divergência"],
    "objetivo": ["finalidade", "meta", "propósito", "alvo"],
    "meta": ["objetivo", "finalidade", "alvo"],
    "período": ["época", "fase", "etapa", "momento"],
    "etapa": ["fase", "estágio", "período", "passo"],
    "região": ["área", "zona", "território", "local"],
    "área": ["região", "zona", "campo", "espaço"],
    "planta": ["vegetal", "espécie", "flora"],
    "animal": ["bicho", "espécie", "criatura"],
    "grupo": ["conjunto", "equipe", "turma", "categoria"],
    "conjunto": ["grupo", "coleção", "reunião"],
    
    # Adjetivos Comuns
    "grande": ["amplo", "extenso", "imenso", "vasto"],
    "pequeno": ["reduzido", "curto", "modesto", "baixo"],
    "bonito": ["belo", "lindo", "atraente", "agradável"],
    "belo": ["bonito", "lindo", "atraente"],
    "lindo": ["bonito", "belo", "maravilhoso"],
    "rápido": ["veloz", "ligeiro", "ágil", "pronto"],
    "veloz": ["rápido", "ligeiro", "ágil"],
    "difícil": ["complexo", "complicado", "árduo", "trabalhoso"],
    "complexo": ["difícil", "complicado", "intrincado"],
    "fácil": ["simples", "acessível", "elementar"],
    "simples": ["fácil", "elementar", "modesto", "puro"],
    "importante": ["fundamental", "relevante", "essencial", "significativo"],
    "fundamental": ["importante", "essencial", "básico", "primordial"],
    "essencial": ["fundamental", "indispensável", "importante"],
    "correto": ["certo", "exato", "adequado", "justo"],
    "certo": ["correto", "exato", "seguro"],
    "antigo": ["velho", "remoto", "primitivo"],
    "novo": ["recente", "inédito", "atual", "moderno"],
    "moderno": ["novo", "recente", "atual"],
    "claro": ["evidente", "nítido", "óbvio", "compreensível"],
    "escuro": ["sombrio", "tenebroso", "apagado"],
    "forte": ["potente", "resistente", "intenso", "vigoroso"],
    "fraco": ["frágil", "débil", "pequeno", "brando"],
    
    # Verbos Comuns
    "começar": ["iniciar", "principiar", "abrir"],
    "iniciar": ["começar", "abrir", "lançar"],
    "terminar": ["concluir", "finalizar", "acabar", "encerrar"],
    "concluir": ["terminar", "finalizar", "encerrar", "deduzir"],
    "finalizar": ["terminar", "concluir", "acabar"],
    "mostrar": ["apresentar", "exibir", "demonstrar", "indicar"],
    "apresentar": ["mostrar", "exibir", "propor", "introduzir"],
    "demonstrar": ["comprovar", "provar", "mostrar", "evidenciar"],
    "ajudar": ["auxiliar", "apoiar", "socorrer", "contribuir"],
    "auxiliar": ["ajudar", "apoiar", "assistir"],
    "fazer": ["realizar", "construir", "produzir", "efetuar"],
    "realizar": ["fazer", "executar", "efetuar", "cumprir"],
    "executar": ["realizar", "fazer", "cumprir"],
    "olhar": ["observar", "mirar", "ver", "contemplar"],
    "observar": ["notar", "olhar", "reparar", "verificar"],
    "notar": ["perceber", "observar", "reparar", "constatar"],
    "perceber": ["notar", "entender", "compreender", "sentir"],
    "entender": ["compreender", "assimilar", "captar"],
    "compreender": ["entender", "assimilar", "abarcar"],
    "encontrar": ["achar", "descobrir", "localizar"],
    "achar": ["encontrar", "julgar", "acreditar", "pensar"],
    "buscar": ["procurar", "pesquisar", "tentar"],
    "procurar": ["buscar", "pesquisar", "tentar"],
    "obter": ["conseguir", "adquirir", "alcançar"],
    "conseguir": ["obter", "alcançar", "atingir"],
    "mudar": ["alterar", "modificar", "transformar", "trocar"],
    "modificar": ["alterar", "mudar", "transformar"],
    "alterar": ["modificar", "mudar", "variar"],
    "usar": ["utilizar", "empregar", "aplicar"],
    "utilizar": ["usar", "empregar", "adotar"],
    "produzir": ["gerar", "criar", "fabricar", "originar"],
    "gerar": ["produzir", "criar", "originar", "causar"],
    "causar": ["provocar", "gerar", "ocasionar", "motivar"],
    "permitir": ["possibilitar", "autorizar", "facilitar", "consentir"],
    "possibilitar": ["permitir", "facilitar", "proporcionar"],
    
    # Advérbios Comuns
    "rapidamente": ["velozmente", "ligeiramente", "depressa"],
    "sempre": ["constantemente", "continuamente"],
    "nunca": ["jamais"],
    "talvez": ["quiçá", "porventura"],
    "perto": ["próximo", "junto"],
    "longe": ["distante", "afastado"]
}


class ModificadorSinonimos:
    def __init__(self):
        # Carregar spaCy
        self.nlp = spacy.load("pt_core_news_sm", disable=["ner"])

    @staticmethod
    def ajustar_inflexao(palavra_orig: str, sinonimo: str, pos: str) -> str:
        """
        Ajusta flexão de gênero, número e tempo verbal para preservar a coesão gramatical.
        """
        if not sinonimo or not palavra_orig:
            return sinonimo
            
        orig_lower = palavra_orig.lower()
        sin_lower = sinonimo.lower()

        # 1. Ajuste para Adjetivos (Gênero e Número)
        if pos == "ADJ":
            res = sin_lower
            # Feminino
            if orig_lower.endswith("a") and not orig_lower.endswith(("ma", "ia")):
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
            return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)

        # 1b. Ajuste para Substantivos (Apenas pares biformes e Número)
        if pos == "NOUN":
            res = sin_lower
            # Apenas substantivos biformes animados mudam -o para -a
            if orig_lower.endswith("a") and res in ("menino", "garoto", "filho", "aluno", "amigo", "tio", "gato", "cachorro", "parceiro"):
                res = res[:-1] + "a"

            # Pluralização
            if orig_lower.endswith("s") and not res.endswith("s"):
                if res.endswith(("a", "e", "i", "o", "u")):
                    res = res + "s"
                elif res.endswith(("r", "s", "z")):
                    res = res + "es"
                elif res.endswith("m"):
                    res = res[:-1] + "ns"
                elif res.endswith("ão"):
                    res = res[:-2] + "ões"
            return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)

        # 2. Ajuste para Verbos
        if pos == "VERB" and sin_lower.endswith(("ar", "er", "ir")):
            radical_sin = sin_lower[:-2]
            vt = sin_lower[-2]  # vogal temática

            if orig_lower.endswith(("ando", "endo", "indo")):
                res = radical_sin + ("ando" if vt == "a" else ("endo" if vt == "e" else "indo"))
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)
            if orig_lower.endswith(("ado", "ido")):
                res = radical_sin + ("ado" if vt == "a" else "ido")
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)
            if orig_lower.endswith(("ou", "eu", "iu")):
                res = radical_sin + ("ou" if vt == "a" else ("eu" if vt == "e" else "iu"))
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)
            if orig_lower.endswith("a") and vt == "a":
                res = radical_sin + "a"
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)
            if orig_lower.endswith("e") and vt in ("e", "ir"):
                res = radical_sin + "e"
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)
            if orig_lower.endswith(("ava", "ia")):
                res = radical_sin + ("ava" if vt == "a" else "ia")
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)
            if orig_lower.endswith(("aram", "eram", "iram")):
                res = radical_sin + ("aram" if vt == "a" else ("eram" if vt == "e" else "iram"))
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)
            if orig_lower.endswith(("ará", "erá", "irá")):
                res = radical_sin + ("ará" if vt == "a" else ("erá" if vt == "e" else "irá"))
                return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, res)

        return ModificadorSinonimos.aplicar_capitalizacao(palavra_orig, sin_lower)

    @staticmethod
    def aplicar_capitalizacao(original: str, substituto: str) -> str:
        """Mantém maiúsculas/minúsculas do termo original."""
        if original.isupper() and len(original) > 1:
            return substituto.upper()
        elif original.istitle():
            return substituto.capitalize()
        return substituto

    def detectar_genero(self, palavra: str) -> str:
        """
        Detecta o gênero gramatical ('Masc' ou 'Fem') de um termo em língua portuguesa
        usando o analisador morfológico do spaCy e regras fonológico-morfossemânticas.
        """
        doc_p = self.nlp(palavra)
        if len(doc_p) > 0:
            gen = doc_p[0].morph.get("Gender")
            if gen:
                return gen[0]
        p_lower = palavra.lower()
        if p_lower.endswith(("a", "as", "ção", "ções", "são", "sões", "dade", "dades", "ência", "ências", "ância", "âncias", "tude")):
            return "Fem"
        return "Masc"

    @staticmethod
    def ajustar_artigo_determinante(artigo_orig: str, genero_alvo: str) -> str:
        """
        Ajusta artigos definidos, indefinidos, contrações e demonstrativos conforme o gênero alvo.
        """
        lower = artigo_orig.lower()
        substituto = artigo_orig
        if genero_alvo == "Masc" and lower in MAPA_GENERO_DET:
            substituto = MAPA_GENERO_DET[lower]
        elif genero_alvo == "Fem" and lower in MAPA_MASC_FEM:
            substituto = MAPA_MASC_FEM[lower]

        return ModificadorSinonimos.aplicar_capitalizacao(artigo_orig, substituto)

    @staticmethod
    def ajustar_adjetivo_genero(adj_orig: str, genero_alvo: str) -> str:
        """
        Ajusta adjetivos regulares modificadores conforme a mudança de gênero do substantivo.
        """
        lower = adj_orig.lower()
        if lower in ADJ_IRREGULARES:
            subst = ADJ_IRREGULARES[lower]
            if genero_alvo == "Masc" and lower in ("boa", "boas", "má", "más"):
                return ModificadorSinonimos.aplicar_capitalizacao(adj_orig, subst)
            elif genero_alvo == "Fem" and lower in ("bom", "bons", "mau", "maus"):
                return ModificadorSinonimos.aplicar_capitalizacao(adj_orig, subst)
            return adj_orig

        # Adjetivos uniformes em gênero (não mudam: grande, forte, excelente, fácil, etc.)
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

        return ModificadorSinonimos.aplicar_capitalizacao(adj_orig, res)

    def proteger_estruturas(self, texto: str) -> Tuple[str, Dict[str, str]]:
        """
        Isola fórmulas, referências numéricas, equações e tokens protegidos
        substituindo-os por marcadores temporários invioláveis.
        """
        mapa_protecao: Dict[str, str] = {}
        contador = 0

        def guardar_token(match):
            nonlocal contador
            token_original = match.group(0)
            marcador = f"<<<PROT_{contador}>>>"
            mapa_protecao[marcador] = token_original
            contador += 1
            return marcador

        texto_protegido = texto

        # 1. Fórmulas LaTeX em bloco e inline
        texto_protegido = re.sub(r"\$\$[\s\S]+?\$\$", guardar_token, texto_protegido)
        texto_protegido = re.sub(r"\$[^\$\n]+?\$", guardar_token, texto_protegido)
        texto_protegido = re.sub(r"\\\(.+?\\\)|\ custody\\\[.+?\\\]", guardar_token, texto_protegido)

        # 2. Identificadores explícitos de fórmula (formula_1, formula_2, etc.)
        texto_protegido = re.sub(r"\b[fF][óÓoO]rmula_\w+\b", guardar_token, texto_protegido)
        texto_protegido = re.sub(r"\bformula_\d+\b", guardar_token, texto_protegido)

        # 3. Equações de planilhas e células ("=A2+1", "a5", "=SOMA(A1:B10)")
        texto_protegido = re.sub(r'"?=[A-Za-z0-9\+\-\*\/\(\)\:\,\;\s]+"?', guardar_token, texto_protegido)
        texto_protegido = re.sub(r'"[A-Za-z][0-9]+"', guardar_token, texto_protegido)

        # 4. URLs e emails
        texto_protegido = re.sub(r"https?://\S+|www\.\S+|\S+@\S+\.\S+", guardar_token, texto_protegido)

        # 5. Citações parentéticas ABNT ((Silva, 2018), et al.)
        texto_protegido = re.sub(r"\([A-ZÀ-Ú][a-zà-úA-ZÀ-Ú\s\.,;&\-]+,?\s*(?:19|20)\d{2}[a-z]?\)", guardar_token, texto_protegido)
        texto_protegido = re.sub(r"\b(?:et\s+al\.?|apud|ibid\.?|op\.\s*cit\.?)\b", guardar_token, texto_protegido)

        # 6. Marcadores de alternativa em provas: (A), (B), a), 1., etc. no início de linha
        texto_protegido = re.sub(r"^\s*(?:\([a-eA-E0-9]\)|[a-eA-E0-9]\))\s*", guardar_token, texto_protegido, flags=re.MULTILINE)

        # 7. Valores com unidades e porcentagens (ex: 50%, R$ 100,00, 20 km/h)
        texto_protegido = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:%|km/h|kg|m|cm|mm|km|reais|R\$|anos|horas|minutos|°C|ºC)\b", guardar_token, texto_protegido)

        return texto_protegido, mapa_protecao

    def desproteger_estruturas(self, texto_modificado: str, mapa_protecao: Dict[str, str]) -> str:
        """Restaura as fórmulas e tokens originais intactos."""
        res = texto_modificado
        for marcador, token_original in mapa_protecao.items():
            res = res.replace(marcador, token_original)
        return res

    def buscar_sinonimos_candidatos(self, palavra: str, lemma: str, pos: str) -> List[Tuple[str, float]]:
        """
        Busca sinônimos no WordNet (OpenWN-PT) e no dicionário curado.
        Retorna lista de tuplas (sinonimo, frequencia_zipf_pt) ordenada
        pela maior frequência na língua portuguesa.
        """
        candidatos = set()
        pal_lower = palavra.lower()
        lem_lower = lemma.lower()

        # 1. Dicionário Curado (alta confiabilidade)
        for termo in (pal_lower, lem_lower):
            if termo in SINONIMOS_CURADOS_PT:
                for sin in SINONIMOS_CURADOS_PT[termo]:
                    if sin.lower() != pal_lower and sin.lower() != lem_lower:
                        candidatos.add(sin.lower())

        # 2. WordNet em Português (OMW)
        if pos in SPACY_TO_WN:
            wn_pos = SPACY_TO_WN[pos]
            synsets = wn.synsets(lem_lower, pos=wn_pos, lang="por")
            if not synsets:
                synsets = wn.synsets(pal_lower, pos=wn_pos, lang="por")

            for s in synsets:
                for l in s.lemmas("por"):
                    cand = l.name().lower().replace("_", " ")
                    # Apenas termos de uma palavra, alfabéticos, distintos do original
                    if " " not in cand and cand != pal_lower and cand != lem_lower and len(cand) >= 3:
                        if re.match(r"^[a-záéíóúâêôãõàüç]+$", cand):
                            candidatos.add(cand)

        if not candidatos:
            return []

        # 3. Ranqueamento por Frequência Zipf em Língua Portuguesa
        ranqueados = []
        for cand in candidatos:
            freq = wordfreq.zipf_frequency(cand, "pt")
            ranqueados.append((cand, freq))

        # Ordenar do mais comum para o menos comum
        ranqueados.sort(key=lambda x: -x[1])
        return ranqueados

    def modificar_texto(
        self,
        texto: str,
        intensidade: float = 0.40,
        apenas_mais_frequentes: bool = True,
        freq_minima: float = 3.0,
        semente: Optional[int] = 42
    ) -> Dict[str, Any]:
        """
        Executa a paráfrase lexical orientada por frequência na língua portuguesa.
        
        Args:
            texto: Texto de entrada (original do banco de dados ou digitado pelo professor).
            intensidade: Proporção de palavras candidatas a substituir (0.1 a 0.9).
            apenas_mais_frequentes: Se True, seleciona o sinônimo com maior Zipf score.
            freq_minima: Frequência mínima Zipf para aceitar a substituição (evita termos estranhos).
            semente: Semente aleatória para reprodutibilidade.
        """
        if semente is not None:
            random.seed(semente)

        # 1. Proteger Fórmulas e Notações Especiais
        texto_protegido, mapa_protecao = self.proteger_estruturas(texto)

        # 2. Análise spaCy
        doc = self.nlp(texto_protegido)

        modificacoes = []
        substituicoes: Dict[int, str] = {}
        
        palavras_elegiveis = 0
        palavras_substituidas = 0

        for token in doc:
            tok_text = token.text

            # Pular se já foi modificado (por exemplo, por concordância com substantivo anterior)
            if token.i in substituicoes:
                continue

            # Pular se for token protegido
            if tok_text.startswith("<<<PROT_") or tok_text.endswith(">>>"):
                continue

            # Pular pontuações, números, stopwords, espaços, nomes próprios ou termos técnicos
            if (
                token.is_punct
                or token.is_space
                or token.like_num
                or token.is_stop
                or token.pos_ == "PROPN"
                or len(tok_text) < 3
                or not token.is_alpha
                or token.text.lower() in TERMOS_TECNICOS_PRESERVADOS
                or token.lemma_.lower() in TERMOS_TECNICOS_PRESERVADOS
            ):
                continue

            # Verificar se é classe gramatical substituível
            if token.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
                palavras_elegiveis += 1

                # Decisão estocástica ponderada pela intensidade
                if random.random() <= intensidade:
                    candidatos = self.buscar_sinonimos_candidatos(tok_text, token.lemma_, token.pos_)
                    
                    if candidatos:
                        # Filtrar pela frequência mínima se solicitado
                        if apenas_mais_frequentes:
                            candidatos_validos = [c for c in candidatos if c[1] >= freq_minima]
                        else:
                            candidatos_validos = candidatos

                        if candidatos_validos:
                            # O sinônimo mais comum (com maior frequência na língua portuguesa)
                            melhor_sinonimo, freq_sin = candidatos_validos[0]
                            
                            # Ajustar flexão de gênero/número/tempo do próprio sinônimo
                            sin_ajustado = self.ajustar_inflexao(tok_text, melhor_sinonimo, token.pos_)
                            
                            freq_orig = wordfreq.zipf_frequency(tok_text.lower(), "pt")

                            substituicoes[token.i] = sin_ajustado
                            palavras_substituidas += 1

                            modificacoes.append({
                                "original": tok_text,
                                "substituto": sin_ajustado,
                                "pos": token.pos_,
                                "freq_original": round(freq_orig, 2),
                                "freq_substituto": round(freq_sin, 2),
                                "ganho_frequencia": round(freq_sin - freq_orig, 2),
                                "motivo": "Sinônimo mais frequente"
                            })

                            # CONCORDÂNCIA NOMINAL: Se for substantivo, ajustar artigos e adjetivos relacionados
                            if token.pos_ == "NOUN":
                                gen_orig = self.detectar_genero(tok_text)
                                gen_alvo = self.detectar_genero(sin_ajustado)

                                if gen_orig != gen_alvo:
                                    # 1. Identificar artigos e determinantes vinculados a este substantivo
                                    dets_candidatos = []
                                    # Filhos sintáticos à esquerda (ex: 'A', 'Uma', 'Esta', 'Da', 'Na', 'Pela')
                                    for child in token.children:
                                        if child.i < token.i and (child.dep_ in ("det", "case") or child.pos_ in ("DET", "ADP")):
                                            dets_candidatos.append(child)

                                    # Tokens anteriores imediatos na janela sintagmática (fallback de adjacência)
                                    for delta_i in (1, 2):
                                        if token.i - delta_i >= 0:
                                            cand_prev = doc[token.i - delta_i]
                                            if cand_prev.lower_ in MAPA_GENERO_DET or cand_prev.lower_ in MAPA_MASC_FEM:
                                                if cand_prev not in dets_candidatos:
                                                    dets_candidatos.append(cand_prev)

                                    for det_tok in dets_candidatos:
                                        if det_tok.i not in substituicoes:
                                            det_novo = self.ajustar_artigo_determinante(det_tok.text, gen_alvo)
                                            if det_novo != det_tok.text:
                                                substituicoes[det_tok.i] = det_novo
                                                f_orig_d = wordfreq.zipf_frequency(det_tok.text.lower(), "pt")
                                                f_novo_d = wordfreq.zipf_frequency(det_novo.lower(), "pt")
                                                modificacoes.append({
                                                    "original": det_tok.text,
                                                    "substituto": det_novo,
                                                    "pos": "DET",
                                                    "freq_original": round(f_orig_d, 2),
                                                    "freq_substituto": round(f_novo_d, 2),
                                                    "ganho_frequencia": 0.0,
                                                    "motivo": f"Concordância com '{sin_ajustado}' ({gen_alvo})"
                                                })

                                    # 2. Identificar adjetivos modificadores diretos vinculados ao substantivo
                                    for child in token.children:
                                        if (child.dep_ in ("amod", "nummod") or child.pos_ == "ADJ") and child.i not in substituicoes:
                                            adj_novo = self.ajustar_adjetivo_genero(child.text, gen_alvo)
                                            if adj_novo != child.text:
                                                substituicoes[child.i] = adj_novo
                                                f_orig_a = wordfreq.zipf_frequency(child.text.lower(), "pt")
                                                f_novo_a = wordfreq.zipf_frequency(adj_novo.lower(), "pt")
                                                modificacoes.append({
                                                    "original": child.text,
                                                    "substituto": adj_novo,
                                                    "pos": "ADJ",
                                                    "freq_original": round(f_orig_a, 2),
                                                    "freq_substituto": round(f_novo_a, 2),
                                                    "ganho_frequencia": 0.0,
                                                    "motivo": f"Concordância com '{sin_ajustado}' ({gen_alvo})"
                                                })

        # Reconstrução do texto preservando espaços e pontuação originais
        tokens_resultado = []
        for token in doc:
            tok_final = substituicoes.get(token.i, token.text)
            tokens_resultado.append(tok_final + token.whitespace_)

        texto_reconstruido = "".join(tokens_resultado)

        # 3. Desproteger Fórmulas e Notações
        texto_final = self.desproteger_estruturas(texto_reconstruido, mapa_protecao)

        taxa_real = (palavras_substituidas / max(palavras_elegiveis, 1)) * 100.0

        return {
            "texto_original": texto,
            "texto_modificado": texto_final,
            "modificacoes": modificacoes,
            "total_modificacoes": len(modificacoes),
            "palavras_elegiveis": palavras_elegiveis,
            "taxa_substituicao_pct": round(taxa_real, 1),
            "formulas_preservadas": len(mapa_protecao)
        }
