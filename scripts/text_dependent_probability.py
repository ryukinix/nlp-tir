from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Fim de lei causaria apagão em musicais, orquestras e museus",
    "Musicais, orquestras e artes vêm crescendo desde 2000 com apoio de lei",
    "Musicais, orquestras e museus obtêm incentivo por meio de lei",
    "Mercado de musicais gera em média cem empregos por peça",
    "Lei incentiva às artes, porém, críticos apontam rombos de cofres públicos"
]

vectorizer = CountVectorizer(ngram_range=(1, 3))
matrix = vectorizer.fit_transform(corpus).todense()
features = vectorizer.get_feature_names_out().tolist()
term_dependency = "de"
term = "lei"
occurrence_dependency = matrix[:, features.index(term_dependency)].sum()
occurrence_term = matrix[:, features.index(term)].sum()
prob = occurrence_term / occurrence_dependency

# P(museus | musicais, orquestras) = 0.67
# P(lei | de) = 0.8 (ta errado no gabarito, n tem opção)
# P(orquestras | musicais) = 0.75
