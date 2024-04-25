from collections import defaultdict
import pandas as pd
import numpy as np
from scipy.spatial import distance


def cosine_similarity(u, v) -> float:
    return 1 - distance.cosine(u, v)


def co_occurrence(sentences, window_size):
    # ref: https://stackoverflow.com/a/58725727/3749971
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i + 1: i + 1 + window_size]
            for t in next_token:
                key = tuple(sorted([t, token]))
                d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab)  # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab, columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df


txt = """\
Primo HIV vírus HTLV silencioso desconhecido HTLV pode causar leucemia paralisia\
"""  # noqa

df = co_occurrence([txt], 4)
hiv = df["hiv"].to_numpy()
htlv = df["htlv"].to_numpy()
# FIXME(@lerax): qua 24 abr 2024 00:56:14
# htlv should count 2, because window
# [HTLV silencioso desconhecido HTLV]
htlv[3] += 1
leucemia = df["leucemia"].to_numpy()
print(df)
print("hiv: \t", hiv)
print("htlv: \t", htlv)
print("leucemia: \t", leucemia)
print("cos(hiv, htlv) = ", cosine_similarity(hiv, htlv))
print("cos(hiv, leucemia) = ", cosine_similarity(hiv, leucemia))
print("cos(htlv, leucemia) = ", cosine_similarity(htlv, leucemia))
