
from functools import lru_cache
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.services.data_service import clean_df

@lru_cache(maxsize=1)
def matrix():
    df=clean_df().copy()
    fields=["listed_in","description","director","cast"]
    for f in fields: df[f]=df[f].fillna("")
    df["profile"]=(df["listed_in"]+" "+df["description"]+" "+df["director"]+" "+df["cast"]).str.lower()
    vec=TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
    return df, vec, vec.fit_transform(df["profile"])

def recommend(show_id, limit=8, type_filter=None, genre_filter=None):
    df, vec, X=matrix()
    idx=df.index[df["show_id"].astype(str).eq(str(show_id))]
    if len(idx)==0: raise KeyError("Unknown show_id")
    i=df.index.get_loc(idx[0])
    scores=cosine_similarity(X[i], X).ravel()
    candidates=df.copy()
    candidates["similarity"]=scores
    candidates=candidates[candidates["show_id"].astype(str).ne(str(show_id))]
    if type_filter: candidates=candidates[candidates["type"].eq(type_filter)]
    if genre_filter: candidates=candidates[candidates["listed_in"].fillna("").str.contains(genre_filter, case=False, regex=False)]
    candidates=candidates.sort_values("similarity", ascending=False).head(limit)
    cols=["show_id","title","type","release_year","rating","duration","listed_in","description","similarity"]
    return candidates[cols].fillna("").assign(similarity=lambda x:x["similarity"].round(4)).to_dict("records")
