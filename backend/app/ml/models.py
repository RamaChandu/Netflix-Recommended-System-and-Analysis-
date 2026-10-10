
from __future__ import annotations
import time, hashlib, json
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from app.services.data_service import clean_df

MODELS={}
RESULTS=[]

def _features(df, task):
    d=df.copy()
    d["genre_primary"]=d["listed_in"].fillna("").str.split(",").str[0]
    d["country_primary"]=d["country"].fillna("").str.split(",").str[0]
    d["rating_clean"]=d["rating"].fillna("Unknown")
    if task=="classification":
        X=d[["release_year","rating_clean","genre_primary","country_primary","duration_value"]]
        y=d["type"]
        numeric=["release_year","duration_value"]; categorical=["rating_clean","genre_primary","country_primary"]
    else:
        X=d[["type","rating_clean","genre_primary","country_primary","duration_value"]]
        y=d["release_year"]
        numeric=["duration_value"]; categorical=["type","rating_clean","genre_primary","country_primary"]
    return X,y,numeric,categorical

def train(task, algorithm):
    df=clean_df()
    X,y,numeric,categorical=_features(df,task)
    strat=y if task=="classification" else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
    prep=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)
    ])
    if task=="classification":
        choices={"logistic_regression":LogisticRegression(max_iter=2000),
                 "decision_tree":DecisionTreeClassifier(max_depth=12,random_state=42),
                 "random_forest":RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1),
                 "gradient_boosting":GradientBoostingClassifier(random_state=42)}
    else:
        choices={"linear_regression":LinearRegression(),"ridge":Ridge(),"lasso":Lasso(random_state=42),
                 "random_forest":RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=-1),
                 "gradient_boosting":GradientBoostingRegressor(random_state=42)}
    if algorithm not in choices: raise ValueError("Unsupported algorithm")
    model=Pipeline([("preprocess",prep),("model",choices[algorithm])])
    t=time.perf_counter(); model.fit(Xtr,ytr); elapsed=time.perf_counter()-t
    pred=model.predict(Xte)
    if task=="classification":
        metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,average="weighted",zero_division=0),
                 "recall":recall_score(yte,pred,average="weighted",zero_division=0),"f1":f1_score(yte,pred,average="weighted",zero_division=0),
                 "confusion_matrix":confusion_matrix(yte,pred,labels=sorted(y.unique())).tolist(),"labels":sorted(y.unique())}
        ranking=metrics["f1"]
    else:
        mse=mean_squared_error(yte,pred)
        metrics={"r2":r2_score(yte,pred),"mae":mean_absolute_error(yte,pred),"mse":mse,"rmse":float(np.sqrt(mse)),
                 "actual_vs_predicted":[{"actual":int(a),"predicted":round(float(p),2)} for a,p in list(zip(yte,pred))[:250]],
                 "residuals":[round(float(a-p),2) for a,p in list(zip(yte,pred))[:250]]}
        ranking=-metrics["rmse"]
    key=f"{task}:{algorithm}"
    MODELS[key]=model
    result={"task":task,"algorithm":algorithm,"metrics":metrics,"training_time_seconds":round(elapsed,4),"ranking_score":ranking}
    RESULTS[:] = [r for r in RESULTS if not (r["task"]==task and r["algorithm"]==algorithm)]
    RESULTS.append(result)
    return result

def compare(task):
    return sorted([r for r in RESULTS if r["task"]==task], key=lambda r:r["ranking_score"], reverse=True)

def predict(task, algorithm, values):
    key=f"{task}:{algorithm}"
    if key not in MODELS: raise ValueError("Train this model before prediction.")
    df=clean_df()
    _,_,numeric,categorical=_features(df,task)
    row=pd.DataFrame([values])
    for c in numeric+categorical:
        if c not in row: row[c]=np.nan
    pred=MODELS[key].predict(row[numeric+categorical])[0]
    out={"prediction":str(pred) if task=="classification" else round(float(pred),2)}
    if task=="classification" and hasattr(MODELS[key], "predict_proba"):
        try: out["probabilities"]={str(k):round(float(v),4) for k,v in zip(MODELS[key].classes_,MODELS[key].predict_proba(row[numeric+categorical])[0])}
        except Exception: pass
    return out
