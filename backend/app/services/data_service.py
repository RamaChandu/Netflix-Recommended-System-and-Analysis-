
from functools import lru_cache
from pathlib import Path
import pandas as pd
from shared.data_processing import load_dataset, clean_dataset, summary, top_genres, top_countries, split_multivalue
from app.config import DATA_PATH

@lru_cache(maxsize=1)
def raw_df():
    return load_dataset(DATA_PATH)

@lru_cache(maxsize=1)
def clean_df():
    return clean_dataset(raw_df())

def overview():
    return summary(raw_df())

def content_types():
    return [{"name": k, "count": int(v)} for k,v in raw_df()["type"].value_counts().items()]

def genres(limit=20):
    return top_genres(raw_df(), limit)

def countries(limit=20):
    return top_countries(raw_df(), limit)

def ratings():
    s = raw_df()["rating"].fillna("Unknown").value_counts()
    return [{"name": k, "count": int(v)} for k,v in s.items()]

def trends():
    c = clean_df()
    release = c.groupby(["release_year","type"]).size().reset_index(name="count")
    additions = c.dropna(subset=["addition_year"]).groupby(["addition_year","type"]).size().reset_index(name="count")
    monthly = c.dropna(subset=["date_added"]).assign(month=c["date_added"].dt.month_name().str[:3])
    month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = monthly.groupby("month").size().reindex(month_order, fill_value=0)
    return {
        "release": release.to_dict("records"),
        "additions": additions.to_dict("records"),
        "monthly": [{"month": k, "count": int(v)} for k,v in monthly.items()]
    }

def data_quality():
    df = raw_df()
    c = clean_df()
    missing = [{"column": k, "count": int(v), "percent": round(float(v/len(df)*100),2)} for k,v in df.isna().sum().items()]
    invalid_dates = int(c["date_added"].isna().sum())
    return {
        "shape": [int(x) for x in df.shape],
        "dtypes": {k:str(v) for k,v in df.dtypes.items()},
        "missing": missing,
        "duplicates": int(df.duplicated().sum()),
        "invalid_dates": invalid_dates,
        "duration_parsed": int(c["duration_value"].notna().sum()),
        "cleaned_columns": list(c.columns),
    }

def titles(params: dict):
    from shared.data_processing import apply_filters
    out = apply_filters(raw_df(), params)
    sort = params.get("sort","title")
    ascending = params.get("order","asc") == "asc"
    if sort in out.columns:
        out = out.sort_values(sort, ascending=ascending, na_position="last")
    page, size = max(1,int(params.get("page",1))), min(100,max(1,int(params.get("size",24))))
    start=(page-1)*size
    rows=out.iloc[start:start+size].copy()
    rows["date_added"] = rows["date_added"].dt.strftime("%Y-%m-%d")
    rows=rows.fillna("")
    return {"items": rows[["show_id","title","type","release_year","date_added","rating","duration","country","listed_in","director","cast","description"]].to_dict("records"),
            "total": int(len(out)), "page": page, "size": size}

def title(show_id: str):
    out = raw_df()
    row = out[out["show_id"].astype(str).eq(str(show_id))]
    if row.empty: return None
    r=clean_df()[clean_df()["show_id"].astype(str).eq(str(show_id))].iloc[0].copy()
    if pd.notna(r["date_added"]): r["date_added"]=r["date_added"].strftime("%Y-%m-%d")
    return r.fillna("").to_dict()
