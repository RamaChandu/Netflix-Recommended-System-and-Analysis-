
from __future__ import annotations
from pathlib import Path
import re, zipfile
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "show_id","type","title","director","cast","country","date_added",
    "release_year","rating","duration","listed_in","description"
]

def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError("ZIP archive contains no CSV file")
            with z.open(csvs[0]) as f:
                df = pd.read_csv(f)
    else:
        df = pd.read_csv(path)
    validate_columns(df)
    return df

def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text_cols = ["director","cast","country","rating","duration","listed_in","description"]
    for c in text_cols:
        out[c] = out[c].replace(r"^\s*$", np.nan, regex=True)
    out["date_added"] = pd.to_datetime(out["date_added"], errors="coerce")
    out["addition_year"] = out["date_added"].dt.year.astype("Int64")
    out["addition_month"] = out["date_added"].dt.month.astype("Int64")
    out["duration_value"] = pd.to_numeric(
        out["duration"].str.extract(r"(\d+)")[0], errors="coerce"
    )
    out["duration_unit"] = pd.Series(
        np.select(
            [out["type"].eq("Movie"), out["type"].eq("TV Show")],
            ["minutes", "seasons"],
            default=None
        ),
        index=out.index,
        dtype="object"
    )
    out["duration_minutes"] = np.where(
        out["type"].eq("Movie"), out["duration_value"], np.nan
    )
    out["seasons"] = np.where(
        out["type"].eq("TV Show"), out["duration_value"], np.nan
    )
    return out

def split_multivalue(df: pd.DataFrame, column: str, value_name: str) -> pd.DataFrame:
    s = df[["show_id", column]].dropna().copy()
    s[column] = s[column].astype(str).str.split(", ")
    return s.explode(column).rename(columns={column: value_name})

def summary(df: pd.DataFrame) -> dict:
    clean = clean_dataset(df)
    genres = split_multivalue(clean, "listed_in", "genre")
    countries = split_multivalue(clean, "country", "country_name")
    return {
        "total_titles": int(len(clean)),
        "movies": int((clean["type"] == "Movie").sum()),
        "tv_shows": int((clean["type"] == "TV Show").sum()),
        "countries": int(countries["country_name"].nunique()),
        "genres": int(genres["genre"].nunique()),
        "release_year_min": int(clean["release_year"].min()),
        "release_year_max": int(clean["release_year"].max()),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }

def apply_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = clean_dataset(df)
    if params.get("type"):
        out = out[out["type"].eq(params["type"])]
    if params.get("rating"):
        out = out[out["rating"].eq(params["rating"])]
    if params.get("country"):
        out = out[out["country"].fillna("").str.contains(str(params["country"]), case=False, regex=False)]
    if params.get("genre"):
        out = out[out["listed_in"].fillna("").str.contains(str(params["genre"]), case=False, regex=False)]
    if params.get("q"):
        out = out[out["title"].fillna("").str.contains(str(params["q"]), case=False, regex=False)]
    if params.get("year_min") is not None:
        out = out[out["release_year"] >= int(params["year_min"])]
    if params.get("year_max") is not None:
        out = out[out["release_year"] <= int(params["year_max"])]
    if params.get("addition_year_min") is not None:
        out = out[out["addition_year"].fillna(0) >= int(params["addition_year_min"])]
    if params.get("addition_year_max") is not None:
        out = out[out["addition_year"].fillna(9999) <= int(params["addition_year_max"])]
    return out

def top_genres(df: pd.DataFrame, limit=15) -> list[dict]:
    g = split_multivalue(clean_dataset(df), "listed_in", "genre")
    return [{"name": k, "count": int(v)} for k, v in
            g["genre"].value_counts().head(limit).items()]

def top_countries(df: pd.DataFrame, limit=15) -> list[dict]:
    c = split_multivalue(clean_dataset(df), "country", "country_name")
    c = c[c["country_name"].ne("")]
    return [{"name": k, "count": int(v)} for k, v in
            c["country_name"].value_counts().head(limit).items()]
