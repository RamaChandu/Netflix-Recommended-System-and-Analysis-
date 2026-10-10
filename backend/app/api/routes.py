
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import io, pandas as pd
from app.services import data_service
from app.services.recommendation_service import recommend
from app.ml.models import train, compare, predict
from app.schemas.models import RecommendationRequest, TrainRequest, PredictRequest

router=APIRouter(prefix="/api")

@router.get("/health")
def health():
    s=data_service.overview()
    return {"status":"ok","dataset_loaded":True,"titles":s["total_titles"]}

@router.get("/summary")
def summary(): return data_service.overview()

@router.get("/titles")
def titles(q:str|None=None,type:str|None=None,rating:str|None=None,country:str|None=None,genre:str|None=None,
           year_min:int|None=None,year_max:int|None=None,addition_year_min:int|None=None,addition_year_max:int|None=None,
           page:int=1,size:int=24,sort:str="title",order:str="asc"):
    return data_service.titles(locals())

@router.get("/titles/{show_id}")
def title(show_id:str):
    r=data_service.title(show_id)
    if r is None: raise HTTPException(404,"Title not found")
    return r

@router.get("/analytics/content-types")
def content_types(): return data_service.content_types()
@router.get("/analytics/genres")
def genres(limit:int=20): return data_service.genres(limit)
@router.get("/analytics/countries")
def countries(limit:int=20): return data_service.countries(limit)
@router.get("/analytics/ratings")
def ratings(): return data_service.ratings()
@router.get("/analytics/trends")
def trends(): return data_service.trends()
@router.get("/analytics/data-quality")
def quality(): return data_service.data_quality()

@router.post("/recommendations")
def recommendations(req:RecommendationRequest):
    try: return {"items":recommend(req.show_id,req.limit,req.type_filter,req.genre_filter)}
    except KeyError as e: raise HTTPException(404,str(e))
    except Exception as e: raise HTTPException(400,str(e))

@router.post("/ml/train")
def ml_train(req:TrainRequest):
    try: return train(req.task,req.algorithm)
    except Exception as e: raise HTTPException(400,str(e))

@router.get("/ml/comparison")
def ml_comparison(task:str): return {"items":compare(task)}

@router.post("/ml/predict")
def ml_predict(req:PredictRequest):
    try: return predict(req.task,req.algorithm,req.values)
    except Exception as e: raise HTTPException(400,str(e))

@router.get("/export/cleaned-data")
def export():
    df=data_service.clean_df().copy()
    df["date_added"]=df["date_added"].dt.strftime("%Y-%m-%d")
    buf=io.StringIO(); df.to_csv(buf,index=False); buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]),media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=netflix_cleaned.csv"})
