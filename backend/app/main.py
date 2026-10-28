from contextlib import asynccontextmanager
import csv
import io
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from .ml import get_service, FEATURES, validate_frame, risk
from .schemas import PredictionRequest, PredictionResponse, FinanceRequest, FinanceResponse, AIRequest, AIResponse, HealthResponse, BatchResponse
from .finance import analyze
from .gemini_service import generate

load_dotenv(Path(__file__).resolve().parents[1]/'.env')

@asynccontextmanager
async def lifespan(app):
    await run_in_threadpool(get_service)
    yield

app=FastAPI(title='ChurnIQ API',version='1.0.0',lifespan=lifespan)
origins=[o.strip().rstrip('/') for o in os.getenv('CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(',') if o.strip()]
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'],expose_headers=['Content-Disposition'])

def check_features(values):
    if set(values)!=set(FEATURES):
        raise HTTPException(422,detail={'message':'Supply exactly the 12 model features.','missing':sorted(set(FEATURES)-set(values)),'unexpected':sorted(set(values)-set(FEATURES))})

@app.get('/api/health',response_model=HealthResponse)
def health():
    return {'status':'ok','model_ready':True,'ai_available':bool(os.getenv('GEMINI_API_KEY','').strip()),'model_name':get_service().metadata['model_name']}

@app.get('/api/overview')
def overview(): return get_service().overview()

@app.get('/api/features')
def features(): return {'features':get_service().features(),'note':'Observed ranges are training ranges, not verified business constraints. Features remain numeric and in their original order.'}

@app.get('/api/model/metrics')
def metrics():
    meta=get_service().metadata
    return {k:meta[k] for k in ['comparison','train_count','validation_count','final_fit_count','threshold','limitations','params']}

@app.get('/api/model/importance')
def importance(): return get_service().importance()

@app.get('/api/data/quality')
def quality(): return get_service().quality()

@app.post('/api/predict',response_model=PredictionResponse)
def predict(request:PredictionRequest):
    check_features(request.features)
    return get_service().predict(request.features)

@app.post('/api/predict-batch',response_model=BatchResponse)
async def batch(file:UploadFile=File(...)):
    content=await file.read(5*1024*1024+1)
    if len(content)>5*1024*1024: raise HTTPException(413,detail='CSV must be 5 MB or smaller.')
    try:
        decoded=content.decode('utf-8-sig')
        header=next(csv.reader(io.StringIO(decoded)))
        if len(header)!=len(set(header)): raise ValueError('CSV contains duplicate column names.')
        frame=pd.read_csv(io.StringIO(decoded),dtype=str,keep_default_na=False)
        if not 1<=len(frame)<=10000: raise ValueError('CSV must contain between 1 and 10,000 rows.')
        x=validate_frame(frame)
    except (ValueError,UnicodeError,StopIteration,csv.Error,pd.errors.ParserError) as exc:
        raise HTTPException(422,detail=f'Invalid CSV: {exc}') from None
    valid=np.isfinite(x.to_numpy()).all(axis=1)
    if not valid.any(): raise HTTPException(422,detail='No valid rows. All 12 features must be finite numeric values.')
    service=get_service()
    probabilities=await run_in_threadpool(lambda:service.model.predict_proba(x.loc[valid])[:,1])
    lookup=dict(zip(np.flatnonzero(valid),probabilities))
    id_columns=[c for c in frame.columns if c.lower() in {'id','customer_id','customerid','customer id'}]
    rows=[]
    for i in range(len(frame)):
        row={'row':i+1,**{c:str(frame.iloc[i][c]) for c in id_columns}}
        if i in lookup:
            p=float(lookup[i]);row.update(probability=p,prediction=int(p>=.5),risk=risk(p),error='')
        else: row.update(probability=None,prediction=None,risk=None,error='Non-numeric, missing, or infinite feature value')
        rows.append(row)
    # Escape spreadsheet formulas in user-supplied identifiers for safe CSV opening.
    export=pd.DataFrame(rows)
    for c in id_columns:
        export[c]=export[c].map(lambda s:"'"+s if s.startswith(('=','+','-','@','\t','\r')) else s)
    return {'rows':rows,'valid_count':int(valid.sum()),'invalid_count':int((~valid).sum()),
            'summary':{k:sum(r['risk']==k for r in rows) for k in ['Low','Medium','High']},
            'csv':export.to_csv(index=False)}

@app.get('/api/competition/submission')
def submission():
    try: result=get_service().submission()
    except ValueError as e: raise HTTPException(422,detail=str(e)) from None
    return StreamingResponse(iter([result.to_csv(index=False)]),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="churniq_submission.csv"'})

@app.post('/api/finance/analyze',response_model=FinanceResponse)
def finance(request:FinanceRequest):
    try: return analyze(request)
    except ValueError as e: raise HTTPException(422,detail=str(e)) from None

@app.post('/api/ai/analyze',response_model=AIResponse)
def ai(request:AIRequest):
    if request.features is not None: check_features(request.features)
    return generate(request,get_service())

@app.post('/api/ai/report',response_model=AIResponse)
def report(request:AIRequest):
    if request.features is not None: check_features(request.features)
    return generate(request.model_copy(update={'mode':'Executive Summary'}),get_service())
