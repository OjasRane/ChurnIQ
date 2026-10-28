import io
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.ml import get_service, FEATURES, ModelService
from app.finance import analyze
from app.schemas import FinanceRequest

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c: yield c

@pytest.fixture(scope='module')
def features(): return {x['name']:x['default'] for x in get_service().features()}

def test_detection_and_target():
    m=get_service()
    assert set(m.files)=={'train','test','sample','noisy'}
    assert len(m.train)==2280
    assert m.y.value_counts().to_dict()=={0:1923,1:357}
    assert 'id' not in m.X.columns
    assert list(m.X.columns)==FEATURES

def test_actual_metrics_and_baseline(client):
    d=client.get('/api/model/metrics').json()
    assert d['train_count']==1710 and d['validation_count']==570
    for model in ['CatBoost','Logistic Regression','CatBoost without State_Code']:
        metrics=d['comparison'][model]
        assert 0<metrics['roc_auc']<=1
        assert sum(map(sum,metrics['confusion_matrix']))==570
        assert len(metrics['roc_curve'])>2 and len(metrics['pr_curve'])>2

def test_prediction_shap_and_reload(client,features):
    response=client.post('/api/predict',json={'features':features})
    assert response.status_code==200
    p=response.json()
    assert 0<=p['probability']<=1 and len(p['contributions'])==12
    logodds=p['base_value']+sum(c['contribution'] for c in p['contributions'])
    assert math.isclose(1/(1+math.exp(-logodds)),p['probability'],abs_tol=1e-9)
    reloaded=ModelService()
    assert reloaded.predict(features)['probability']==p['probability']
    assert get_service() is get_service()

def test_bad_prediction(client,features):
    assert client.post('/api/predict',json={'features':{}}).status_code==422
    assert client.post('/api/predict',json={'features':{**features,'id':10}}).status_code==422
    invalid={**features,FEATURES[0]:'not a number'}
    assert client.post('/api/predict',json={'features':invalid}).status_code==422

def test_bulk(client,features):
    frame=pd.DataFrame([{**features,'id':'001'},{**features,'id':'002',FEATURES[0]:'invalid'},{**features,'id':'003'}])
    response=client.post('/api/predict-batch',files={'file':('customers.csv',frame.to_csv(index=False),'text/csv')})
    assert response.status_code==200
    result=response.json()
    assert result['valid_count']==2 and result['invalid_count']==1
    assert result['rows'][0]['id']=='001'
    assert result['rows'][1]['error']
    assert len(pd.read_csv(io.StringIO(result['csv'])))==3
    assert client.post('/api/predict-batch',files={'file':('bad.csv','id\n1','text/csv')}).status_code==422
    assert client.post('/api/predict-batch',files={'file':('bad.csv',b'','text/csv')}).status_code==422

def test_submission_exact_id_order_and_model_output(client):
    response=client.get('/api/competition/submission')
    assert response.status_code==200
    exported=pd.read_csv(io.StringIO(response.text),dtype={'id':str})
    sample=pd.read_csv(get_service().files['sample'],dtype={'id':str})
    test=pd.read_csv(get_service().files['test'])
    assert list(exported.columns)==['id','prediction'] and len(exported)==570
    assert exported.id.tolist()==sample.id.tolist()
    assert set(exported.prediction.unique())<={0,1}
    predicted=(get_service().model.predict_proba(test[FEATURES])[:,1]>=.5).astype(int)
    mapping=dict(zip(test.id,predicted))
    assert exported.prediction.tolist()==[mapping[int(i)] for i in sample.id]

def test_financial_calculation_and_allocation(client):
    body={'assumptions':{'monthly_revenue':100,'contribution_margin':.5,'lifetime_months':2,'monthly_discount_rate':.1,'intervention_cost':10,'effectiveness':.3,'campaign_budget':10},'probabilities':[.8,.9,.01]}
    response=client.post('/api/finance/analyze',json=body)
    assert response.status_code==200
    result=response.json();dc=50/1.1+50/1.1**2
    assert math.isclose(result['discounted_contribution'],dc)
    assert result['probability_source']=='hypothetical'
    assert result['selected_count']==1
    assert result['rows'][1]['selected'] and not result['rows'][0]['selected']
    assert math.isclose(result['total_net_benefit'],.9*.3*dc-10)
    assert result['budget_used']<=10
    body['assumptions']['intervention_cost']=0
    result=client.post('/api/finance/analyze',json=body).json()
    assert result['scenario_roi'] is None and all(r['roi'] is None for r in result['rows'])
    body['probabilities']=[1.1]
    assert client.post('/api/finance/analyze',json=body).status_code==422

def test_health_metadata_and_cors(client):
    health=client.get('/api/health')
    assert health.status_code==200 and health.json()['model_ready']
    for path in ['/overview','/features','/model/importance','/data/quality']:
        assert client.get('/api'+path).status_code==200
    assert client.get('/api/data/quality').json()['noisy']['missing_cells']==3624
    cors=client.options('/api/predict',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST'})
    assert cors.headers['access-control-allow-origin']=='http://localhost:5173'
    denied=client.options('/api/predict',headers={'Origin':'https://untrusted.example','Access-Control-Request-Method':'POST'})
    assert 'access-control-allow-origin' not in denied.headers

def test_missing_gemini(client,monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    for path in ['/api/ai/analyze','/api/ai/report']:
        r=client.post(path,json={'mode':'Executive Summary'})
        assert r.status_code==503 and 'GEMINI_API_KEY' in r.json()['detail']
    assert client.get('/api/health').status_code==200
