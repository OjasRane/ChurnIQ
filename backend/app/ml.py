"""Reproducible train-only fitting; fixed holdout, no test-label assumptions."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import os
import platform
import threading
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn import __version__ as sklearn_version
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, brier_score_loss)

FEATURES = ['Stability_Index','Incident_Marker','Mission_Tenure','Allocation_Class',
 'Activity_Signature','Cycle_Rate','Event_Density','Interaction_Range','Colony_Class',
 'Protocol_Class','State_Code','Mission_Priority_Index']
BASE = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.getenv('MODEL_DIR', str(BASE / 'artifacts')))
PARAMS = dict(iterations=450, depth=6, learning_rate=0.05, l2_leaf_reg=5,
              loss_function='Logloss', auto_class_weights='Balanced',
              random_seed=42, thread_count=2, verbose=False, allow_writing_files=False)
LIMITATIONS = [
 'One stratified 25% holdout (random_state=42); no cross-validation or independent external evaluation.',
 'Fixed hyperparameters and 0.5 classification threshold; no holdout-based tuning. Final inference model is refit on all 2,280 labeled rows; metrics describe the holdout model, not this refit.',
 'State_Code may be a contemporaneous status or leakage-prone feature. Its meaning and collection time are unverified; an ablation is reported, not proof of absence of leakage.',
 'Anonymized features have no verified business meanings. SHAP contributions are log-odds and describe model behavior, not causal effects.',
 'Class balancing affects probability calibration. Brier score is reported; probabilities are not calibrated with a separate calibration set.',
 'There are no causal intervention outcomes or verified monetary lifetime values. Financial results are hypothetical scenarios.',
 'Noisy data is diagnosed separately, never mapped or joined to competition records. Submission placeholders are never training labels.'
]

def discover():
    candidates = [Path(os.environ['DATA_DIR'])] if os.getenv('DATA_DIR') else [BASE/'data', BASE.parent/'data']
    directory = next((p for p in candidates if p.is_dir()), None)
    if directory is None:
        raise RuntimeError('Dataset directory missing. Bundle backend/data or set DATA_DIR.')
    found = {}
    for p in sorted(directory.glob('*.csv')):
        columns = set(pd.read_csv(p, nrows=0).columns)
        kind = None
        if set(FEATURES+['Outcome']) <= columns: kind = 'train'
        elif set(FEATURES+['id']) <= columns and 'Outcome' not in columns: kind = 'test'
        elif columns == {'id','prediction'}: kind = 'sample'
        elif 'Churn' in columns: kind = 'noisy'
        if kind:
            if kind in found: raise RuntimeError(f'Ambiguous {kind} CSV files in {directory}')
            found[kind] = p
    if not {'train','test','sample'} <= found.keys():
        raise RuntimeError('Could not discover training, test, and sample CSVs by columns.')
    return found


def validate_frame(frame):
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing: raise ValueError('Missing required features: '+', '.join(missing))
    return frame.loc[:, FEATURES].apply(pd.to_numeric, errors='coerce')


def evaluate(y, probability):
    predicted = probability >= 0.5
    fpr, tpr, _ = roc_curve(y, probability)
    precision, recall, _ = precision_recall_curve(y, probability)
    return dict(accuracy=float(accuracy_score(y,predicted)),
        precision=float(precision_score(y,predicted,zero_division=0)),
        recall=float(recall_score(y,predicted,zero_division=0)),
        f1=float(f1_score(y,predicted,zero_division=0)),
        roc_auc=float(roc_auc_score(y,probability)),
        pr_auc=float(average_precision_score(y,probability)),
        brier_score=float(brier_score_loss(y,probability)),
        confusion_matrix=confusion_matrix(y,predicted,labels=[0,1]).tolist(),
        roc_curve=[dict(fpr=float(a),tpr=float(b)) for a,b in zip(fpr,tpr)],
        pr_curve=[dict(recall=float(a),precision=float(b)) for a,b in zip(recall,precision)])


def risk(p): return 'High' if p >= .7 else 'Medium' if p >= .3 else 'Low'


class ModelService:
    def __init__(self):
        self.files = discover()
        self.train = pd.read_csv(self.files['train'])
        self.X = validate_frame(self.train)
        y = pd.to_numeric(self.train['Outcome'],errors='coerce')
        if set(y.unique()) != {0,1}: raise ValueError('Outcome must contain only binary labels 0 and 1.')
        if not np.isfinite(self.X.to_numpy()).all(): raise ValueError('Training features must be finite numeric values.')
        self.y = y.astype(int)
        fingerprint = hashlib.sha256(self.files['train'].read_bytes()+json.dumps(PARAMS,sort_keys=True).encode()+b'churniq-v2').hexdigest()
        ARTIFACTS.mkdir(parents=True,exist_ok=True)
        self.model = CatBoostClassifier()
        meta_path = ARTIFACTS/'metadata.json'
        if meta_path.exists() and (ARTIFACTS/'model.cbm').exists():
            metadata = json.loads(meta_path.read_text())
            if metadata.get('fingerprint') == fingerprint and metadata.get('features') == FEATURES:
                self.model.load_model(str(ARTIFACTS/'model.cbm'))
                self.metadata = metadata
                return
        self.fit(fingerprint)

    def fit(self, fingerprint):
        Xtr, Xv, ytr, yv = train_test_split(self.X,self.y,test_size=.25,stratify=self.y,random_state=42)
        evaluation_model = CatBoostClassifier(**PARAMS).fit(Xtr,ytr)
        pv = evaluation_model.predict_proba(Xv)[:,1]
        baseline = make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                                 LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42))
        baseline.fit(Xtr,ytr)
        ablation = CatBoostClassifier(**PARAMS).fit(Xtr.drop(columns='State_Code'),ytr)
        comparison = {'CatBoost':evaluate(yv,pv),
                      'Logistic Regression':evaluate(yv,baseline.predict_proba(Xv)[:,1]),
                      'CatBoost without State_Code':evaluate(yv,ablation.predict_proba(Xv.drop(columns='State_Code'))[:,1])}
        # Explanations and aggregate cohort insights are validation-derived, never test labels.
        shap = evaluation_model.get_feature_importance(Pool(Xv.iloc[:128]),type='ShapValues',thread_count=2)
        global_shap = sorted([{'feature':c,'importance':float(v)} for c,v in zip(FEATURES,np.abs(shap[:,:-1]).mean(axis=0))],key=lambda x:-x['importance'])
        self.model = CatBoostClassifier(**PARAMS).fit(self.X,self.y)
        self.model.save_model(str(ARTIFACTS/'model.cbm'))
        self.metadata = dict(fingerprint=fingerprint,features=FEATURES,model_name='CatBoostClassifier',
            params=PARAMS,train_count=len(Xtr),validation_count=len(Xv),final_fit_count=len(self.X),
            threshold=.5,comparison=comparison,limitations=LIMITATIONS,
            global_shap=global_shap,validation_risk_counts={k:sum(risk(float(p))==k for p in pv) for k in ['Low','Medium','High']},
            runtime={'python':platform.python_version(),'sklearn':sklearn_version})
        temp = ARTIFACTS/'metadata.tmp'
        temp.write_text(json.dumps(self.metadata,indent=2,allow_nan=False)); temp.replace(ARTIFACTS/'metadata.json')

    def features(self):
        result=[]
        for c in FEATURES:
            col=self.X[c]; values=sorted(float(x) for x in col.unique())
            result.append(dict(name=c,type='number',min=float(col.min()),max=float(col.max()),
                default=float(col.median()),integer=bool((col%1==0).all()),
                options=values if len(values)<=8 else None))
        return result

    def predict(self, values):
        x=pd.DataFrame([values],columns=FEATURES)
        p=float(self.model.predict_proba(x)[0,1])
        shap=self.model.get_feature_importance(Pool(x),type='ShapValues',thread_count=2)[0]
        contributions=sorted([dict(feature=c,value=float(values[c]),contribution=float(v)) for c,v in zip(FEATURES,shap[:-1])],key=lambda a:-abs(a['contribution']))
        return dict(probability=p,predicted_class=int(p>=.5),risk=risk(p),contributions=contributions,
                    base_value=float(shap[-1]),input_summary=values,shap_units='log-odds')

    def importance(self):
        return dict(importance=sorted([dict(feature=c,importance=float(v)) for c,v in zip(FEATURES,self.model.get_feature_importance(thread_count=2))],key=lambda a:-a['importance']),
                    global_shap=self.metadata['global_shap'],limitations=LIMITATIONS)

    def overview(self):
        positive=int(self.y.sum())
        return dict(customer_count=len(self.train),positive_count=positive,churn_rate=positive/len(self.train),
          feature_count=len(FEATURES),target_distribution=[{'name':'Retained','value':len(self.train)-positive},{'name':'Churned','value':positive}],
          features=self.features(),model_name='CatBoost',validation_count=self.metadata['validation_count'],
          validation_risk_counts=self.metadata['validation_risk_counts'])

    def quality(self):
        def describe(df):
            return dict(rows=len(df),columns=len(df.columns),missing_cells=int(df.isna().sum().sum()),
                duplicate_rows=int(df.duplicated().sum()),
                missing_by_column={c:int(n) for c,n in df.isna().sum().items()},
                target_distribution={str(k):int(v) for k,v in df['Churn' if 'Churn' in df else 'Outcome'].value_counts().items()})
        return dict(train=describe(self.train),noisy=describe(pd.read_csv(self.files['noisy'])) if 'noisy' in self.files else None,
                    note='Noisy data is separate: no training, row matching, or automatic feature-name mapping.')

    def submission(self):
        test=pd.read_csv(self.files['test'],dtype={'id':str})
        sample=pd.read_csv(self.files['sample'],dtype={'id':str})
        # IDs differ only in zero padding in the supplied files. Match canonical integers, output original sample strings.
        def canonical(series):
            if not series.str.fullmatch(r'\d+').all(): raise ValueError('Expected numeric competition IDs.')
            return series.map(lambda s:str(int(s)))
        keys=canonical(test.id); desired=canonical(sample.id)
        if keys.duplicated().any() or desired.duplicated().any() or set(keys)!=set(desired):
            raise ValueError('Test and sample IDs must be unique and match exactly after zero-padding normalization.')
        x=validate_frame(test)
        if not np.isfinite(x.to_numpy()).all(): raise ValueError('Competition test contains non-finite features.')
        predictions=pd.Series((self.model.predict_proba(x)[:,1]>=.5).astype(int),index=keys)
        return pd.DataFrame({'id':sample.id,'prediction':predictions.loc[desired].to_numpy()})

_lock=threading.Lock()
@lru_cache(maxsize=1)
def get_service():
    with _lock: return ModelService()
