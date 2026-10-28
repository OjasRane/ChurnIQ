from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

class StrictModel(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)

class PredictionRequest(StrictModel):
    features: dict[str,FiniteFloat]

class Contribution(BaseModel):
    feature:str
    value:float
    contribution:float

class PredictionResponse(BaseModel):
    probability:float
    predicted_class:int
    risk:str
    contributions:list[Contribution]
    base_value:float
    input_summary:dict[str,float]
    shap_units:str

class FinancialAssumptions(StrictModel):
    monthly_revenue:float=Field(default=50,ge=0,le=1e9)
    contribution_margin:float=Field(default=.4,ge=0,le=1)
    lifetime_months:int=Field(default=24,ge=1,le=600)
    monthly_discount_rate:float=Field(default=.01,ge=0,le=1)
    intervention_cost:float=Field(default=25,ge=0,le=1e9)
    effectiveness:float=Field(default=.3,ge=0,le=1)
    campaign_budget:float=Field(default=1000,ge=0,le=1e12)
    currency:str=Field(default='USD',min_length=1,max_length=12,pattern=r'^[A-Za-z₹€£$ ]+$')

class FinanceRequest(StrictModel):
    probability_source:Literal['hypothetical','single_prediction','uploaded_portfolio']='hypothetical'
    assumptions:FinancialAssumptions=Field(default_factory=FinancialAssumptions)
    probabilities:list[FiniteFloat]=Field(min_length=1,max_length=10000)

class FinanceRow(BaseModel):
    row:int
    probability:float
    value_at_risk:float
    gross_benefit:float
    net_benefit:float
    roi:float|None
    selected:bool=False

class FinanceResponse(BaseModel):
    probability_source:str
    assumptions:FinancialAssumptions
    discounted_contribution:float
    rows:list[FinanceRow]
    total_value_at_risk:float
    total_gross_benefit:float
    total_net_benefit:float
    scenario_roi:float|None
    selected_count:int
    budget_used:float
    remaining_budget:float
    sensitivity:list[dict[str,float|None]]
    allocation_method:str
    disclaimer:str

class AIRequest(StrictModel):
    mode:Literal['Explain Prediction','Retention Strategy','Economic Interpretation','Executive Summary','Project Conclusion']='Executive Summary'
    features:dict[str,FiniteFloat]|None=None
    finance:FinanceRequest|None=None

class AIResponse(BaseModel):
    markdown:str
    model:str

class HealthResponse(BaseModel):
    status:str
    model_ready:bool
    ai_available:bool
    model_name:str

class BatchRow(BaseModel):
    model_config=ConfigDict(extra='allow')
    row:int
    probability:float|None
    prediction:int|None
    risk:str|None
    error:str

class BatchResponse(BaseModel):
    rows:list[BatchRow]
    valid_count:int
    invalid_count:int
    summary:dict[str,int]
    csv:str
