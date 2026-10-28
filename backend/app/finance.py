from .schemas import FinanceRequest
import math

DISCLAIMER='Scenario estimates, not realized or experimentally established profits. Effectiveness is an assumption, not causal uplift. No true causal intervention data exists.'

def analyze(request:FinanceRequest):
    a=request.assumptions
    if any(not 0<=p<=1 for p in request.probabilities):
        raise ValueError('Every churn probability must be between 0 and 1.')
    dc=sum(a.monthly_revenue*a.contribution_margin/(1+a.monthly_discount_rate)**t for t in range(1,a.lifetime_months+1))
    rows=[]
    for i,p in enumerate(request.probabilities):
        value=p*dc; gross=value*a.effectiveness; net=gross-a.intervention_cost
        rows.append(dict(row=i+1,probability=p,value_at_risk=value,gross_benefit=gross,net_benefit=net,
                         roi=net/a.intervention_cost*100 if a.intervention_cost else None,selected=False))
    eligible=sorted([r for r in rows if r['net_benefit']>0],key=lambda r:(-r['net_benefit'],r['row']))
    capacity=len(eligible) if a.intervention_cost==0 else min(len(eligible),math.floor(a.campaign_budget/a.intervention_cost))
    selected=eligible[:capacity]
    for row in selected: row['selected']=True
    used=len(selected)*a.intervention_cost
    total_net=sum(r['net_benefit'] for r in selected)
    # Sensitivity uses the current whole cohort, before campaign selection, for comparison.
    sensitivity=[]
    for e in [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]:
        gross=sum(p*dc*e for p in request.probabilities)
        cost=len(rows)*a.intervention_cost
        sensitivity.append(dict(effectiveness=e,net_benefit=gross-cost,roi=(gross-cost)/cost*100 if cost else None))
    return dict(probability_source=request.probability_source,assumptions=a.model_dump(),discounted_contribution=dc,rows=rows,
        total_value_at_risk=sum(r['value_at_risk'] for r in rows),
        total_gross_benefit=sum(r['gross_benefit'] for r in selected),total_net_benefit=total_net,
        scenario_roi=total_net/used*100 if used else None,selected_count=len(selected),
        budget_used=used,remaining_budget=a.campaign_budget-used,sensitivity=sensitivity,
        allocation_method='Exact selection under identical per-customer costs: rank positive net benefits descending, then take the largest affordable prefix. No heterogeneous costs or uplift estimates.',
        disclaimer=DISCLAIMER)
