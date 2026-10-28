from app.ml import get_service
if __name__ == '__main__':
    model=get_service()
    import json
    print(json.dumps({'model':model.metadata['model_name'],'comparison':{k:{a:b for a,b in v.items() if a not in ('roc_curve','pr_curve')} for k,v in model.metadata['comparison'].items()}},indent=2))
