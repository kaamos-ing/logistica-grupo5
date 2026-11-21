"""Recalcula artefactos desde datos/. Ejecutar: python prepare.py"""
from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import time
import joblib
import pandas as pd
from analytics import read_data, dimension1, dimension2, dimension3, dimension4, DATA_NAMES
from routing import solve_pilot

ROOT=Path(__file__).resolve().parent


def main():
    target=ROOT/'artifacts';target.mkdir(exist_ok=True)
    print('Leyendo datos...',flush=True)
    tables=read_data(ROOT/'datos')
    summary={n:{'filas':len(df),'columnas':len(df.columns)} for n,df in tables.items()}
    rows=[]
    for i,func in enumerate([dimension1,dimension2,dimension3,dimension4],1):
        print(f'Dimensión {i}: calculando...',flush=True);start=time.perf_counter()
        result=func(tables)
        if i==2:
            result['routing']=solve_pilot(result['pilot'])
        joblib.dump(result,target/f'dim{i}.joblib',compress=3)
        if i==1:
            rows.append({'Dimensión':i,'Modelo':'XGBoost',**result['metrics']})
        else:
            table=result['comparison']
            rows.extend([{'Dimensión':i,**r} for r in table.to_dict('records')])
        print(f'  OK {time.perf_counter()-start:.1f}s',flush=True)
    packages={p:version(p) for p in ['streamlit','pandas','numpy','scipy','scikit-learn','statsmodels','xgboost','plotly','joblib']}
    manifest={'dataset':'https://www.kaggle.com/datasets/yogape/logistics-operations-database',
              'tables':summary,'versions':packages,'files':{
                  n+'.csv':hashlib.sha256((ROOT/'datos'/(n+'.csv')).read_bytes()).hexdigest() for n in DATA_NAMES}}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame(rows).to_csv(target/'metricas_verificadas.csv',index=False)
    # Only runtime dependencies; Kaggle and Colab are unnecessary after packaging.
    (ROOT/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in packages.items())+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
