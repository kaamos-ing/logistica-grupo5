"""Modelos reproducibles adaptados de los cuatro cuadernos corregidos.

Se ejecutan con prepare.py; la interfaz carga resultados ya calculados.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from xgboost import XGBRegressor
from routing import haversine, find_pilot

SEED = 42
DATA_NAMES = ['trucks', 'trips', 'loads', 'routes', 'delivery_events', 'facilities',
              'driver_monthly_metrics', 'truck_utilization_metrics', 'trailers']


def read_data(folder):
    return {n: pd.read_csv(Path(folder)/(n+'.csv')) for n in DATA_NAMES}


def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    valid = y != 0
    return {'MAE': float(mean_absolute_error(y, pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
            'MAPE (%)': float(np.mean(np.abs((y[valid]-pred[valid])/y[valid]))*100) if valid.any() else np.nan,
            'R²': float(r2_score(y, pred))}


def unique_trips(t):
    trips, loads, routes, events, facilities = [t[x] for x in ['trips', 'loads', 'routes', 'delivery_events', 'facilities']]
    pu = events.loc[events.event_type == 'Pickup', ['trip_id', 'facility_id']].rename(columns={'facility_id': 'pickup_facility'})
    de = events.loc[events.event_type == 'Delivery', ['trip_id', 'facility_id', 'scheduled_datetime']].rename(
        columns={'facility_id': 'delivery_facility', 'scheduled_datetime': 'delivery_scheduled'})
    v = (trips.merge(loads, on='load_id', validate='one_to_one')
         .merge(routes, on='route_id', validate='many_to_one')
         .merge(pu, on='trip_id', validate='one_to_one')
         .merge(de, on='trip_id', validate='one_to_one'))
    geo = facilities.set_index('facility_id')
    for label, col in [('origen', 'pickup_facility'), ('destino', 'delivery_facility')]:
        for coord in ['latitude', 'longitude']:
            v[label+'_'+coord] = v[col].map(geo[coord])
    v['dispatch_date'] = pd.to_datetime(v.dispatch_date)
    v['delivery_scheduled'] = pd.to_datetime(v.delivery_scheduled)
    return v.sort_values(['dispatch_date', 'trip_id']).reset_index(drop=True)


def dimension1(t):
    v = t['trips'].merge(t['loads'][['load_id','weight_lbs','pieces','load_type','booking_type','route_id']],
                          on='load_id', how='left', validate='one_to_one')
    v = v.merge(t['routes'][['route_id','origin_city','destination_city','typical_distance_miles','typical_transit_days']],
                on='route_id', how='left', validate='many_to_one')
    dates = pd.to_datetime(v.dispatch_date, errors='coerce')
    v['dispatch_month'], v['dispatch_day'] = dates.dt.month, dates.dt.dayofweek
    features = ['typical_distance_miles','typical_transit_days','weight_lbs','pieces','load_type','booking_type',
                'route_id','origin_city','destination_city','dispatch_month','dispatch_day']
    v = v.dropna(subset=['actual_duration_hours'])
    X, y = v[features], v.actual_duration_hours
    cat = X.select_dtypes(include=['object','string','category']).columns.tolist()
    if 'route_id' not in cat:
        cat.append('route_id')
    num = [c for c in features if c not in cat]
    prep = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                           ('encoder', OneHotEncoder(handle_unknown='ignore'))]), cat)])
    model = Pipeline([('preprocesamiento', prep), ('regresor', XGBRegressor(
        n_estimators=500, learning_rate=.03, max_depth=5, min_child_weight=3, subsample=.8,
        colsample_bytree=.8, objective='reg:squarederror', random_state=SEED, n_jobs=2))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=SEED)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    importance = pd.DataFrame({'Variable': model.named_steps['preprocesamiento'].get_feature_names_out(),
                              'Importancia (%)': model.named_steps['regresor'].feature_importances_*100}).sort_values('Importancia (%)', ascending=False)
    delivery = t['delivery_events'].query("event_type == 'Delivery'").copy()
    delivery['Desviación (h)'] = (pd.to_datetime(delivery.actual_datetime)-pd.to_datetime(delivery.scheduled_datetime)).dt.total_seconds()/3600
    util = t['truck_utilization_metrics'].copy()
    util['month'] = pd.to_datetime(util.month)
    return {'model': model, 'metrics': metrics(yte, pred), 'n_train': len(Xtr), 'n_test': len(Xte),
        'importance': importance, 'features': features,
        'predictions': pd.DataFrame({'Real (h)': yte, 'Predicción (h)': pred}).reset_index(drop=True),
        'routes': t['routes'].copy(), 'defaults': X[num].median().to_dict(),
        'load_types': sorted(X.load_type.dropna().unique()), 'booking_types': sorted(X.booking_type.dropna().unique()),
        'util_monthly': util.groupby('month').utilization_rate.mean().mul(100).rename('Utilización registrada (%)'),
        'util_above100': float((util.utilization_rate > 1).mean()*100),
        'delay': delivery['Desviación (h)'].sample(min(5000,len(delivery)),random_state=SEED),
        'otd_flag': float(delivery.on_time_flag.mean()*100),
        'fuel': t['trips'][['average_mpg','actual_distance_miles','fuel_gallons_used']].describe().T,
        'eda': v[['actual_duration_hours','weight_lbs','pieces','typical_distance_miles']].describe().T}


def dimension2(t):
    v = unique_trips(t)
    v['distancia_geodesica_millas'] = haversine(v.origen_latitude, v.origen_longitude, v.destino_latitude, v.destino_longitude)
    features = ['typical_distance_miles','weight_lbs','pieces','typical_transit_days','distancia_geodesica_millas']
    target = 'actual_duration_hours'
    d = v.dropna(subset=features+[target])
    d = d[(d[target] > 0) & (d[features] >= 0).all(axis=1)]
    tr, val, test = [d[d.dispatch_date.dt.year == year] for year in [2022,2023,2024]]
    if min(len(tr),len(val),len(test)) < 5:
        raise ValueError('Faltan viajes para la división temporal 2022/2023/2024.')
    def new_rf():
        return RandomForestRegressor(n_estimators=55,max_depth=10,min_samples_leaf=3,random_state=SEED,n_jobs=2)
    select = new_rf().fit(tr[features],tr[target])
    sample = val.sample(min(1500,len(val)),random_state=SEED)
    pi = permutation_importance(select,sample[features],sample[target],scoring='neg_mean_absolute_error',n_repeats=3,random_state=SEED,n_jobs=1)
    importance = pd.DataFrame({'Variable': features,'Aumento MAE (h)': pi.importances_mean}).sort_values('Aumento MAE (h)',ascending=False)
    reduced = importance.Variable.head(3).tolist()
    dev = pd.concat([tr,val]); models = {}; rows = []; predictions = {}
    for label, cols in [('Completo',features),('Reducido',reduced)]:
        m = new_rf().fit(dev[cols],dev[target]); p = m.predict(test[cols])
        models[label] = m; predictions[label] = p
        rows.append({'Modelo':label,'Variables':len(cols),**metrics(test[target],p)})
    pilot = find_pilot(v)
    return {'models':models,'features':features,'reduced':reduced,'importance':importance,
        'comparison':pd.DataFrame(rows),'pilot':pilot,'train_sizes':[len(tr),len(val),len(test)],
        'predictions':pd.DataFrame({'Real (h)':test[target].to_numpy(),**predictions}),
        'n_trips':len(v),'active_trucks':int((t['trucks'].status=='Active').sum()),
        'eda':d[features+[target]].describe().T}


def weekly_features(series):
    X = pd.DataFrame(index=series.index)
    for lag in [1,2,3,4]:
        X[f'lag_{lag}'] = series.shift(lag)
    X['rolling_mean_4'] = series.rolling(4).mean().shift(1)
    X['trend'] = np.arange(len(series))
    week = series.index.isocalendar().week.astype(int)
    X['sin_anual'] = np.sin(2*np.pi*week/52)
    X['cos_anual'] = np.cos(2*np.pi*week/52)
    return X


def dimension3(t):
    de = t['delivery_events'].query("event_type == 'Delivery'").copy()
    for c in ['scheduled_datetime','actual_datetime']:
        de[c] = pd.to_datetime(de[c],errors='coerce')
    de['delay_min'] = (de.actual_datetime-de.scheduled_datetime).dt.total_seconds()/60
    de['puntual'] = de.actual_datetime <= de.scheduled_datetime
    d = (t['loads'].merge(de[['load_id','facility_id','scheduled_datetime','actual_datetime','delay_min','puntual']],on='load_id',how='left',validate='one_to_one')
         .merge(t['facilities'][['facility_id','state','latitude','longitude']],on='facility_id',how='left',validate='many_to_one'))
    d['load_date'] = pd.to_datetime(d.load_date)
    d = d.sort_values('revenue',ascending=False).reset_index(drop=True)
    d['ingreso_acum_pct'] = d.revenue.cumsum()/d.revenue.sum()*100
    d['cargas_acum_pct'] = (d.index+1)/len(d)*100
    d['clase'] = np.where(d.ingreso_acum_pct<=80,'A',np.where(d.ingreso_acum_pct<=95,'B','C'))
    abc = d.groupby('clase').agg(Cargas=('load_id','count'),Ingreso_USD=('revenue','sum'))
    abc['% cargas'] = abc.Cargas/abc.Cargas.sum()*100
    abc['% ingreso'] = abc.Ingreso_USD/abc.Ingreso_USD.sum()*100
    a = d[d.clase=='A'].copy()
    states = a.groupby('state').agg(Cargas_A=('load_id','nunique'),Puntualidad=('puntual','mean'),Desviacion_min=('delay_min','mean')).reset_index()
    states['Puntualidad'] *= 100
    states['Participación (%)'] = states.Cargas_A/states.Cargas_A.sum()*100
    points = a.groupby(['facility_id','latitude','longitude','state']).agg(Cargas=('load_id','count')).reset_index()
    series = a[a.load_date.between('2022-01-03','2024-12-29')].set_index('load_date').resample('W-SUN').load_id.nunique().rename('num_loads_A')
    X = weekly_features(series); cols = list(X.columns)
    base = X.join(series).dropna(); tr = base[base.index.year<=2023]; te = base[base.index.year==2024]
    def new_model(alpha):
        return Pipeline([('scaler',StandardScaler()),('ridge',Ridge(alpha=alpha))])
    rows = []
    for alpha in np.logspace(-2,4,13):
        scores = []
        for it,iv in TimeSeriesSplit(n_splits=5).split(tr):
            m = new_model(alpha).fit(tr[cols].iloc[it],tr.num_loads_A.iloc[it])
            scores.append(mean_absolute_error(tr.num_loads_A.iloc[iv],m.predict(tr[cols].iloc[iv])))
        rows.append({'Alpha':float(alpha),'MAE validación':float(np.mean(scores))})
    cv = pd.DataFrame(rows); alpha = float(cv.loc[cv['MAE validación'].idxmin(),'Alpha'])
    model = new_model(alpha).fit(tr[cols],tr.num_loads_A)
    pred = np.clip(model.predict(te[cols]),0,None)
    ps = {'Ridge':pred,'Ingenuo':series.shift(1).loc[te.index],
          'Promedio móvil 4 semanas':series.rolling(4).mean().shift(1).loc[te.index],
          'Media entrenamiento':np.repeat(tr.num_loads_A.mean(),len(te))}
    comp = pd.DataFrame([{'Modelo':n,**metrics(te.num_loads_A,p)} for n,p in ps.items()])
    select = new_model(alpha).fit(base[base.index.year==2022][cols],base[base.index.year==2022].num_loads_A)
    validation = base[base.index.year==2023]
    pi = permutation_importance(select,validation[cols],validation.num_loads_A,scoring='neg_mean_absolute_error',n_repeats=30,random_state=SEED)
    importance = pd.DataFrame({'Variable':cols,'Aumento MAE':pi.importances_mean}).sort_values('Aumento MAE',ascending=False)
    reduced = importance.loc[importance['Aumento MAE']>0,'Variable'].head(4).tolist()
    if len(reduced)<2:
        reduced=importance.Variable.head(2).tolist()
    rmodel = new_model(alpha).fit(tr[reduced],tr.num_loads_A)
    reduction = pd.DataFrame([{'Modelo':'Completo','Variables':len(cols),**metrics(te.num_loads_A,pred)},
        {'Modelo':'Reducido','Variables':len(reduced),**metrics(te.num_loads_A,np.clip(rmodel.predict(te[reduced]),0,None))}])
    return {'abc':abc,'pareto':d[['cargas_acum_pct','ingreso_acum_pct']].iloc[::40],
        'states':states,'points':points,'series':series,'alpha':alpha,'cv':cv,'comparison':comp,
        'predictions':pd.DataFrame({'Real':te.num_loads_A,**ps}), 'model':model,
        'coefficients':pd.DataFrame({'Variable':cols,'Coeficiente estandarizado':model.named_steps['ridge'].coef_}),
        'features':cols,'reduction':reduction,'importance':importance,'n_train':len(tr),'n_test':len(te),
        'punctual':float(a.dropna(subset=['actual_datetime','scheduled_datetime']).puntual.mean()*100),
        'eda':d[['revenue','weight_lbs','pieces','delay_min']].describe().T}


def dimension4(t):
    de = t['delivery_events'].query("event_type == 'Delivery'").copy()
    de['scheduled_datetime'] = pd.to_datetime(de.scheduled_datetime)
    de = de[de.scheduled_datetime.between('2022-01-01','2024-12-31 23:59:59')]
    monthly = de.resample('MS',on='scheduled_datetime').size().astype(float).rename('Entregas')
    if len(monthly)!=36:
        raise ValueError('Se esperan 36 meses completos entre 2022 y 2024.')
    train, test = monthly.iloc[:24], monthly.iloc[24:]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = SARIMAX(train,order=(1,0,0),seasonal_order=(1,0,0,12),trend='c').fit(disp=False,maxiter=200)
        forecast = fit.get_forecast(steps=12); pred = forecast.predicted_mean
        full = SARIMAX(monthly,order=(1,0,0),seasonal_order=(1,0,0,12),trend='c').fit(disp=False,maxiter=200)
        future = full.get_forecast(steps=12)
        hw = ExponentialSmoothing(train,trend='add',seasonal='add',seasonal_periods=12,initialization_method='estimated').fit().forecast(12)
        adf = adfuller(monthly,regression='c',autolag='AIC')
        kp = kpss(monthly,regression='c',nlags='auto')
    preds={'SARIMA con constante':pred,'Seasonal Naïve':train.iloc[-12:].to_numpy(),'Holt-Winters aditivo':hw,'Media histórica':np.repeat(train.mean(),12)}
    comparison=pd.DataFrame([{'Modelo':n,**metrics(test,p)} for n,p in preds.items()])
    configs=[]
    for order,seasonal,trend in [((1,0,0),(1,0,0,12),'n'),((1,0,0),(1,0,0,12),'c'),((1,0,0),(0,0,0,0),'c'),((0,0,0),(1,0,0,12),'c'),((2,0,0),(1,0,0,12),'c'),((2,0,0),(0,0,0,0),'c')]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m=SARIMAX(train,order=order,seasonal_order=seasonal,trend=trend).fit(disp=False,maxiter=200)
        configs.append({'Configuración':f'{order}{seasonal} / {trend}','AIC':m.aic,
                        'Convergió':bool(m.mle_retvals.get('converged',False)),**metrics(test,m.forecast(12))})
    dm=t['driver_monthly_metrics']; p95=float(dm.trips_completed.quantile(.95))
    resources={'Conductores':int(dm.driver_id.nunique()),'Camiones':int((t['trucks'].status=='Active').sum()),
               'Tráileres':int((t['trailers'].status=='Active').sum())}
    cap=int(min(resources.values())*p95)
    ci=forecast.conf_int(); fci=future.conf_int()
    # PCA exploratorio, idéntico papel descriptivo que en el cuaderno; no alimenta SARIMA.
    de['mes']=de.scheduled_datetime.dt.to_period('M')
    de['retraso_min']=(pd.to_datetime(de.actual_datetime)-de.scheduled_datetime).dt.total_seconds()/60
    X=de.groupby('mes').agg(entregas=('event_id','size'),detencion_prom=('detention_minutes','mean'),
        puntualidad=('on_time_flag','mean'),retraso_prom=('retraso_min','mean'),instalaciones_destino=('facility_id','nunique'))
    dm=dm.copy();dm['mes']=pd.to_datetime(dm.month).dt.to_period('M')
    X=X.join(dm.groupby('mes').agg(conductores=('driver_id','nunique'),viajes_conductor=('trips_completed','mean'),
        viajes_totales=('trips_completed','sum'),millas=('total_miles','sum'),ingresos=('total_revenue','sum'),
        mpg=('average_mpg','mean'),galones=('total_fuel_gallons','sum'),otd_conductor=('on_time_delivery_rate','mean'),ralenti=('average_idle_hours','mean')))
    X['dias_mes']=X.index.days_in_month
    docks=int(t['facilities'].dock_doors.sum())
    X['utilizacion']=monthly.to_numpy()/cap*100;X['brecha']=cap-monthly.to_numpy()
    X['entregas_muelle']=X.entregas/docks;X['muelles_totales']=docks;X['n_instalaciones']=len(t['facilities'])
    constants=X.columns[X.std()<1e-9].tolist()
    remaining=X.drop(columns=constants+['utilizacion','brecha','entregas_muelle']).dropna()
    pca=PCA().fit(StandardScaler().fit_transform(remaining))
    return {'monthly':monthly,'comparison':comparison,'configurations':pd.DataFrame(configs),
        'predictions':pd.DataFrame({'Real':test,**preds,'Límite inferior 95%':ci.iloc[:,0],'Límite superior 95%':ci.iloc[:,1]}),
        'future':pd.DataFrame({'Pronóstico':future.predicted_mean,'Límite inferior 95%':fci.iloc[:,0],'Límite superior 95%':fci.iloc[:,1]}),
        'resources':resources,'p95':p95,'capacity':cap,'maintenance':int((t['trucks'].status=='Maintenance').sum()),
        'docks':docks,'stationarity':pd.DataFrame([{'Prueba':'ADF','Estadístico':adf[0],'p-valor':adf[1]},
                                               {'Prueba':'KPSS','Estadístico':kp[0],'p-valor':kp[1]}]),
        'residuals':pd.Series(fit.resid,index=train.index), 'ljung':acorr_ljungbox(fit.resid,lags=[6,12],return_df=True),
        'coverage':int(((test>=ci.iloc[:,0])&(test<=ci.iloc[:,1])).sum()),
        'converged':bool(fit.mle_retvals.get('converged',False)),'full_converged':bool(full.mle_retvals.get('converged',False)),
        'pca':pd.DataFrame({'Componente':np.arange(1,len(pca.explained_variance_ratio_)+1),
                           'Varianza (%)':pca.explained_variance_ratio_*100,'Acumulada (%)':np.cumsum(pca.explained_variance_ratio_)*100}),
        'correlation':X.corr(),'constants':constants,'pca_columns':list(remaining.columns)}
