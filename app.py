from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from routing import solve_pilot

ROOT=Path(__file__).resolve().parent
TEAL, NAVY, ORANGE = '#087E8B', '#142F46', '#E58A42'
PALETTE=[TEAL,ORANGE,'#4778A8','#B76079','#7B8E4E']
st.set_page_config(page_title='Nexo · Logística predictiva | Grupo 5',page_icon='🚚',layout='wide')
st.markdown('''<style>
.block-container{padding-top:4.5rem;max-width:1440px;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.035em}h1{font-size:2.55rem!important}
[data-testid="stSidebar"]{background:#142F46}
[data-testid="stSidebar"] *{color:#F0F7FA}
[data-testid="stSidebar"] [data-baseweb="radio"]{padding:.4rem 0}
[data-testid="stMetric"]{background:white;border:1px solid #E0E7EC;border-radius:14px;padding:18px}
[data-testid="stMetricLabel"]{color:#536B7C}
[data-testid="stMetricValue"]{font-size:clamp(1.05rem,2vw,1.8rem)!important}
@media(max-width:1100px){[data-testid="stMetric"]{padding:10px}[data-testid="stMetricValue"]{font-size:1.05rem!important}}
.eyebrow{font-size:.75rem;letter-spacing:.15em;font-weight:700;color:#087E8B;margin-bottom:.7rem}
.hero{border-radius:18px;padding:30px 34px;background:#142F46;color:#F8FBFE;margin:1rem 0 1.5rem}
.hero h2{color:white;margin:0;font-size:1.65rem}.hero p{color:#CADBE6;max-width:800px;margin-bottom:0}
.footer{border-top:1px solid #DCE4EB;padding-top:16px;margin-top:40px;color:#607789;font-size:.8rem}
</style>''',unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def artifact(n):
    return joblib.load(ROOT/'artifacts'/f'dim{n}.joblib')


@st.cache_data(show_spinner=False)
def manifest():
    return json.loads((ROOT/'artifacts/manifest.json').read_text(encoding='utf-8'))


def plot(fig, key=None, height=390):
    fig.update_layout(height=height,font=dict(family='Arial',color=NAVY),paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',margin=dict(l=20,r=20,t=45,b=25),legend_title_text='',
        colorway=PALETTE)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor='#ECF0F4')
    st.plotly_chart(fig,width="stretch",key=key)


def table(df, key=None):
    st.dataframe(df,hide_index=not isinstance(df.index,pd.DatetimeIndex),width="stretch",key=key)


def download(df,name,label='Descargar tabla CSV'):
    st.download_button(label,df.to_csv(index=True).encode('utf-8-sig'),file_name=name,mime='text/csv',key=name)


def cards(items):
    for col,(label,value,help_) in zip(st.columns(len(items)),items):
        col.metric(label,value,help=help_)


def heading(k,title,subtitle):
    st.markdown(f'<div class="eyebrow">GRUPO 5 / DIMENSIÓN {k}</div>',unsafe_allow_html=True)
    st.title(title);st.write(subtitle)


def line(df,title,ylabel=''):
    return px.line(df,x=df.index,y=list(df.columns),title=title,labels={'value':ylabel,'index':'Fecha','variable':'Serie'})


def metric_cards(m,unit):
    cards([('MAE',f"{m['MAE']:.2f} {unit}",'Error absoluto medio en el conjunto de prueba.'),
           ('RMSE',f"{m['RMSE']:.2f} {unit}",'Penaliza más los errores grandes.'),
           ('R²',f"{m['R²']:.4f}",'Coeficiente de determinación; no es un porcentaje de aciertos.')])


def page_home():
    st.markdown('<div class="eyebrow">MODELAMIENTO PREDICTIVO / UNIVERSIDAD DE LIMA</div>',unsafe_allow_html=True)
    st.title('Nexo · Logística predictiva')
    st.markdown('<div class="hero"><h2>De los datos a la decisión de reparto.</h2><p>Cuatro dimensiones para anticipar tiempos de viaje, comparar secuencias de entrega, priorizar cargas y contrastar demanda con capacidad.</p></div>',unsafe_allow_html=True)
    m=manifest()
    cards([('Viajes históricos',f"{m['tables']['trips']['filas']:,}",'Base de viajes del caso académico.'),
           ('Instalaciones',str(m['tables']['facilities']['filas']),'Puntos de origen y destino del conjunto de datos.'),
           ('Dimensiones','4','Modelos corregidos integrados en una sola aplicación.'),
           ('Período del estudio','2022–2025','Validación histórica 2024 y escenario de proyección 2025; no representa la operación actual.')])
    st.subheader('Explora el proyecto')
    for cols,indices in [(st.columns(2),[0,1]),(st.columns(2),[2,3])]:
        for col,i in zip(cols,indices):
            with col.container(border=True):
                st.markdown(f'**0{i+1} / {TITLES[i]}**')
                st.write(DESCRIPTIONS[i])
                st.button('Explorar dimensión',key=f'open_{i}',width="stretch",
                          on_click=lambda n=i+1: st.session_state.update(nav=PAGES[n]))
    st.info('Los resultados se recalcularon desde los CSV y los cuadernos corregidos. Puedes descargar las tablas de cada dimensión y consultar la metodología para sustentar el proyecto.')
    with st.expander('Equipo y fuentes'):
        st.write('Eber Brando Bravo Vega · Jade Mariana Angelica Santa Cruz Lorenzo · Dulce Carito Tapia Quispe · Gabriela Ximena Vilchez Sotelo')
        st.link_button('Base de datos del estudio',m['dataset'])
        st.link_button('Informe de trabajo','https://docs.google.com/document/d/1JcC0SIsz6okcqzZyO0d1mvDaLVhLwB9N/edit')
        st.write('Aplicación académica. Los ahorros del piloto son calculados dentro del escenario, sin validación de campo.')


def page_dim1():
    d=artifact(1)
    heading(1,'Duración y desempeño','Predice la duración de un viaje con las características conocidas de su ruta, carga y fecha de despacho.')
    metric_cards(d['metrics'],'h')
    st.caption(f"XGBoost · {d['n_train']:,} viajes de entrenamiento / {d['n_test']:,} de prueba · partición aleatoria 80/20 del cuaderno, semilla 42.")
    tabs=st.tabs(['Predicción de un viaje','Evaluación del modelo','Desempeño histórico','Método'])
    with tabs[0]:
        st.subheader('Estima la duración de un despacho')
        with st.form('xgb_form'):
            a,b,c=st.columns(3)
            routes=d['routes'].set_index('route_id',drop=False)
            rid=a.selectbox('Ruta',routes.index.tolist(),format_func=lambda x:f"{x} · {routes.loc[x,'origin_city']} → {routes.loc[x,'destination_city']}")
            weight=b.number_input('Peso de carga (lb)',min_value=1.0,max_value=150000.0,value=float(d['defaults']['weight_lbs']),step=500.)
            pieces=c.number_input('Piezas',min_value=1,max_value=2000,value=int(d['defaults']['pieces']))
            load=a.selectbox('Tipo de carga',d['load_types'])
            booking=b.selectbox('Tipo de reserva',d['booking_types'])
            date=c.date_input('Fecha de despacho',value=pd.Timestamp('2025-01-06').date())
            submit=st.form_submit_button('Calcular duración',type='primary',width="stretch")
        if submit:
            route=routes.loc[rid]
            values={c:route[c] for c in ['route_id','origin_city','destination_city','typical_distance_miles','typical_transit_days']}
            values.update(weight_lbs=weight,pieces=pieces,load_type=load,booking_type=booking,
                          dispatch_month=date.month,dispatch_day=date.weekday())
            estimate=float(d['model'].predict(pd.DataFrame([values])[d['features']])[0])
            st.success(f'Duración estimada: **{estimate:.2f} horas** ({estimate/24:.2f} días).')
            st.caption('Estimación de duración total del viaje; no calcula por sí sola la puntualidad ni un ETA con tráfico en tiempo real.')
    with tabs[1]:
        a,b=st.columns(2)
        with a:
            df=d['predictions'].sample(min(2500,len(d['predictions'])),random_state=42)
            fig=px.scatter(df,x='Real (h)',y='Predicción (h)',opacity=.35,title='Real frente a predicho',color_discrete_sequence=[TEAL])
            low=float(df.min().min());high=float(df.max().max())
            fig.add_trace(go.Scatter(x=[low,high],y=[low,high],mode='lines',name='Predicción ideal',line=dict(color=ORANGE,dash='dash')));plot(fig)
        with b:
            imp=d['importance'].head(12).sort_values('Importancia (%)')
            plot(px.bar(imp,x='Importancia (%)',y='Variable',orientation='h',title='Variables de mayor contribución'))
        download(d['predictions'],'dimension1_predicciones.csv');table(d['importance'])
    with tabs[2]:
        cards([('Entregas a tiempo',f"{d['otd_flag']:.2f}%",'Indicador on_time_flag del dataset; puede tener una tolerancia distinta al horario exacto.'),
               ('Utilización >100%',f"{d['util_above100']:.2f}%",'Registros del indicador original; se muestran sin recortar ni reinterpretar como ocupación física.')])
        plot(line(d['util_monthly'].to_frame(),'Utilización operativa registrada','%'))
        plot(px.histogram(d['delay'],nbins=50,title='Desviación real menos programada',labels={'value':'Horas'},color_discrete_sequence=[TEAL]))
        st.write('Rendimiento de combustible: resumen descriptivo')
        table(d['fuel'].reset_index(names='Variable').round(3))
    with tabs[3]:
        st.write('**Objetivo:** actual_duration_hours. El modelo usa distancia y duración típicas, peso, piezas, tipo de carga/reserva, ruta, ciudades y mes/día del despacho. No usa combustible consumido, duración real o distancia observada como entradas de este predictor.')
        st.write('XGBoost: 500 árboles, learning_rate 0.03, profundidad 5, min_child_weight 3, subsample 0.8 y colsample_bytree 0.8. Se imputan nulos y se codifican categorías con one-hot.')
        st.write('La prueba aleatoria reproduce el cuaderno; no mide generalización a rutas inéditas ni constituye una validación cronológica independiente. La importancia de una variable no demuestra causalidad.')
        table(d['eda'].reset_index(names='Variable').round(3))


def page_dim2():
    d=artifact(2)
    heading(2,'Rutas y escenarios de flota','Compara la asignación y secuencia de las mismas seis entregas con tres y cinco camiones.')
    cards([('Pedidos del piloto','6','Misma fecha de despacho y depósito común.'),
           ('Carga del piloto',f"{d['pilot'].weight_lbs.sum():,.0f} lb",'Cargas históricas seleccionadas por la regla del cuaderno.'),
           ('Fecha del piloto',str(d['pilot'].dispatch_date.iloc[0].date()),'Escenario histórico; no son pedidos actuales.')])
    tabs=st.tabs(['Optimizar secuencias','Modelo de duración','Agrupamiento exploratorio','Método'])
    with tabs[0]:
        with st.form('routing_form'):
            a,b,c=st.columns(3)
            capacity=a.number_input('Capacidad por camión (lb)',min_value=1000,max_value=150000,value=45000,step=1000)
            speed=b.number_input('Velocidad supuesta (mi/h)',min_value=5.,max_value=90.,value=50.,step=5.)
            service=c.number_input('Servicio por entrega (h)',min_value=0.,max_value=8.,value=.5,step=.25)
            calculate=st.form_submit_button('Comparar 3 y 5 camiones',type='primary',width="stretch")
        if calculate or 'routing_result' not in st.session_state:
            st.session_state.routing_result=solve_pilot(d['pilot'],capacity,speed,service)
            st.session_state.routing_settings=(capacity,speed,service)
        r=st.session_state.routing_result
        st.caption('Supuestos del resultado mostrado: capacidad {:,} lb, velocidad {} mi/h y servicio {} h/parada.'.format(*st.session_state.routing_settings))
        table(r['comparison'].round(3))
        if r['details'].empty:
            st.warning('La capacidad indicada no permite atender este piloto con las flotas comparadas. Aumenta la capacidad para encontrar una asignación factible.')
        else:
            choices=[k for k in [3,5] if (k,'Optimizada') in r['paths']]
            a,b=st.columns([1,3]);n=a.selectbox('Flota a visualizar',choices,format_func=lambda x:f'{x} camiones')
            kind=a.radio('Secuencia',['Optimizada','Referencia'])
            fig=go.Figure()
            for j,path in enumerate(r['paths'][n,kind],1):
                seq=[0]+[i+1 for i in path]+[0]
                fig.add_trace(go.Scatter(x=r['lon'][seq],y=r['lat'][seq],mode='lines+markers+text',
                    text=['Depósito']+[str(i+1) for i in path]+[''],textposition='top center',name=f'Camión {j}'))
            fig.add_trace(go.Scatter(x=[r['lon'][0]],y=[r['lat'][0]],mode='markers',marker=dict(symbol='square',size=13,color=NAVY),name='Depósito'))
            fig.update_layout(title='Secuencias geodésicas · cada pedido tiene un número',xaxis_title='Longitud',yaxis_title='Latitud')
            with b:plot(fig,key='route_map',height=420)
            table(r['details'].query('Camiones == @n').round(3));download(r['details'],'dimension2_secuencias.csv')
        st.info('Se minimiza la distancia total con N rutas no vacías y capacidad limitada. Las líneas representan distancias geodésicas, no carreteras. Reducir distancia total no garantiza reducir el tiempo del camión que termina último.')
        with st.expander('Ver los seis pedidos originales'):
            table(d['pilot'][['trip_id','load_id','pickup_facility','delivery_facility','weight_lbs','delivery_scheduled']])
    with tabs[1]:
        table(d['comparison'].round(4))
        st.caption('Random Forest · entrenamiento 2022, selección de variables en 2023 y evaluación final en 2024. El ajuste final usa 2022–2023.')
        a,b=st.columns(2)
        with a:plot(px.bar(d['importance'].sort_values('Aumento MAE (h)'),x='Aumento MAE (h)',y='Variable',orientation='h',title='Importancia por permutación en validación'))
        with b:
            df=d['predictions'].sample(min(2000,len(d['predictions'])),random_state=42)
            plot(px.scatter(df,x='Real (h)',y='Reducido',opacity=.3,title='Modelo reducido: real frente a predicho'))
        st.write('Variables seleccionadas: **'+', '.join(d['reduced'])+'**.')
        with st.form('rf_simulator'):
            a,b,c=st.columns(3)
            typical=a.number_input('Distancia típica (mi)',min_value=0.,value=float(d['pilot'].typical_distance_miles.iloc[0]))
            days=b.number_input('Tránsito típico (días)',min_value=0.,value=float(d['pilot'].typical_transit_days.iloc[0]))
            weight=c.number_input('Peso (lb)',min_value=1.,value=float(d['pilot'].weight_lbs.iloc[0]),key='rf_weight')
            pieces=a.number_input('Piezas',min_value=1,value=int(d['pilot'].pieces.iloc[0]),key='rf_pieces')
            geo=b.number_input('Distancia geodésica origen/destino (mi)',min_value=0.,value=float(d['pilot'].distancia_geodesica_millas.iloc[0]))
            run=st.form_submit_button('Estimar duración histórica comparable')
        if run:
            x=pd.DataFrame([dict(typical_distance_miles=typical,typical_transit_days=days,weight_lbs=weight,pieces=pieces,distancia_geodesica_millas=geo)])
            pred=d['models']['Reducido'].predict(x[d['reduced']])[0]
            st.success(f'Duración estimada del viaje: **{pred:.2f} h**.')
        download(d['comparison'],'dimension2_modelos.csv')
    with tabs[2]:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        n=st.select_slider('Número de grupos',options=[3,5])
        pilot=d['pilot'].copy()
        pilot['Grupo']=(KMeans(n_clusters=n,random_state=42,n_init=10).fit_predict(StandardScaler().fit_transform(pilot[['destino_latitude','destino_longitude']]))+1).astype(str)
        plot(px.scatter(pilot,x='destino_longitude',y='destino_latitude',color='Grupo',hover_data=['delivery_facility','weight_lbs'],title='Grupos de destinos del piloto'))
        st.write('K-Means es una exploración espacial. Las secuencias y la capacidad se resuelven por separado en el optimizador exacto.')
    with tabs[3]:
        st.write('Se integran viajes, cargas, rutas e instalaciones con una sola fila por viaje. La selección temporal usa cinco variables conocidas antes del viaje y conserva las tres más importantes en 2023.')
        st.write('Random Forest: 55 árboles, profundidad 10, mínimo 3 observaciones por hoja, semilla 42. La optimización enumera los órdenes de cada subconjunto de pedidos y usa programación dinámica para encontrar la menor distancia factible con tres/cinco camiones.')
        st.write('El Random Forest estima viajes históricos; no se suman sus predicciones como si fueran tiempos validados de las nuevas secuencias. Para estas se usa distancia / velocidad + servicio. Capacidad de 45.000 lb es un supuesto editable.')
        st.write('El piloto no incorpora carreteras, tráfico, ventanas de tiempo, jornadas legales ni costos fijos. El óptimo es exacto dentro de esas hipótesis y para los seis pedidos del estudio.')


def page_dim3():
    d=artifact(3)
    heading(3,'Cargas prioritarias','Identifica la concentración de ingresos y estima el número semanal de cargas Clase A.')
    m=d['comparison'].query("Modelo == 'Ridge'").iloc[0]
    cards([('Cargas Clase A',f"{int(d['abc'].loc['A','Cargas']):,}",'Clasificación retrospectiva por ingreso acumulado hasta el 80%.'),
           ('MAE de Ridge',f"{m['MAE']:.2f} cargas",'Evaluación de 52 semanas de 2024.'),
           ('Alpha elegido',f"{d['alpha']:g}",'Menor MAE en cinco cortes de validación temporal del entrenamiento.'),
           ('Puntualidad exacta A',f"{d['punctual']:.2f}%",'Entrega real menor o igual al horario programado; distinta del flag OTD del dataset.')])
    tabs=st.tabs(['ABC y destinos','Pronóstico semanal','Escenario de próxima semana','Método'])
    with tabs[0]:
        a,b=st.columns(2)
        with a:
            fig=px.line(d['pareto'],x='cargas_acum_pct',y='ingreso_acum_pct',title='Pareto de ingresos',labels={'cargas_acum_pct':'Cargas acumuladas (%)','ingreso_acum_pct':'Ingresos acumulados (%)'})
            fig.add_hline(y=80,line_dash='dash',line_color=ORANGE);fig.add_hline(y=95,line_dash='dot');plot(fig)
        with b:
            state=d['states'].sort_values('Cargas_A').tail(12)
            plot(px.bar(state,x='Cargas_A',y='state',orientation='h',title='Destinos con más cargas Clase A',labels={'state':'Estado','Cargas_A':'Cargas'}))
        table(d['abc'].reset_index().round(2))
        points=d['points']
        plot(px.scatter(points,x='longitude',y='latitude',size='Cargas',color='state',hover_data=['facility_id','Cargas'],title='Concentración por instalación de destino',labels={'longitude':'Longitud','latitude':'Latitud'}),height=420)
        download(d['states'],'dimension3_estados.csv')
    with tabs[1]:
        plot(line(d['predictions'],'Evaluación a una semana · 2024','Cargas Clase A'))
        table(d['comparison'].round(4))
        st.caption(f"{d['n_train']} semanas de entrenamiento / {d['n_test']} de prueba. Cada semana de prueba utiliza las semanas previas observadas; no es un pronóstico fijo de 52 pasos hecho al inicio de 2024.")
        a,b=st.columns(2)
        with a:
            fig=px.line(d['cv'],x='Alpha',y='MAE validación',log_x=True,markers=True,title='Selección de la penalización L2');plot(fig)
        with b:
            co=d['coefficients'].sort_values('Coeficiente estandarizado')
            plot(px.bar(co,x='Coeficiente estandarizado',y='Variable',orientation='h',title='Coeficientes de Ridge'))
        st.write('Comparación con reducción de variables');table(d['reduction'].round(4))
        download(d['predictions'],'dimension3_prueba_2024.csv')
    with tabs[2]:
        st.write('Cambia las cuatro semanas anteriores para evaluar una semana próxima al cierre de la serie. Es un escenario de sensibilidad del modelo entrenado, no una garantía de demanda.')
        with st.form('ridge_form'):
            cols=st.columns(4);lags=[]
            for i,col in enumerate(cols,1):
                lags.append(col.number_input(f'Hace {i} semana(s): cargas A',min_value=0,value=int(d['series'].iloc[-i]),key=f'lag_{i}'))
            date=st.date_input('Semana objetivo (domingo)',value=pd.Timestamp('2025-01-05').date())
            submit=st.form_submit_button('Estimar cargas prioritarias',type='primary')
        if submit:
            date=pd.Timestamp(date)
            if date.dayofweek!=6:
                st.warning('Selecciona un domingo: la serie está agregada por semanas que terminan ese día.')
            else:
                week=date.isocalendar().week
                x={f'lag_{i+1}':v for i,v in enumerate(lags)}
                x.update(rolling_mean_4=np.mean(lags),trend=int((date-d['series'].index[0]).days//7),
                         sin_anual=np.sin(2*np.pi*week/52),cos_anual=np.cos(2*np.pi*week/52))
                p=max(0,float(d['model'].predict(pd.DataFrame([x])[d['features']])[0]))
                st.success(f'Nivel esperado: **{p:.1f} cargas Clase A**. Para planificar unidades enteras: {int(np.ceil(p))}.')
                st.caption('Se mantiene el ajuste 2022–2023 utilizado en la evaluación; no se inventa una métrica de precisión para este escenario.')
    with tabs[3]:
        st.write('Ridge con StandardScaler; rezagos 1–4, promedio móvil de cuatro semanas, tendencia, seno y coseno anual. Se evalúan 13 alphas entre 0.01 y 10.000 con TimeSeriesSplit de cinco cortes.')
        st.write('ABC y mapa son análisis retrospectivos, calculados con todos los ingresos disponibles; no son un optimizador de rutas ni una clasificación ex ante validada. El backtest semanal se interpreta condicionado a esa clasificación retrospectiva.')
        st.write('El R² bajo y la mejora moderada frente a la media histórica indican que el modelo describe el nivel semanal mejor que los picos. No se ofrece un intervalo de pronóstico que el cuaderno no haya calculado.')


def page_dim4():
    d=artifact(4)
    heading(4,'Demanda y capacidad','Proyección del escenario 2025 con SARIMA, contrastada con los recursos del conjunto de datos.')
    m=d['comparison'].query("Modelo == 'SARIMA con constante'").iloc[0]
    cards([('Entregas 2022–2024',f"{int(d['monthly'].sum()):,}",'Solo eventos Delivery por fecha programada; enero 2025 excluido por incompleto.'),
           ('MAPE en 2024',f"{m['MAPE (%)']:.2f}%",'12 meses fuera del ajuste inicial.'),
           ('Capacidad de referencia',f"{d['capacity']:,}/mes",'Mínimo de recursos × productividad P95 del conjunto histórico.'),
           ('Cobertura del intervalo',f"{d['coverage']}/12 meses",'Cobertura observada en 2024 del intervalo de pronóstico nominal del 95%.')])
    tabs=st.tabs(['Escenario de capacidad','Validación 2024','Diagnósticos y PCA','Método'])
    with tabs[0]:
        with st.form('capacity_form'):
            a,b,c,e=st.columns(4)
            trucks=a.number_input('Camiones activos',min_value=1,value=d['resources']['Camiones'])
            drivers=b.number_input('Conductores disponibles',min_value=1,value=d['resources']['Conductores'])
            trailers=c.number_input('Tráileres activos',min_value=1,value=d['resources']['Tráileres'])
            productivity=e.number_input('Entregas por recurso / mes',min_value=1.,value=d['p95'],step=1.)
            submit=st.form_submit_button('Evaluar capacidad',type='primary',width="stretch")
        if submit or 'capacity_settings' not in st.session_state:
            st.session_state.capacity_settings=(trucks,drivers,trailers,productivity)
        trucks,drivers,trailers,productivity=st.session_state.capacity_settings
        capacity=int(min(trucks,drivers,trailers)*productivity)
        df=d['future'].copy();df['Capacidad']=capacity
        df['Utilización (%)']=df['Pronóstico']/capacity*100;df['Brecha']=capacity-df['Pronóstico']
        deficit=max(0,float(df['Pronóstico'].max()-capacity));upper_deficit=max(0,float(df['Límite superior 95%'].max()-capacity))
        extra=int(np.ceil(deficit/productivity));upper_extra=int(np.ceil(upper_deficit/productivity))
        cards([('Capacidad del escenario',f'{capacity:,}/mes','Se actualiza al evaluar los controles.'),
               ('Meses sobre capacidad',str(int((df['Pronóstico']>capacity).sum())),'Comparación contra el pronóstico central.'),
               ('Máximo déficit central',f'{deficit:.0f} entregas','El máximo déficit mensual proyectado, no su suma anual.')])
        fig=go.Figure();fig.add_trace(go.Scatter(x=d['monthly'].index,y=d['monthly'],name='Histórico',line=dict(color='#8A9BAC')))
        fig.add_trace(go.Scatter(x=df.index,y=df['Límite superior 95%'],line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=df.index,y=df['Límite inferior 95%'],fill='tonexty',fillcolor='rgba(8,126,139,.14)',line=dict(width=0),name='Intervalo de pronóstico 95%'))
        fig.add_trace(go.Scatter(x=df.index,y=df['Pronóstico'],mode='lines+markers',name='Escenario 2025',line=dict(color=TEAL)))
        fig.add_hline(y=capacity,line_color=ORANGE,line_dash='dash',annotation_text=f'Capacidad {capacity:,}')
        fig.update_layout(title='Demanda mensual y capacidad del escenario',yaxis_title='Entregas');plot(fig,height=430)
        table(df.round(2));download(df,'dimension4_proyeccion_capacidad.csv')
        names=['camiones','conductores','tráileres'];values=[trucks,drivers,trailers]
        bottlenecks=[name for name,v in zip(names,values) if v==min(values)]
        st.write(f"Recurso limitante: **{', '.join(bottlenecks)}**. Recursos requeridos para cubrir el pico mensual bajo la productividad indicada:")
        need=int(np.ceil(float(df['Pronóstico'].max())/productivity))
        upper_need=int(np.ceil(float(df['Límite superior 95%'].max())/productivity))
        table(pd.DataFrame([{'Recurso':name.capitalize(),'Disponibles':v,
                            'Adicionales (pico central)':max(0,need-v),
                            'Adicionales (límite 95%)':max(0,upper_need-v)} for name,v in zip(names,values)]))
        if bottlenecks==['camiones']:
            st.caption(f'El dataset registra {d["maintenance"]} camiones en mantenimiento. Su eventual retorno requiere comprobar disponibilidad real; no se asume que ya estén operativos.')
        st.caption('Productividad de referencia: P95 de viajes por conductor y mes, aplicado como proxy a cada recurso. Es una capacidad estimada bajo equivalencia viaje/entrega, no una medición certificada de capacidad física.')
    with tabs[1]:
        cols=['Real','SARIMA con constante','Seasonal Naïve','Holt-Winters aditivo','Media histórica']
        plot(line(d['predictions'][cols],'Comparación de pronósticos · 2024','Entregas'))
        table(d['comparison'].round(4));download(d['comparison'],'dimension4_comparacion.csv')
        with st.expander('Comparación de seis configuraciones SARIMA'):
            table(d['configurations'].round(4))
            st.caption('AIC calculado en 2022–2023; métricas en 2024. La comparación sobre 2024 participa en la elección del modelo y no sustituye una segunda prueba independiente.')
    with tabs[2]:
        a,b=st.columns(2)
        with a:
            st.write('Estacionariedad de los 36 meses');table(d['stationarity'].round(6))
            st.caption('ADF: H0 raíz unitaria. KPSS: H0 estacionariedad; su p-valor puede estar limitado por tablas.')
            st.write('Ljung–Box de residuos del ajuste 2022–2023');table(d['ljung'].reset_index(names='Rezago').round(6))
        with b:plot(line(d['residuals'].rename('Residuo').to_frame(),'Residuos del ajuste','Entregas'))
        plot(px.bar(d['pca'],x='Componente',y='Varianza (%)',title='PCA exploratorio de variables mensuales'))
        table(d['pca'].round(3))
        st.caption('El PCA es descriptivo; no se usa como entrada del SARIMA univariado. Se retiran constantes y transformaciones redundantes de entregas antes de calcularlo.')
        with st.expander('Correlaciones de las variables candidatas'):
            plot(px.imshow(d['correlation'],color_continuous_scale='RdBu_r',zmin=-1,zmax=1),height=640)
    with tabs[3]:
        st.write('SARIMA(1,0,0)(1,0,0)[12] con constante. Ajuste 2022–2023 (24 meses), evaluación de 2024 (12 meses) y reentrenamiento con 36 meses para el escenario de 2025. Máximo 200 iteraciones.')
        st.write('La constante permite aprender el nivel medio. Se contrastan Seasonal Naïve, Holt-Winters aditivo y media histórica; se mantienen los mismos cortes y parámetros del cuaderno corregido.')
        st.write('Solo hay dos ciclos anuales en el entrenamiento inicial. El modelo no incorpora explícitamente días del mes, cambios de recursos ni variables externas. El intervalo expresa incertidumbre del modelo y no asegura cobertura futura.')
        st.write(f'Convergencia del ajuste inicial: {d["converged"]}. Convergencia del ajuste de 36 meses: {d["full_converged"]}.')


TITLES=['Duración y desempeño','Rutas y escenarios de flota','Cargas prioritarias','Demanda y capacidad']
DESCRIPTIONS=['XGBoost para duración de viajes y diagnóstico de la operación histórica.',
              'Random Forest temporal y piloto de secuencias optimizadas con 3 y 5 camiones.',
              'Clasificación ABC, concentración geográfica y Ridge para cargas semanales.',
              'SARIMA y comparación de demanda mensual frente a recursos disponibles.']
PAGES=['Inicio']+[f'0{i+1} · {title}' for i,title in enumerate(TITLES)]+['Datos y reproducibilidad']
with st.sidebar:
    st.markdown('## NEXO\nLogística predictiva')
    st.caption('GRUPO 5 · MODELAMIENTO / 806')
    selected=st.radio('Navegación',PAGES,key='nav',label_visibility='collapsed')
    st.divider();st.write('Caso académico\n\nHistórico 2022–2024\n\nEscenario 2025')
if not (ROOT/'artifacts/manifest.json').exists():
    st.error('Faltan los resultados preparados. Desde la carpeta del proyecto ejecuta: python prepare.py. Incluye artifacts/ al publicar.')
    st.stop()
if selected=='Inicio':page_home()
elif selected==PAGES[1]:page_dim1()
elif selected==PAGES[2]:page_dim2()
elif selected==PAGES[3]:page_dim3()
elif selected==PAGES[4]:page_dim4()
else:
    st.title('Datos y reproducibilidad')
    st.write('Base empaquetada con la aplicación; no se descargan datos de Kaggle ni se solicitan credenciales durante su uso.')
    m=manifest();table(pd.DataFrame([{'Tabla':n,**v} for n,v in m['tables'].items()]))
    st.link_button('Fuente original',m['dataset'])
    with st.expander('Versiones verificadas'):
        st.json(m['versions'])
    st.download_button('Descargar métricas verificadas',(ROOT/'artifacts/metricas_verificadas.csv').read_bytes(),'metricas_verificadas.csv','text/csv')
    st.caption('Para recalcular desde los CSV: python prepare.py. Los cuadernos originales se conservan en fuentes/ y la trazabilidad de datos en fuentes/datos_manifest.json.')
st.markdown('<div class="footer">NEXO · Grupo 5 · Universidad de Lima · Resultados académicos reproducibles</div>',unsafe_allow_html=True)
