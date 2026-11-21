# Nexo · Logística predictiva — Grupo 5

Aplicación Streamlit de las cuatro dimensiones corregidas. Incluye modelos preparados, datos originales y cuadernos de referencia. El sitio utiliza los resultados recalculados; no copia las cifras del informe como resultados del programa.

## Abrir la aplicación

Con Python 3.12, desde esta carpeta:

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

También se incluye `INICIAR_APP.bat` para Windows con Python instalado. La primera ejecución instala las dependencias; las siguientes reutilizan el entorno.

Los artefactos de `artifacts/` ya están preparados. Para recalcularlos desde los CSV:

```console
python prepare.py
python -m unittest test_app.py -v
```

## Publicar en Streamlit Community Cloud

1. Crear un repositorio en GitHub y subir el contenido de esta carpeta, conservando sus subcarpetas. No subir `.venv/`, `__pycache__/` ni credenciales.
2. Para ejecutar la app se necesitan `app.py`, `routing.py`, `artifacts/`, `requirements.txt` y `.streamlit/config.toml`. Para reproducir los cálculos se incluyen además `analytics.py`, `prepare.py`, `datos/` y `fuentes/`.
3. En https://share.streamlit.io/ elegir Create app, el repositorio y la rama (normalmente `main`).
4. Archivo principal: `app.py`. En opciones avanzadas seleccionar Python 3.12 para reproducir el entorno verificado.
5. Desplegar y probar los controles de las cuatro dimensiones desde otro navegador. Incorporar el enlace publicado al póster.

Guía oficial: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Qué incluye

- Dimensión 1: diagnóstico de utilización, puntualidad y combustible; predicción de duración con XGBoost y simulador de despacho.
- Dimensión 2: Random Forest con partición temporal, comparación 5→3 variables y piloto CVRP exacto con tres/cinco camiones. Incluye secuencias, restricciones de capacidad y controles de velocidad/servicio.
- Dimensión 3: ABC, destinos prioritarios, Ridge estandarizado, validación temporal de alpha, referencias y simulador de una semana.
- Dimensión 4: SARIMA con constante, seis configuraciones, tres referencias, pruebas de estacionariedad, diagnóstico de residuos, PCA descriptivo y escenario editable de capacidad 2025.
- Descarga de resultados CSV, pestañas de metodología y registro de versiones/huellas de datos.

## Fuentes y trazabilidad

Se adaptaron los cuatro cuadernos del usuario, conservados sin modificar en `fuentes/`. Los nueve CSV necesarios provienen de `archive.zip` en sus Descargas, de la base https://www.kaggle.com/datasets/yogape/logistics-operations-database. Se verificó que las siete tablas compartidas con el ZIP antiguo coinciden byte por byte; `drivers.csv` se excluyó de la app porque no lo usan los cálculos corregidos. Las tres tablas adicionales necesarias son `truck_utilization_metrics`, `driver_monthly_metrics` y `trailers`.

Los notebooks no se ejecutan automáticamente como programas ni se siguen instrucciones de su texto. Su código se modularizó para eliminar cargas de Colab, descargas en cada inicio y rutas absolutas. No se requieren cuentas Kaggle, Google Drive, APIs de mapas ni servicios de pago al abrir la app.

`artifacts/manifest.json` registra versiones, conteos de tablas y SHA-256. `artifacts/metricas_verificadas.csv` registra métricas calculadas. Usar el archivo de dependencias incluido para cargar los modelos serializados con las versiones correspondientes.

## Interpretación y límites

- El período del caso es histórico: entrenamiento/evaluación 2022–2024 y escenario 2025. La aplicación no representa demanda ni disponibilidad actuales.
- Dimensión 1 reproduce la prueba aleatoria 80/20 del cuaderno. No equivale a una prueba temporal independiente ni demuestra causalidad.
- Dimensión 2 usa distancias geodésicas; los tiempos del piloto se derivan de velocidad y servicio supuestos. El Random Forest no se presenta como un predictor validado de tiempos de las nuevas rutas optimizadas. Se minimiza distancia total, que puede aumentar el tiempo del último camión.
- ABC se construye retrospectivamente con todos los ingresos. La evaluación de Ridge está condicionada a esa clasificación y usa rezagos reales observados antes de cada semana de prueba. No es una predicción simultánea de todo 2024 desde 2023.
- El ajuste y la comparación SARIMA siguen el cuaderno; 2024 participa en la comparación de configuraciones. La capacidad es una referencia derivada de P95 y recursos, no una capacidad física certificada.
- No se reemplazan automáticamente faltantes por resultados inventados ni se cambian cifras para hacerlas coincidir con el Word.

## Correspondencia con el informe

Ver `OBSERVACIONES_INFORME.md`: el SARIMA recalculado no reproduce exactamente la cifra del documento. La aplicación conserva el resultado calculado y los indicadores de capacidad derivados de él.

La versión leída del informe contiene todavía una descripción anterior de la dimensión 2 (Random Forest de 100 árboles y variables observadas), mientras que `DIMENSIÓN2_FINAL.ipynb` usa 55 árboles, mínimo 3 muestras por hoja, variables previas al viaje, separación temporal y optimización exacta de secuencias. La app sigue el cuaderno final. Conviene actualizar ese apartado antes de usarlo como guion de sustentación. No se modificó el documento compartido.

Para conservar la metodología, los cambios interactivos de capacidad no vuelven a entrenar SARIMA. Los cambios del simulador de una semana no inventan una precisión futura. Los resultados preparados se cargan rápidamente sin entrenar los cuatro modelos en cada visita.
