# Verificación de la aplicación

Fecha: 20 de noviembre de 2026. Entorno: Windows, Python 3.12 y versiones fijadas en `requirements.txt`.

## Resultados

- Cinco pruebas automatizadas de datos, métricas, particiones temporales y restricciones de rutas aprobadas (`python -m unittest test_app.py -v`).
- Trece comprobaciones de interfaz aprobadas, registradas en `VERIFICACION_UI.json`: las seis pantallas, predicciones XGBoost, Random Forest y Ridge, capacidad inviable/restaurada del ruteo, escenario de capacidad y navegación desde tarjetas.
- Portada y módulo de rutas revisados visualmente en el navegador a 1280 × 720. Servidor local accesible y controles renderizados.
- Se verificó la coincidencia de los CSV comunes con el proyecto antiguo mediante SHA-256. Las huellas y versiones están en `artifacts/manifest.json`.

## Alcance

Las pruebas comprueban reproducción y funcionamiento local. No constituyen validación de uso operativo en una empresa. La publicación en Streamlit Community Cloud queda pendiente del inicio de sesión del propietario y de la comprobación del despliegue.

Las diferencias detectadas entre el informe compartido y los cuadernos finales se describen en `OBSERVACIONES_INFORME.md`. Se conservaron las métricas calculadas, incluyendo MAPE SARIMA de aproximadamente 1.85 %, frente a 1.72 % del informe. No se modificó el documento compartido.
