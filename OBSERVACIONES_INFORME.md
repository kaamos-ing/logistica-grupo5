# Diferencias encontradas al construir la aplicación

El documento compartido se leyó como referencia y no se modificó. La app ejecuta los cuatro cuadernos corregidos y calcula las métricas desde los datos.

## SARIMA: cifra del informe no reproducida

Al ejecutar literalmente la preparación de la serie y la celda del SARIMA(1,0,0)(1,0,0)[12] con constante y `maxiter=200`, en el entorno verificado se obtiene aproximadamente MAE 44.68, RMSE 57.24 y MAPE 1.85%. El informe indica MAE 41.70, RMSE 54.42 y MAPE 1.72%.

Se verificó que la serie tiene 36 meses y 85.333 entregas y que las métricas de las referencias coinciden (por ejemplo Seasonal Naïve: RMSE 66.79). La diferencia no se corrigió copiando las cifras del documento. También se comprobó la misma celda por separado, con serie entera y flotante, y se fijó un entorno de bibliotecas científicas compatible. El ajuste reporta convergencia. No se determinó de forma concluyente la causa del resultado publicado en el informe.

Antes de sustentar, deben contrastar el resultado de su ejecución original de Colab con `artifacts/metricas_verificadas.csv`. Si no se conserva un ajuste que produzca las cifras del documento, actualizar las tablas y los indicadores derivados con los resultados recalculados. La app calcula dinámicamente la proyección y los déficits; no fija cinco meses, dos camiones o siete camiones como constantes del programa.

## Dimensión 2: descripción anterior en el documento

El apartado de algoritmos del informe describe 100 árboles, `min_samples_split=5` y variables como distancia real, detención e inactividad. El cuaderno `DIMENSIÓN2_FINAL.ipynb` usa 55 árboles, `min_samples_leaf=3`, cinco variables disponibles antes del viaje, validación temporal y selección de tres variables por permutación. Además distingue K-Means exploratorio del piloto de ruteo exacto. La aplicación sigue este cuaderno final.

## Otros resultados

Ridge reproduce alpha=100, MAE≈11.6005, RMSE≈14.2724 y MAPE≈3.8633%. El piloto reproduce 105.569 lb y distancias optimizadas de aproximadamente 7.235,556 mi (3 camiones) y 8.365,939 mi (5 camiones). Las diferencias pequeñas en los modelos de árboles pueden depender de las versiones; se conservan las métricas reales del entorno fijado.

Estas observaciones no implican que falten archivos: se encontraron todas las tablas necesarias en el archivo local `archive.zip`.
