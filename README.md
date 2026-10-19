# FundiScan

Prototipo educativo de inspección visual de impulsores de fundición: clasificación conforme/defectuosa con EfficientNet-B0 y recomendaciones de control de calidad con Gemini.

Certamen I · IA Generativa para la Gestión Industrial · UDD.

**Integrantes:** completar con los tres nombres antes de entregar.

**App pública:** agregar el enlace de Streamlit una vez publicado.

## Ejecución y publicación

Consulta [GUIA_STREAMLIT.md](GUIA_STREAMLIT.md). La interfaz usa Streamlit y el modelo se ejecuta en CPU. No necesita volver a entrenarse para publicar.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Configura GEMINI_API_KEY en los secretos de Streamlit. GEMINI_MODEL es opcional y usa gemini-flash-latest por defecto. Nunca subas la clave al repositorio. Sin acceso a Gemini se muestra un respaldo fijo, identificado como tal; este respaldo no demuestra una llamada real a IA generativa.

## Archivos de la app

- app.py: interfaz, inferencia, métricas y recomendaciones.
- requirements.txt: dependencias para CPU Linux.
- modelo/efficientnet_b0_fundicion.pt: pesos entrenados.
- modelo/config.json: clases, transformaciones, umbral y métricas reales.
- ejemplos/: cinco imágenes para la demostración.
- .streamlit/config.toml: apariencia y tamaño máximo de carga.

Conservar también notebooks/ y docs/ del trabajo original: este paquete de publicación no reemplaza el notebook ejecutado ni los documentos de la entrega.

## Datos

[Casting product image data for quality inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product), R. Dabhi / Pilot Technocast (India).

| División | Conformes | Defectuosas | Total |
|---|---:|---:|---:|
| Entrenamiento | 2443 | 3194 | 5637 |
| Validación | 432 | 564 | 996 |
| Prueba | 262 | 453 | 715 |

Total: 7348 imágenes. **Licencia del dataset: pendiente de transcribir y verificar desde Kaggle en la ficha de datos.** La licencia del código y la del dataset son asuntos distintos.

## Resultados del modelo integrado

EfficientNet-B0, umbral de defecto 0,10, elegido en validación según costo relativo FN=20 y FP=1. Estos costos son supuestos del ejercicio.

- Accuracy: 98,88 %.
- Recall de defectuosas: 99,34 %.
- Precisión de defectuosas: 98,90 %.
- Accuracy balanceada: 98,71 %.
- Matriz de confusión: [[257, 5], [3, 450]], filas y columnas en orden conforme, defectuosa.
- Tres defectuosas no detectadas y cinco falsas alarmas en 715 imágenes.

Comparación del notebook en las mismas imágenes de prueba con umbral 0,50:

| Enfoque | Accuracy | Recall defectuosas | FN | FP |
|---|---:|---:|---:|---:|
| CLIP sin entrenar | 52,45 % | 25,17 % | 339 | 1 |
| EfficientNet-B0 | 99,44 % | 99,12 % | 4 | 0 |
| YOLO11n-cls | 99,72 % | 99,56 % | 2 | 0 |

YOLO obtuvo los mejores resultados de esta comparación. La aplicación integra EfficientNet-B0; no se afirma que sea el mejor modelo. La tabla usa un umbral distinto al resultado final de la aplicación. Si el conjunto de prueba influyó en la selección del modelo, debe reconocerse como limitación y reservarse una evaluación independiente para trabajos posteriores.

## Límites y uso responsable

Entrenamiento con imágenes controladas de una planta. No se ha validado el comportamiento ante otras piezas, plantas o fotos reales de campo. Las probabilidades son puntajes del modelo, no garantías calibradas. Bajo 85 % de confianza se solicita revisión humana; una confianza alta también puede acompañar un error. El modelo no localiza ni identifica el tipo de defecto. Gemini solo recibe texto y no ve la imagen.

No usar este prototipo para liberar o descartar piezas automáticamente. La decisión corresponde al responsable de calidad.

## Uso de IA

El desarrollo y la adaptación de la aplicación contaron con asistencia de IA. El equipo debe revisar, comprender y explicar el código, las decisiones experimentales y los resultados. Las recomendaciones generadas por Gemini se distinguen de los textos fijos de respaldo.
