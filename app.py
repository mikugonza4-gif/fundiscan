"""FundiScan: Streamlit + EfficientNet-B0 + Gemini. Ejecutar: streamlit run app.py."""
import hashlib
import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import timm
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

BASE = Path(__file__).resolve().parent
CONFIANZA_MINIMA = 0.85
st.set_page_config(page_title="FundiScan · Inspección de fundición", page_icon="⚙️", layout="wide")

try:
    CONFIG = json.loads((BASE / "modelo/config.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    st.error("Falta modelo/config.json o su contenido no es válido. Revisa los archivos del repositorio.")
    st.stop()
CLASES = CONFIG["clases"]
POS = CONFIG["clase_positiva"]
I_POS = CLASES.index(POS)
UMBRAL = float(CONFIG["umbral"])
METRICAS = CONFIG["metricas_prueba"]


@st.cache_resource(show_spinner="Cargando el modelo de inspección…")
def cargar_modelo():
    """Una sola copia en CPU, compartida entre sesiones; solo inferencia."""
    torch.set_num_threads(2)
    modelo = timm.create_model(CONFIG["arquitectura"], pretrained=False, num_classes=len(CLASES))
    pesos = torch.load(BASE / "modelo/efficientnet_b0_fundicion.pt", map_location="cpu", weights_only=True)
    modelo.load_state_dict(pesos)
    modelo.eval()
    preparar = transforms.Compose([
        transforms.Resize((CONFIG["tam"], CONFIG["tam"])),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["media"], CONFIG["desv"]),
    ])
    return modelo, preparar


def clasificar(imagen):
    modelo, preparar = cargar_modelo()
    with torch.inference_mode():
        probs = torch.softmax(modelo(preparar(imagen.convert("RGB")).unsqueeze(0)), dim=1)[0].tolist()
    p_def = probs[I_POS]
    clase = POS if p_def >= UMBRAL else CLASES[1 - I_POS]
    confianza = probs[CLASES.index(clase)]
    return {"clase": clase, "confianza": confianza, "p_defectuosa": p_def,
            "baja_confianza": confianza < CONFIANZA_MINIMA,
            "probabilidades": dict(zip(CLASES, probs))}


def secreto(nombre, defecto=""):
    if os.environ.get(nombre):
        return os.environ[nombre]
    try:
        return st.secrets.get(nombre, defecto)
    except FileNotFoundError:
        return defecto


def respaldo(e):
    return (
        f"**Resultado:** clasificación automática {e['clase']}; no confirma un defecto específico.\n\n"
        "**Acción inmediata:** solicitar revisión de un inspector antes de liberar o descartar la pieza. "
        "Si el modelo indica defecto o baja confianza, retenerla para reinspección.\n\n"
        "**Verificación:** revisar iluminación, enfoque y vista frontal; contrastar con el procedimiento de calidad. "
        "No es posible determinar causas físicas con estos datos.\n\n"
        "**Registro:** anotar resultado, lote, turno y decisión del inspector.\n\n"
        "_Prototipo educativo: la decisión final corresponde al responsable de calidad._"
    )


def recomendar(e):
    clave = secreto("GEMINI_API_KEY")
    if not clave:
        return "Respaldo fijo · falta configurar Gemini", respaldo(e)
    try:
        from google import genai
        from google.genai import types
        instrucciones = (
            "Eres FundiScan, asistente educativo de control de calidad de impulsores de fundición. "
            "Solo recibes texto del clasificador, no ves la imagen. No inventes medidas, ubicación ni tipo de defecto. "
            "Habla únicamente de inspección y calidad. Las causas, si las mencionas, son hipótesis por verificar. "
            "No autorices liberación, descarte o refusión automáticamente. "
            "Si baja_confianza es true, comienza exigiendo revisión de un inspector. "
            "La confianza es un puntaje del modelo, no certeza calibrada. Escribe en español, máximo 170 palabras, "
            "con secciones Resultado, Acción inmediata, Verificación y Registro. "
            "Finaliza recordando el uso educativo y la decisión humana."
        )
        nombre = secreto("GEMINI_MODEL", "gemini-flash-latest")
        with genai.Client(api_key=clave, http_options=types.HttpOptions(timeout=30000)) as cliente:
            respuesta = cliente.models.generate_content(
                model=nombre,
                contents=json.dumps({"resultado": e, "umbral": UMBRAL,
                                     "metricas_prueba": METRICAS}, ensure_ascii=False),
                config=types.GenerateContentConfig(system_instruction=instrucciones,
                                                   temperature=0.3, max_output_tokens=4096),
            )
        if not respuesta.text:
            raise ValueError("Respuesta vacía")
        return f"Generado por Gemini · {nombre}", respuesta.text.strip()
    except Exception as err:
        # No mostrar credenciales ni detalles internos de la API a los visitantes.
        print("Gemini: consulta fallida", type(err).__name__)
        return "Respaldo fijo · Gemini no respondió; revisa clave, modelo o cuota", respaldo(e)


st.markdown("""<style>
.fundi-head {padding:1.5rem;border-radius:16px;background:linear-gradient(115deg,#1e293b,#334155,#9a3412);color:white;margin-bottom:1rem}
.fundi-head h1 {color:white;margin:0;padding:0}.fundi-head p {color:#e2e8f0;margin:.5rem 0 0}
</style><div class="fundi-head"><div style="display:flex;align-items:center;gap:16px"><svg width="64" height="64" viewBox="0 0 64 64" fill="none" aria-hidden="true">
  <circle cx="32" cy="32" r="27" stroke="#fb923c" stroke-width="4"/>
  <circle cx="32" cy="32" r="8" fill="#fb923c"/>
  <path d="M32 24 C40 14 50 16 54 22 M40 32 C50 40 48 50 42 54 M32 40 C24 50 14 48 10 42 M24 32 C14 24 16 14 22 10"
        stroke="#f8fafc" stroke-width="3.5" stroke-linecap="round"/>
</svg><h1>FundiScan</h1></div><p>Inspección visual de impulsores de fundición con inteligencia artificial</p></div>""", unsafe_allow_html=True)
st.caption("Certamen I · IA Generativa para la Gestión Industrial · UDD")
inspeccion, desempeno, acerca = st.tabs(["🔍 Inspección", "📊 Desempeño", "ℹ️ Acerca"])

with inspeccion:
    izquierda, derecha = st.columns([1, 1])
    imagen = None
    datos = None
    with izquierda:
        fuente = st.radio("Elige una imagen", ["Subir foto", "Cámara", "Ejemplos"], horizontal=True)
        if fuente == "Subir foto":
            archivo = st.file_uploader("Foto frontal de la pieza", type=["jpg", "jpeg", "png"])
            if archivo is not None:
                datos = archivo.getvalue()
        elif fuente == "Cámara":
            archivo = st.camera_input("Fotografía la pieza")
            if archivo is not None:
                datos = archivo.getvalue()
        else:
            ejemplos = sorted(p for p in (BASE / "ejemplos").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if ejemplos:
                seleccionado = st.selectbox("Imágenes de prueba", ejemplos, format_func=lambda p: p.name)
                datos = seleccionado.read_bytes()
            else:
                st.info("Agrega las imágenes a la carpeta ejemplos/.")
        if datos:
            try:
                with Image.open(io.BytesIO(datos)) as original:
                    if original.width * original.height > 20_000_000:
                        raise ValueError("La imagen supera 20 megapíxeles. Reduce su tamaño.")
                    imagen = ImageOps.exif_transpose(original).convert("RGB")
                st.image(imagen, caption="Imagen que se analizará", width=320)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as err:
                st.error(f"No se pudo abrir la imagen: {err}")
                datos = None
        # Cambiar o retirar la imagen invalida también la recomendación anterior.
        firma = (fuente, hashlib.sha256(datos).hexdigest() if datos else None)
        if st.session_state.get("firma") != firma:
            st.session_state["firma"] = firma
            st.session_state["resultado"] = None
            st.session_state["recomendacion"] = None
        if st.button("🔍 Inspeccionar pieza", type="primary", disabled=imagen is None):
            st.session_state["resultado"] = None
            st.session_state["recomendacion"] = None
            try:
                with st.spinner("Analizando la pieza…"):
                    st.session_state["resultado"] = clasificar(imagen)
            except Exception as err:
                st.error("No se pudo cargar o ejecutar el modelo. Revisa modelo/, sus pesos y las dependencias.")
                print("Error de inferencia:", type(err).__name__, str(err))

    with derecha:
        e = st.session_state.get("resultado")
        if e:
            titulo = f"Clasificación del modelo: {e['clase'].upper()}"
            if e["baja_confianza"]:
                st.warning(titulo + " · Requiere revisión humana por baja confianza.")
            elif e["clase"] == POS:
                st.error(titulo + " · Retener para reinspección.")
            else:
                st.success(titulo + " · Confirmar según el procedimiento de calidad.")
            st.metric("Confianza en la clase decidida", f"{e['confianza']:.2%}")
            for clase, prob in e["probabilidades"].items():
                st.progress(float(prob), text=f"{clase.capitalize()}: {prob:.2%}")
            st.caption(f"Se decide defectuosa si P(defectuosa) ≥ {UMBRAL:.0%}. "
                       "Por eso puede indicarse defectuosa aunque conforme tenga mayor probabilidad.")
            st.caption("La confianza es un puntaje del modelo, no una garantía. "
                       f"La accuracy global en prueba es {METRICAS['accuracy']:.2%}; no cambia con cada foto.")
        else:
            st.info("Selecciona una imagen y presiona Inspeccionar pieza.")
        if st.button("💬 Generar recomendación con Gemini", disabled=not bool(e)):
            with st.spinner("Consultando a Gemini…"):
                st.session_state["recomendacion"] = recomendar(e)
        rec = st.session_state.get("recomendacion")
        if rec:
            st.caption(rec[0])
            st.markdown(rec[1])
    st.warning("Prototipo educativo. No reemplaza la inspección de calidad. Bajo 85 % de confianza se requiere "
               "revisión humana; una confianza alta tampoco descarta errores. Fotos de otras piezas, "
               "fondos o condiciones de iluminación pueden producir resultados incorrectos.")

with desempeno:
    st.subheader(f"Evaluación final · {CONFIG['n_prueba']} imágenes de prueba")
    columnas = st.columns(4)
    for col, etiqueta, clave in zip(columnas,
            ["Accuracy", "Recall defectuosas", "Precisión defectuosas", "Accuracy balanceada"],
            ["accuracy", "recall_defectuosa", "precision_defectuosa", "accuracy_balanceada"]):
        col.metric(etiqueta, f"{METRICAS[clave]:.2%}")
    st.write("Matriz de confusión: filas = clase real; columnas = predicción.")
    st.dataframe(pd.DataFrame(CONFIG["matriz_confusion"],
                 index=[f"Real {c}" for c in CLASES], columns=[f"Predice {c}" for c in CLASES]))
    st.write(f"**{METRICAS['defectuosas_no_detectadas']} defectuosas no detectadas** y "
             f"**{METRICAS['falsas_alarmas']} falsas alarmas**. El error prioritario es dejar pasar una pieza defectuosa.")
    st.write(f"Umbral aplicado: **{UMBRAL:.2f}**, elegido en validación con costo relativo "
             f"FN = {CONFIG['costos']['falso_negativo']} y FP = {CONFIG['costos']['falso_positivo']}. "
             "Son supuestos de costo del ejercicio, no costos monetarios medidos.")
    st.subheader("Comparación de enfoques en prueba · umbral 0,50")
    filas = [{"Modelo": nombre, "Accuracy": f"{m['accuracy']:.2%}",
              "Acc. balanceada": f"{m['accuracy_balanceada']:.2%}",
              "Recall defectuosas": f"{m['recall_defectuosa']:.2%}",
              "Falsos negativos": m['defectuosas_no_detectadas'],
              "Falsos positivos": m['falsas_alarmas']}
             for nombre, m in CONFIG['comparacion'].items()]
    st.dataframe(pd.DataFrame(filas), hide_index=True)
    st.info("YOLO obtuvo mejores métricas en esta comparación. Esta app integra EfficientNet-B0. "
            "La tabla usa umbral 0,50; las métricas finales de arriba usan 0,10. "
            "Si la comparación en prueba influyó en la selección del modelo, debe declararse como limitación metodológica.")

with acerca:
    st.subheader("De la foto a la recomendación")
    st.write("EfficientNet-B0 clasifica la pieza. Gemini recibe únicamente el resultado numérico en texto "
             "y propone acciones para el inspector; no recibe la imagen ni identifica tipos de defecto.")
    st.markdown(f"**Datos:** [Casting product image data for quality inspection]({CONFIG['dataset']}), "
                "R. Dabhi / Pilot Technocast. Imágenes de piezas en condiciones controladas.")
    st.dataframe(pd.DataFrame(CONFIG["conteo"]).T)
    st.write("Limitaciones: no se ha validado su uso en otra planta, otras piezas o fotos de campo. "
             "El ejemplo llamado dudosa es una imagen de demostración, no una tercera clase.")
    st.caption("Desarrollo asistido por IA; corresponde al equipo revisar y explicar el código y los resultados.")
