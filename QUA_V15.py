import streamlit as st
import speech_recognition as sr
import pandas as pd
from hubspot import HubSpot
from hubspot.crm.objects import ApiException, PublicObjectSearchRequest
import datetime
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import matplotlib.pyplot as plt
from PIL import Image
import base64
from fpdf import FPDF

# Configuración de la página de Streamlit
st.set_page_config(page_title="Análisis de Llamadas", layout="wide", page_icon="📞")

# Estilos CSS personalizados para márgenes y bordes
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
        padding: 20px;
        border: 2px solid #1f77b4;  /* Borde azul oscuro */
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #1f77b4;  /* Azul oscuro */
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #165a8a;  /* Azul más oscuro */
    }
    .stHeader {
        color: #1f77b4;  /* Azul oscuro */
    }
    .stDataFrame {
        background-color: white;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .stPlot {
        margin-top: 20px;
    }
    .status-bar {
        background-color: #1f77b4;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo de la aplicación
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_of_image("cvcc.png")  # Cambia "logo.png" por la ruta de tu logo
st.markdown(
    f'<img src="data:image/png;base64,{logo_base64}" width="150">',
    unsafe_allow_html=True,
)

# Función para descargar una llamada
def download_call_audio(call_id, recording_url, access_token):
    try:
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(recording_url, headers=headers)
        response.raise_for_status()
        audio_file_path = f"{call_id}.wav"
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(response.content)
        st.success(f"Grabación de la llamada {call_id} descargada correctamente.")
        return audio_file_path
    except requests.exceptions.HTTPError as e:
        st.error(f"Error al descargar la grabación: {e}")
        return None

# Función para transcribir una llamada
def transcribe_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language="es-ES")
        st.write("Transcripción completa:")
        st.write(text)
        return text
    except sr.UnknownValueError:
        st.warning("Google Speech Recognition no pudo entender el audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Error al solicitar resultados de Google Speech Recognition: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado durante la transcripción: {e}")
        return None

# Función para analizar una transcripción
def analyze_transcription(transcription):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    messages = [
        {
            "role": "system",
            "content": """
Eres un experto en feedback y ventas por teléfono. Tu objetivo es analizar una conversación que te voy a proporcionar y evaluar su cumplimiento con los PASOS OBLIGATORIOS (✅). Sigue las siguientes pautas para dar feedback:

###
PASOS OBLIGATORIOS (✅)

1. **Apertura**:
   - Saludo casual ("¡Hola!")
   - Usar solo el nombre del lead (sin Sr./Sra./Don/Doña)
   - Presentarte solo con tu nombre (sin apellido)
   - Mencionar que llamas del Taller de Bienes Raíces con Carlos Devis

2. **Romper el hielo**:
   - Elegir UN solo tema: ciudad, clima, gastronomía o lugares turísticos
   - Hacer preguntas sobre el tema elegido

3. **Identificación del dolor/necesidad**:
   - Preguntar motivación sobre bienes raíces
   - Identificar obstáculos
   - Si no es claro, profundizar con preguntas sobre:
     - Ahorros
     - Situación financiera
     - Fuente de ingresos
     - Situación personal
   - Confirmar el dolor identificado con el lead

4. **Presentación de credenciales**:
   - Mencionar los 700+ testimonios de éxito
   - Compartir un ejemplo relevante al caso del lead
   - Preguntar si quisieran lograr resultados similares

5. **Presentación de la metodología**:
   - Explicar los 5 pasos:
     - Cambio de pensamiento
     - Organización financiera
     - Ahorrar
     - Invertir
     - Repetir el proceso

6. **Verificar dudas**:
   - Preguntar si hay dudas o preguntas

7. **Presentación de programas**:
   - Mencionar las dos opciones principales:
     - Programa Avanzado ($1,497 USD)
     - Programa Mentoría ($4,999 USD)

8. **Cierre (Obligatorio)**:
   - Mencionar SIEMPRE el precio de página
   - Ofrecer SIEMPRE precio promocional
   - Dar máximo 48 horas de plazo como último recurso

###

Para cada paso, indica si se cumplió (✅) o no (❌) y proporciona una explicación breve. Al final, da una calificación única de 0 a 5 siendo 5 una llamada perfecta. Además, proporciona sugerencias de mejora para cada paso no cumplido.
"""
        },
        {
            "role": "user",
            "content": transcription
        }
    ]
    ai_msg = llm.invoke(messages)
    return ai_msg.content

# Función para generar el gráfico de barras con totalización
def generar_grafico_barras_con_totalizacion(df_resultados):
    # Clasificar las llamadas según su nota
    llamadas_rojas = df_resultados[(df_resultados['Rating'] >= 0) & (df_resultados['Rating'] <= 2)].shape[0]
    llamadas_amarillas = df_resultados[(df_resultados['Rating'] > 2) & (df_resultados['Rating'] <= 4)].shape[0]
    llamadas_verdes = df_resultados[(df_resultados['Rating'] >= 4.8) & (df_resultados['Rating'] <= 5)].shape[0]

    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.bar(['1', '2', '3', '4', '5'], [llamadas_rojas, llamadas_amarillas, llamadas_verdes, 0, 0], 
                    color=['red', 'yellow', 'green', 'blue', 'blue'])

    # Añadir el total de llamadas al final de cada barra
    for barra in barras:
        height = barra.get_height()
        ax.text(barra.get_x() + barra.get_width() / 2., height,
                f'{int(height)}', ha='center', va='bottom')

    # Configuraciones adicionales del gráfico
    ax.set_xlabel('Rango de Notas')
    ax.set_ylabel('Cantidad de Llamadas')
    ax.set_title('Distribución de Llamadas por Rango de Notas')
    ax.grid(True)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

    # Calcular y mostrar el promedio total de las notas
    promedio_total = df_resultados['Rating'].mean()
    st.write(f"Promedio Total de Notas: {promedio_total:.2f}")

# Función para generar el informe en PDF
def generar_informe_pdf(resultados, promedio_calificaciones):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Título del informe
    pdf.cell(200, 10, txt="Informe de Análisis de Llamadas", ln=True, align="C")
    pdf.ln(10)

    # Promedio de calificaciones
    pdf.cell(200, 10, txt=f"Promedio de Calificaciones: {promedio_calificaciones:.2f}/5.0", ln=True)
    pdf.ln(10)

    # Detalles de cada llamada
    for resultado in resultados:
        pdf.cell(200, 10, txt=f"Llamada ID: {resultado['Call ID']}", ln=True)
        pdf.multi_cell(0, 10, txt=resultado["Analysis"].encode("latin1", "replace").decode("latin1"))
        pdf.ln(5)

    # Guardar el PDF
    pdf_path = "informe_llamadas.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Interfaz de usuario en Streamlit
def main():
    st.sidebar.title("Configuración")

    # Claves de API automáticas (puedes cambiarlas por tus claves reales)
    hubspot_token = "pat-na1-bbfa9305-63e1-4178-b013-1fe6a09fbfbe"  # Clave de HubSpot
    google_api_key = "AIzaSyCz29DEKgE0TzX8SHa6adNRGrw7FBAavNo"  # Clave de Google

    os.environ["HUBSPOT_ACCESS_TOKEN"] = hubspot_token
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Selección de fechas con un calendario
    st.sidebar.write("Selecciona el rango de fechas:")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fecha_desde = st.date_input("Fecha de inicio:")
    with col2:
        fecha_hasta = st.date_input("Fecha de fin:")

    if fecha_desde and fecha_hasta:
        fecha_desde_dt = datetime.datetime.combine(fecha_desde, datetime.time())
        fecha_hasta_dt = datetime.datetime.combine(fecha_hasta, datetime.time())
        fecha_desde_timestamp = int(fecha_desde_dt.timestamp() * 1000)
        fecha_hasta_timestamp = int(fecha_hasta_dt.timestamp() * 1000)

        search_request = PublicObjectSearchRequest(
            filter_groups=[{
                "filters": [
                    {"propertyName": "hs_createdate", "operator": "GTE", "value": str(fecha_desde_timestamp)},
                    {"propertyName": "hs_createdate", "operator": "LTE", "value": str(fecha_hasta_timestamp)}
                ]
            }],
            properties=["hs_call_recording_url", "hs_createdate"]
        )

        try:
            client = HubSpot(access_token=hubspot_token)
            api_response = client.crm.objects.search_api.do_search("calls", search_request)
            results = api_response.results

            # Mostrar resultados
            call_ids = [result.id for result in results]
            recording_urls = [result.properties.get("hs_call_recording_url") for result in results]
            valid_calls = [{"Call ID": call_id, "Recording URL": url} for call_id, url in zip(call_ids, recording_urls) if url]
            df = pd.DataFrame(valid_calls)

            if not df.empty:
                st.write("Llamadas encontradas:")
                st.dataframe(df)

                # Seleccionar llamadas para analizar
                selected_call_ids = st.multiselect("Selecciona las llamadas para analizar:", df["Call ID"].tolist())

                if selected_call_ids:
                    resultados = []
                    for call_id in selected_call_ids:
                        recording_url = df.loc[df["Call ID"] == call_id, "Recording URL"].values[0]
                        audio_file_path = download_call_audio(call_id, recording_url, hubspot_token)

                        if audio_file_path:
                            transcription = transcribe_audio_to_text(audio_file_path)
                            if transcription:
                                analisis = analyze_transcription(transcription)
                                resultados.append({"Call ID": call_id, "Transcription": transcription, "Analysis": analisis})

                    if resultados:
                        st.write("Resultados del análisis:")
                        for resultado in resultados:
                            st.write(f"Llamada ID: {resultado['Call ID']}")
                            st.write("Análisis:")
                            st.write(resultado["Analysis"])
                            st.write("-" * 80)

                        # Calcular el promedio de calificaciones
                        def extract_rating(analysis):
                            lines = analysis.split('\n')
                            for line in lines:
                                if "calificación única de" in line:
                                    return float(line.split()[-1])
                            return 0

                        df_resultados = pd.DataFrame(resultados)
                        df_resultados['Rating'] = df_resultados['Analysis'].apply(extract_rating)

                        # Mostrar el gráfico de barras con totalización
                        generar_grafico_barras_con_totalizacion(df_resultados)

                        # Generar y descargar el informe en PDF
                        informe_pdf = generar_informe_pdf(resultados, df_resultados['Rating'].mean())
                        with open(informe_pdf, "rb") as file:
                            st.download_button(
                                label="Descargar Informe en PDF",
                                data=file,
                                file_name="informe_llamadas.pdf",
                                mime="application/pdf"
                            )

                        # Botones finales
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Nueva Búsqueda"):
                                st.experimental_rerun()
                        with col2:
                            if st.button("Cerr