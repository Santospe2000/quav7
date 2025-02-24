# Importar bibliotecas
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

# Configuración de la página de Streamlit
st.set_page_config(page_title="Análisis de Llamadas", layout="wide")
st.title("Análisis de Llamadas con IA")

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

# Función para generar el gráfico semaforizado
def generar_grafico_semaforizado(promedio_calificaciones):
    if promedio_calificaciones >= 4:
        color_semaforo = "green"
        estado_semaforo = "Verde (Excelente)"
    elif 2.5 <= promedio_calificaciones < 4:
        color_semaforo = "yellow"
        estado_semaforo = "Amarillo (Regular)"
    else:
        color_semaforo = "red"
        estado_semaforo = "Rojo (Deficiente)"

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh(0, promedio_calificaciones, color=color_semaforo)
    ax.set_xlim(0, 5)
    ax.set_title("Resultado General del Análisis (Semaforizado)")
    ax.set_xlabel("Calificación Promedio")
    ax.set_yticks([])
    st.pyplot(fig)
    st.write(f"Estado del Semáforo: {estado_semaforo}")

# Interfaz de usuario en Streamlit
def main():
    # Solicitar tokens de acceso
    hubspot_token = st.text_input("Introduce tu clave de API de HubSpot:", type="password")
    google_api_key = st.text_input("Introduce tu clave de API de Google:", type="password")

    if hubspot_token and google_api_key:
        os.environ["HUBSPOT_ACCESS_TOKEN"] = hubspot_token
        os.environ["GOOGLE_API_KEY"] = google_api_key

        # Solicitar el rango de fechas
        fecha_desde = st.text_input("Introduce la fecha de inicio (YYYY-MM-DD):")
        fecha_hasta = st.text_input("Introduce la fecha de fin (YYYY-MM-DD):")

        if fecha_desde and fecha_hasta:
            # Convertir fechas y realizar búsqueda
            fecha_desde_dt = datetime.datetime.strptime(fecha_desde, "%Y-%m-%d")
            fecha_hasta_dt = datetime.datetime.strptime(fecha_hasta, "%Y-%m-%d")
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
                            promedio_calificaciones = df_resultados['Rating'].mean()

                            # Mostrar el promedio y el gráfico semaforizado
                            st.write(f"Promedio de Calificaciones: {promedio_calificaciones:.2f}/5.0")
                            generar_grafico_semaforizado(promedio_calificaciones)

                else:
                    st.warning("No se encontraron llamadas con grabaciones válidas.")
            except ApiException as e:
                st.error(f"Error en la búsqueda de llamadas: {e}")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()