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
import time
from colorama import Fore, Style, init  # Para colores en la consola

# Inicializar colorama
init(autoreset=True)

# Configurar la clave de HubSpot de manera segura
if "HUBSPOT_ACCESS_TOKEN" not in os.environ:
    hubspot_token = st.text_input("Introduce tu clave de API de HubSpot:", type="password")
    os.environ["HUBSPOT_ACCESS_TOKEN"] = hubspot_token

# Configurar la clave de Google de manera segura
if "GOOGLE_API_KEY" not in os.environ:
    google_api_key = st.text_input("Introduce tu clave de API de Google:", type="password")
    os.environ["GOOGLE_API_KEY"] = google_api_key

# Inicializar el cliente de HubSpot
client = HubSpot(access_token=os.environ["HUBSPOT_ACCESS_TOKEN"])

# Función para buscar llamadas en un rango de fechas
def fetch_all_calls(fecha_desde_timestamp, fecha_hasta_timestamp):
    all_results = []
    after = None
    while True:
        search_request = PublicObjectSearchRequest(
            filter_groups=[{
                "filters": [
                    {
                        "propertyName": "hs_createdate",
                        "operator": "GTE",
                        "value": str(fecha_desde_timestamp)
                    },
                    {
                        "propertyName": "hs_createdate",
                        "operator": "LTE",
                        "value": str(fecha_hasta_timestamp)
                    }
                ]
            }],
            properties=["hs_call_recording_url", "hs_createdate"],
            limit=100,  # Límite de resultados por página
            after=after  # Paginación
        )
        try:
            api_response = client.crm.objects.search_api.do_search("calls", search_request)
            results = api_response.results
            all_results.extend(results)
            if api_response.paging and api_response.paging.next and api_response.paging.next.after:
                after = api_response.paging.next.after
            else:
                break
        except ApiException as e:
            st.error(f"Exception when calling search_api->do_search: {e}")
            break
    return all_results

# Función para descargar una llamada
def download_call_audio(call_id, recording_url):
    try:
        headers = {
            "Authorization": f"Bearer {os.environ['HUBSPOT_ACCESS_TOKEN']}"
        }
        response = requests.get(recording_url, headers=headers)
        response.raise_for_status()  # Esto lanzará una excepción si la solicitud no fue exitosa
        audio_file_path = f"{call_id}.wav"
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(response.content)
        st.success(f"Grabación de la llamada {call_id} descargada correctamente en formato .wav.")
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
        st.success("Transcripción completa:")
        st.write(text)
        return text
    except sr.UnknownValueError:
        st.error(f"Google Speech Recognition no pudo entender el audio de la llamada {call_id}.")
        return None
    except sr.RequestError as e:
        st.error(f"Error al solicitar resultados de Google Speech Recognition: {e}")
        return None
    except Exception as e:
        st.error(f"Error inesperado durante la transcripción: {e}")
        return None

# Función para analizar una transcripción con reintentos
def analyze_transcription_with_retries(transcription, max_retries=5, initial_delay=2):
    retries = 0
    while retries < max_retries:
        try:
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

Para cada paso, indica si se cumplió (✅) o no (❌) y proporciona una explicación breve. Al final, da una calificación única de 0 a 5 siendo 5 una llamada perfecta.
"""
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ]

            # Invocar el modelo de Google Vertex AI
            ai_msg = llm.invoke(messages)
            return ai_msg.content
        except Exception as e:
            st.error(f"Error en el análisis de la transcripción: {e}. Reintentando en {initial_delay} segundos...")
            time.sleep(initial_delay)
            retries += 1
            initial_delay *= 2  # Retroceso exponencial
    st.error(f"No se pudo analizar la transcripción después de {max_retries} intentos.")
    return None

# Función para calcular el porcentaje de cumplimiento
def calculate_compliance_percentage(analisis_pasos):
    pasos_cumplidos = analisis_pasos.count("✅")
    total_pasos = 8  # Total de pasos obligatorios
    return (pasos_cumplidos / total_pasos) * 100

# Función para generar el semáforo y el análisis DOFA
def generate_traffic_light_and_dofa(porcentaje_cumplimiento):
    if porcentaje_cumplimiento == 100:
        semaforo = f"{Fore.GREEN}Verde{Style.RESET_ALL}"
        dofa = "Excelente cumplimiento de todos los pasos. Mantener el enfoque y continuar con las mejores prácticas."
    elif porcentaje_cumplimiento >= 80:
        semaforo = f"{Fore.YELLOW}Amarillo{Style.RESET_ALL}"
        dofa = "Buen cumplimiento, pero hay áreas de mejora. Revisar los pasos no cumplidos y reforzar la capacitación."
    elif porcentaje_cumplimiento >= 60:
        semaforo = f"{Fore.LIGHTYELLOW_EX}Naranja{Style.RESET_ALL}"
        dofa = "Cumplimiento moderado. Es necesario revisar y mejorar varios pasos clave para aumentar la efectividad."
    else:
        semaforo = f"{Fore.RED}Rojo{Style.RESET_ALL}"
        dofa = "Cumplimiento bajo. Se requiere una revisión completa del proceso y una capacitación intensiva."
    return semaforo, dofa

# Función para generar el reporte general
def generate_general_report(resultados_finales):
    if not resultados_finales:
        return None

    # Calcular el promedio de cumplimiento
    porcentajes_cumplimiento = [resultado["Porcentaje Cumplimiento"] for resultado in resultados_finales]
    promedio_cumplimiento = sum(porcentajes_cumplimiento) / len(porcentajes_cumplimiento)

    # Calcular la distribución del semáforo
    distribucion_semaforo = {
        "Verde": 0,
        "Amarillo": 0,
        "Naranja": 0,
        "Rojo": 0
    }
    for resultado in resultados_finales:
        semaforo = resultado["Semáforo"]
        if "Verde" in semaforo:
            distribucion_semaforo["Verde"] += 1
        elif "Amarillo" in semaforo:
            distribucion_semaforo["Amarillo"] += 1
        elif "Naranja" in semaforo:
            distribucion_semaforo["Naranja"] += 1
        elif "Rojo" in semaforo:
            distribucion_semaforo["Rojo"] += 1

    # Generar el análisis DOFA general
    if promedio_cumplimiento == 100:
        dofa_general = "Excelente cumplimiento en todas las llamadas. Mantener el enfoque y continuar con las mejores prácticas."
    elif promedio_cumplimiento >= 80:
        dofa_general = "Buen cumplimiento general, pero hay áreas de mejora. Revisar los pasos no cumplidos y reforzar la capacitación."
    elif promedio_cumplimiento >= 60:
        dofa_general = "Cumplimiento moderado. Es necesario revisar y mejorar varios pasos clave para aumentar la efectividad."
    else:
        dofa_general = "Cumplimiento bajo. Se requiere una revisión completa del proceso y una capacitación intensiva."

    # Crear el reporte general
    reporte_general = {
        "Promedio de Cumplimiento": promedio_cumplimiento,
        "Distribución del Semáforo": distribucion_semaforo,
        "Análisis DOFA General": dofa_general
    }
    return reporte_general

# Función para mostrar el reporte general
def display_general_report(reporte_general):
    if not reporte_general:
        st.warning("No hay datos para generar un reporte general.")
        return

    st.write("\n" + "=" * 80)
    st.write("REPORTE GENERAL DE LLAMADAS EVALUADAS")
    st.write("=" * 80)

    # Mostrar el promedio de cumplimiento
    st.write(f"\n{Fore.CYAN}Promedio de Cumplimiento:{Style.RESET_ALL}")
    st.write(f"{reporte_general['Promedio de Cumplimiento']:.2f}%")

    # Mostrar la distribución del semáforo
    st.write(f"\n{Fore.CYAN}Distribución del Semáforo:{Style.RESET_ALL}")
    for nivel, cantidad in reporte_general["Distribución del Semáforo"].items():
        st.write(f"{nivel}: {cantidad} llamadas")

    # Mostrar el análisis DOFA general
    st.write(f"\n{Fore.CYAN}Análisis DOFA General:{Style.RESET_ALL}")
    st.write(reporte_general["Análisis DOFA General"])

    st.write("\n" + "=" * 80)

# Función principal para procesar llamadas
def process_calls():
    st.title("Análisis de Llamadas de Ventas")

    # Solicitar el rango de fechas desde la interfaz de Streamlit
    fecha_desde = st.date_input("Introduce la fecha de inicio (YYYY-MM-DD):")
    fecha_hasta = st.date_input("Introduce la fecha de fin (YYYY-MM-DD):")

    # Convertir las fechas a objetos datetime
    fecha_desde_dt = datetime.datetime.combine(fecha_desde, datetime.time.min)
    fecha_hasta_dt = datetime.datetime.combine(fecha_hasta, datetime.time.max)

    # Convertir las fechas a timestamp UNIX en milisegundos
    fecha_desde_timestamp = int(fecha_desde_dt.timestamp() * 1000)
    fecha_hasta_timestamp = int(fecha_hasta_dt.timestamp() * 1000)

    # Obtener todas las llamadas en el rango de fechas
    results = fetch_all_calls(fecha_desde_timestamp, fecha_hasta_timestamp)

    # Extraer los call IDs y URLs de grabación
    call_ids = [result.id for result in results]
    recording_urls = [result.properties.get("hs_call_recording_url") for result in results]
    hs_createdates = [result.properties.get("hs_createdate") for result in results]

    # Filtrar las llamadas que tienen una URL de grabación válida
    valid_calls = [{"Call ID": call_id, "Recording URL": url, "hs_createdate": hs_createdate}
                   for call_id, url, hs_createdate in zip(call_ids, recording_urls, hs_createdates) if url]

    # Crear un DataFrame con los resultados
    df = pd.DataFrame(valid_calls)

    st.write(f"Se encontraron {len(df)} llamadas con grabaciones válidas.")
    st.write(df)

    if df.empty:
        st.warning("No se encontraron llamadas con grabaciones. Por favor, realiza otra búsqueda en otras fechas.")
    else:
        # Permitir al usuario seleccionar los IDs de las llamadas a transcribir
        st.write("\nLlamadas disponibles para transcripción:")
        for index, row in df.iterrows():
            st.write(f"Call ID: {row['Call ID']}, Fecha: {row['hs_createdate']}")

        selected_call_ids = st.text_input("\nIntroduce los IDs de las llamadas que deseas transcribir (separados por comas):").split(',')
        selected_call_ids = [call_id.strip() for call_id in selected_call_ids]

        # Filtrar el DataFrame con los IDs seleccionados
        df_selected = df[df['Call ID'].isin(selected_call_ids)]

        if df_selected.empty:
            st.warning("No se encontraron llamadas con los IDs proporcionados. Por favor, verifica los IDs e intenta nuevamente.")
        else:
            # Procesar las llamadas seleccionadas
            resultados_finales = []
            for index, row in df_selected.iterrows():
                call_id = row["Call ID"]
                recording_url = row["Recording URL"]

                st.write(f"\nProcesando llamada {call_id}...")

                # Descargar la grabación
                audio_file_path = download_call_audio(call_id, recording_url)

                # Transcribir el audio a texto
                transcription = None
                if audio_file_path:
                    transcription = transcribe_audio_to_text(audio_file_path)

                # Si la transcripción es válida, analizar la llamada
                if transcription:
                    analisis_pasos = analyze_transcription_with_retries(transcription)
                    if analisis_pasos:  # Solo continuar si el análisis fue exitoso
                        porcentaje_cumplimiento = calculate_compliance_percentage(analisis_pasos)
                        semaforo, dofa = generate_traffic_light_and_dofa(porcentaje_cumplimiento)
                        resultados_finales.append({
                            "Call ID": call_id,
                            "Transcription": transcription,
                            "Análisis Pasos Obligatorios": analisis_pasos,
                            "Porcentaje Cumplimiento": porcentaje_cumplimiento,
                            "Semáforo": semaforo,
                            "DOFA": dofa
                        })
                else:
                    st.warning(f"La llamada {call_id} no pudo ser transcrita. Omitiendo...")

            # Crear un DataFrame con los resultados finales
            df_resultados = pd.DataFrame(resultados_finales)

            # Guardar los resultados en un archivo CSV
            df_resultados.to_csv("analisis_llamadas.csv", index=False)
            st.success("\nAnálisis de llamadas guardado en 'analisis_llamadas.csv'.")

            # Generar el reporte general
            reporte_general = generate_general_report(resultados_finales)

            # Mostrar el reporte general
            display_general_report(reporte_general)

# Inicializar el modelo de Google Vertex AI
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Ejecutar el proceso principal
if __name__ == "__main__":
    process_calls()