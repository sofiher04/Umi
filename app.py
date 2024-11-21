import os
#from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import PyPDF2
from PIL import Image as Image, ImageOps as ImagOps
import glob
from gtts import gTTS
import os
import time
from streamlit_lottie import st_lottie
import json
import paho.mqtt.client as mqtt
import pytz
import cv2
import yolov5
import streamlit as st
import numpy as np
import pandas as pd
#from ultralytics import YOLO

#import sys
#sys.path.append('./ultralytics/yolo')
#from utils.checks import check_requirements

MQTT_BROKER = "broker.mqttdashboard.com"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor_st"

# Variables de estado para los datos del sensor
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = None

def text_to_speech(text, tld):
                
    tts = gTTS(response,"es", tld , slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name, text


                
def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
      now = time.time()
      n_days = n * 86400
      for f in mp3_files:
         if os.stat(f).st_mtime < now - n_days:
             os.remove(f)

def send_mqtt_message(message):
    """Función para enviar un mensaje MQTT"""
    try:
        client = mqtt.Client()
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.publish("h_ctrl", message)
        client.disconnect()
        return True
    except Exception as e:
        st.error(f"Error al enviar mensaje MQTT: {e}")
        return False


def get_mqtt_message():
    """Función para obtener un único mensaje MQTT"""
    message_received = {"received": False, "payload": None}
    
    def on_message(client, userdata, message):
        try:
            payload = json.loads(message.payload.decode())
            message_received["payload"] = payload
            message_received["received"] = True
        except Exception as e:
            st.error(f"Error al procesar mensaje: {e}")
    
    try:
        client = mqtt.Client()
        client.on_message = on_message
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.subscribe(MQTT_TOPIC)
        client.loop_start()
        
        timeout = time.time() + 5
        while not message_received["received"] and time.time() < timeout:
            time.sleep(0.1)
        
        client.loop_stop()
        client.disconnect()
        
        return message_received["payload"]
    
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

try:
    os.mkdir("temp")
except:
    pass

with st.sidebar:
    st.subheader("¿Tienes alguna pregunta?")
    st.write(
    """¡Pregúntale a Umi! Está aquí para ayudarte
       
       
    """
                )            

st.title('Hola!!! Soy UMI 💬')
#image = Image.open('Instructor.png')
#st.image(image)
with open('umbird.json') as source:
     animation=json.load(source)
st.lottie(animation,width =350)

#ke = st.text_input('Ingresa tu Clave')
#os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"] #ke

#st.write(st.secrets["settings"]["key"])

pdfFileObj = open('Plantas y cuidados.pdf', 'rb')
 
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)


    # upload file
#pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

   # extract the text
#if pdf is not None:
from langchain.text_splitter import CharacterTextSplitter
 #pdf_reader = PdfReader(pdf)
pdf_reader  = PyPDF2.PdfReader(pdfFileObj)
text = ""
for page in pdf_reader.pages:
         text += page.extract_text()

   # split into chunks
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap=20,length_function=len)
chunks = text_splitter.split_text(text)

# create embeddings
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)


# Columnas para sensor y pregunta
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Datos del Sensor")
    if st.button("Obtener Lectura"):
        with st.spinner('Obteniendo datos del sensor...'):
            sensor_data = get_mqtt_message()
            st.session_state.sensor_data = sensor_data
            
            if sensor_data:
                st.success("Datos recibidos")
                st.metric("Temperatura", f"{sensor_data.get('Temp', 'N/A')}°C")
                st.metric("Humedad", f"{sensor_data.get('Hum', 'N/A')}%")
            else:
                st.warning("No se recibieron datos del sensor")

with col2:
    st.subheader("Realiza tu consulta")
    user_question = st.text_area("Escribe tu pregunta aquí:")
    
    if user_question:
        # Incorporar datos del sensor en la pregunta si están disponibles
        if st.session_state.sensor_data:
            enhanced_question = f"""
            Contexto actual del sensor:
            - Temperatura: {st.session_state.sensor_data.get('Temp', 'N/A')}°C
            - Humedad: {st.session_state.sensor_data.get('Hum', 'N/A')}%
            
            Pregunta del usuario:
            {user_question}
            """
        else:
            enhanced_question = user_question
        
        docs = knowledge_base.similarity_search(enhanced_question)
        llm = OpenAI(model_name="gpt-4")
        chain = load_qa_chain(llm, chain_type="stuff")
        
        with st.spinner('Analizando tu pregunta...'):
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=enhanced_question)
                print(cb)
            
            st.write("Respuesta:", response)

            if st.button("Escuchar"):
              result, output_text = text_to_speech(response, 'es-es')
              audio_file = open(f"temp/{result}.mp3", "rb")
              audio_bytes = audio_file.read()
              st.markdown(f"## Escucha:")
              st.audio(audio_bytes, format="audio/mp3", start_time=0)

             
#                            print("Deleted ", f)
            
            
#          remove_files(7)


# Cerrar archivo PDF
pdfFileObj.close()
st.subheader("Compara tu planta con alguna de estas imágenes y pregúntale a Umi sobre sus cuidados")

# Ruta de las imágenes
image_files = [
    "Cacti.jpeg",
    "MonsteraDeliciosa.jpg",
    "Orquid.jpg",
    "Suculentas.jpg",
]

# Mostrar las imágenes en columnas
cols = st.columns(len(image_files))

for col, img_path in zip(cols, image_files):
    with col:
        image = Image.open(img_path)
        st.image(image, use_column_width=True, caption=os.path.basename(img_path))

# load pretrained model
model = yolov5.load('yolov5s.pt')
#model = yolov5.load('yolov5nu.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# take a picture with the camera
st.title("Detección de Objetos en Imágenes")

with st.sidebar:
            st.subheader('Parámetros de Configuración')
            model.iou= st.slider('Seleccione el IoU',0.0, 1.0)
            st.write('IOU:', model.iou)

with st.sidebar:
            model.conf = st.slider('Seleccione el Confidence',0.0, 1.0)
            st.write('Conf:', model.conf)


picture = st.camera_input("Capturar foto",label_visibility='visible' )

if picture:
    #st.image(picture)

    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
  
    # perform inference
    results = model(cv2_img)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] 
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    col1, col2 = st.columns(2)

    with col1:
        # show detection bounding boxes on image
        results.render()
        # show image with detections 
        st.image(cv2_img, channels = 'BGR')

    with col2:      

        # get label names
        label_names = model.names
        # count categories
        category_count = {}
        for category in categories:
            if category in category_count:
                category_count[category] += 1
            else:
                category_count[category] = 1        

        data = []        
        # print category counts and labels
        for category, count in category_count.items():
            label = label_names[int(category)]            
            data.append({"Categoría":label,"Cantidad":count})
        data2 =pd.DataFrame(data)
        
        # agrupar los datos por la columna "categoria" y sumar las cantidades
        df_sum = data2.groupby('Categoría')['Cantidad'].sum().reset_index() 
        df_sum
