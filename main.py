from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np
import joblib
from skimage.feature import hog


# Definir la clase para las respuestas de las predicciones
class PredictionResponse(BaseModel):
    id: int
    name: str
    description: str

# Cargar el modelo entrenado
clf = joblib.load('modelo_clasificacion.pkl')


class_predictions = {
    0: {"name": "Límite de Velocidad 20", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 20 kilómetros por hora."},
    1: {"name": "Límite de Velocidad 30", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 30 kilómetros por hora."},
    2: {"name": "Límite de Velocidad 50", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 50 kilómetros por hora."},
    3: {"name": "Límite de Velocidad 60", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 60 kilómetros por hora."},
    4: {"name": "Límite de Velocidad 70", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 70 kilómetros por hora."},
    5: {"name": "Límite de Velocidad 80", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 80 kilómetros por hora."},
    6: {"name": "Fin de Límite de Velocidad 80", "description": "Esta señal de tráfico indica el final de una restricción de velocidad previamente impuesta de 80 kilómetros por hora."},
    7: {"name": "Límite de Velocidad 100", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 100 kilómetros por hora."},
    8: {"name": "Límite de Velocidad 120", "description": "Esta señal indica que la velocidad máxima permitida en esta zona es de 120 kilómetros por hora."},
    9: {"name": "Prohibición de adelantar para vehiculos", "description": "Esta señal prohíbe el adelantar a otros vehículos."},
    10: {"name": "Prohibición de adelantar para camiones", "description": "Esta señal prohíbe a los vehículos pesados, como los camiones, adelantar a otros vehículos."},
    11: {"name": "Intersección con prioridad", "description": "Esta señal de tráfico triangular advierte a los conductores de la necesidad de ceder el paso a otros usuarios que se aproximan desde una dirección prioritaria en una intersección."},
    12: {"name": "Señal de advertencia de obras.", "description": "Indica que se debe proceder con precaución y estar atentos a posibles desviaciones, trabajadores en la carretera, maquinaria pesada, o cualquier otro tipo de irregularidad en la vía que pueda afectar la circulación normal."},
    13: {"name": "Borde de advertencia", "description": "El borde rojo es universalmente reconocido y diseñado para llamar la atención inmediatamente. (condicion peligrosa)"},
    14: {"name": "Stop", "description": " Se utiliza en situaciones donde es necesario que los conductores se detengan completamente."},
    15: {"name": "Prohibido el Paso / No Entrar", "description": "Indica que no se permite la entrada a los vehículos en la carretera a la que se enfrenta la señal."},
    16: {"name": "Prohibición de Paso para Camiones", "description": "Esta señal indica que la entrada no está permitida para camiones o vehículos de carga pesada. "},
    17: {"name": "Prohibido el Paso", "description": "Indica que la circulación no está permitida en esa dirección para ningún tipo de vehículo."},
    18: {"name": "Peligro General", "description": "Esta señal se utiliza para advertir a los conductores de un peligro inespecífico o una condición peligrosa en la carretera."},
    19: {"name": "Prohibido girar a la izquierda", "description": "Esta señal se utiliza para indicar a los conductores que no está permitido realizar un giro a la izquierda en la intersección o carretera próxima."},
    20: {"name": "Prohibido girar a la derecha", "description": "Esta señal se utiliza para indicar a los conductores que no está permitido realizar un giro a la derecha en la intersección o carretera próxima."},
    21: {"name": "Curva Peligrosa a la Izquierda", "description": "Advierte al conductor de una curva pronunciada hacia la izquierda en el camino adelante."},
    22: {"name": "Badenes o Resaltos", "description": "Indica la presencia de badenes, resaltos o baches en la carretera que requieren disminuir la velocidad."},
    23: {"name": "Pavimento deslizante", "description": "advierte sobre una vía que puede estar resbaladiza, por ejemplo, debido a lluvia, hielo o derrames de aceite."},
    24: {"name": "Estrechamiento de calzada por ambos lados", "description": "Indica que más adelante la carretera se estrecha por ambos lados, advirtiendo a los conductores que deben proceder con precaución."},
    25: {"name": "Obras en la carretera", "description": "Esta señal advierte a los conductores de la presencia de obras viales en la zona y sugiere proceder con cautela."},
    26: {"name": "Semaforo proximo", "description": "Esta señal generalmente advierte a los conductores sobre la presencia de un semáforo en las proximidades."},
    27: {"name": "Precaución: paso de peatones", "description": "Advierte a los conductores que se aproximan a una zona donde los peatones podrían estar cruzando la carretera y deben proceder con precaución."},
    28: {"name": "Cuidado con los peatones", "description": "Esta señal es comúnmente utilizada en zonas escolares o áreas residenciales donde los niños pueden estar presentes."},
    29: {"name": "Señal de Prohibición de Bicicletas", "description": "Indica que el tránsito de bicicletas está prohibido en esta ruta o área."},
    30: {"name": "Señal de Advertencia de Hielo o Nieve", "description": "Advierte a los conductores que deben tener cuidado debido a la posible presencia de hielo o nieve en la carretera."},
    31: {"name": "Prohibido el paso de animales salvajes", "description": "Esta señal advierte a los conductores que no está permitido que animales salvajes crucen esa zona."},
    32: {"name": "Prohibido", "description": "Una señal de una prohibicion"},
    33: {"name": "Sentido obligatorio hacia la derecha", "description": "La flecha señala la dirección y sentido que los vehículos tiene la obligación de seguir. En este caso es hacia la derecha."},
    34: {"name": "Sentido obligatorio hacia la izquierda", "description": "La flecha señala la dirección y sentido que los vehículos tiene la obligación de seguir. En este caso es hacia la izquierda."},
    35: {"name": "Sentido unico", "description": "La señal indica que la calzada es de sentido unico"},
    36: {"name": "Únicas direcciones y sentidos permitidos adelante/derecho", "description": "Las flechas señalan las únicas direcciones y sentidos que los vehículos pueden tomar. En este caso, es hacia la derecha."},
    37: {"name": "Únicas direcciones y sentidos permitidos adelante/izquierdo", "description": "Las flechas señalan las únicas direcciones y sentidos que los vehículos pueden tomar. En este caso, es hacia la izquierda."},
    38: {"name": "Paso obligatorio derecho", "description": "la flecha señala el lado del refugio (zona comprendida en el ancho de una calzada y destinada a la estancia de peatones para fraccionar el tiempo de cruce) por el que los vehículos han de pasar. A la derecha"},
    39: {"name": "Paso obligatorio izquierdo", "description": "la flecha señala el lado del refugio (zona comprendida en el ancho de una calzada y destinada a la estancia de peatones para fraccionar el tiempo de cruce) por el que los vehículos han de pasar. A la izquierda"},
    40: {"name": "Intersección de sentido giratorio obligatorio", "description": "las flechas señalan la dirección y sentido del movimiento giratorio que los vehículos deben seguir."},
    41: {"name": "No rebasar vehiculos", "description": "La señal de tránsito prohíbe a los conductores rebasar en la zona."},
    42: {"name": "No rebasar vehiculos - camiones", "description": "La señal de tránsito prohíbe a los conductores de camiones rebasar en la zona."},
}

app = FastAPI()

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Convertir la imagen cargada en un array de numpy
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Realiza la extracción de características (asegúrate de aplicar las mismas transformaciones que en el entrenamiento)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (224, 224))  # Ajusta el ancho y alto deseados

    # Aplica HOG a la imagen en escala de grises
    features, hog_image = hog(resized_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

    # Calcula el histograma de color y aplana
    color_histogram = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

    # Combina las características HOG y de color
    combined_features = np.concatenate([features, color_histogram])

    # Aplana las características
    combined_features = combined_features.reshape(1, -1)

  #  img_array = np.array(resized_img).reshape(1, -1)  # Aplana la imagen y añade una dimensión extra para el batch

    # Realiza la predicción
    prediction = clf.predict(combined_features)[0]  # Asumiendo que clf.predict devuelve un array, tomamos el primer elemento

    # Obtener el nombre y la descripción de la predicción
    pred_name = class_predictions[prediction]['name']
    pred_description = class_predictions[prediction]['description']

    # Crear y retornar la respuesta
    response = PredictionResponse(id=prediction, name=pred_name, description=pred_description)
    return response

#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="127.0.0.1", port=8000)

