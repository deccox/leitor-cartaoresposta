import cv2
import numpy as np
from rotate import *
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass
from typing import Tuple

def display(input_img):
    cv2.imshow("ig", input_img)
    cv2.waitKey(0)
    
    
# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lock global para atualização thread-safe do contador
correct_lock = Lock()
correct_counter = 0

@dataclass
class QuestionData:
    coords: Tuple[int, int, int, int]
    question_num: int
    image: np.ndarray
    answer: str
def process_question(data: QuestionData) -> int:
    """Processa uma única questão em uma thread"""
    correct_local = 0  # Contador local para esta thread
    
    x, y, w, h = data.coords
    roi_resized = data.image[y+w//2:y+h, x:x+w]
    
    circulos = cv2.HoughCircles(
        roi_resized, 
        cv2.HOUGH_GRADIENT, 
        dp=1.5,
        minDist=50,
        param1=60,
        param2=40,
        minRadius=25,
        maxRadius=38
    )
    
    if circulos is not None:
        circulos = np.round(circulos[0, :]).astype("int")
        circulos = sorted(circulos, key=lambda x: np.sqrt(x[0]**2 + x[1]**2))
        count_valid_circles = 0  # Contador de círculos com mean_val < 100
        valid_circle = None      # Armazena o único círculo válido encontrado, se houver
        
        for a, b, c in circulos:
            mask = np.zeros(roi_resized.shape, dtype="uint8")
            cv2.circle(mask, (a, b), c, 255, -1)
            mean_val = cv2.mean(roi_resized, mask=mask)[0]
            cv2.circle(roi_resized, (a, b), c, 255, 2)
            if mean_val < 100:
                count_valid_circles += 1
                if count_valid_circles > 1:
                    # Mais de um círculo válido encontrado, pular este retângulo
                    print(f"Aqui foram animais e marcaram {count_valid_circles} círculos")
                    return correct_local
                valid_circle = (a, b, c)  # Armazena o único círculo válido até o momento
        
        if count_valid_circles == 1 and valid_circle:
            a, b, c = valid_circle
            # Encontrar posição do círculo preenchido
            matches = np.all(circulos == np.array([a, b, c]), axis=1)
            index = np.where(matches)[0]

            if len(index) > 0 and index[0] < 5:
                pos = ['A', 'B', 'C', 'D', 'E'][index[0]]
                if pos == data.answer:
                    correct_local += 1
    #display(roi_resized)
    return correct_local

def processImg(path: str, gabarito: dict) -> None:
    """
    Função responsável por carregar a imagem e aplicar as transformacoes necessárias para encontrar os contornos

    Args:
        path (str): o Nome da imagem ex ('cartao1.png')
        gabarito (dict): Um dict contendo o gabarito do cartao resposta
    """
    input_img = cv2.imread(path)
    
    
    scale = 0.45
    
    # Pre-processamento inicial
    input_img = cv2.resize(input_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar processamentos da função rotate
    imageNoNoise = noNoise(input_img)
    contoursPoints = countours(imageNoNoise)
    points = findPoints(contoursPoints)
    image, points = rotateImg(input_img, points)
    input_img = cropImg(image, points)
    
    #display(image)
    # Detectar retângulos
    blurred = cv2.GaussianBlur(input_img, (5, 5), 1)
    canny = cv2.Canny(blurred, 10, 50)
    contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    # Processar contornos
    cntrRect = []
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            cntrRect.append(cv2.boundingRect(approx))
    
    # Ordenar retângulos
    cntrRect = orderQuestions(cntrRect)
    
    # Preparar imagem final
    input_img = cv2.resize(input_img, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    input_img = cv2.medianBlur(input_img, 5)
    
    # Preparar dados para processamento paralelo
    question_data = []
    for ctr in cntrRect:
        x, y, w, h = ctr[0]
        question = ctr[1]
        
        # Ajustar coordenadas para a escala
        coords = (x*5, y*5, w*5, h*5)
        
        question_data.append(QuestionData(
            coords=coords,
            question_num=question,
            image=input_img,
            answer=gabarito[question]
        ))
    
    # Processar questões em paralelo
    total_correct = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_question, data) for data in question_data]
        for future in as_completed(futures):
            try:
                total_correct += future.result()
            except Exception as e:
                logger.error(f"Erro no processamento: {str(e)}")
    
    logger.info(f"Total corretas: {total_correct}/50")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    
    gabarito = {
        1: 'B', 2: 'A', 3: 'D', 4: 'A', 5: 'E', 6: 'B', 7: 'C', 8: 'D', 9: 'A', 10: 'B',
        11: 'A', 12: 'C', 13: 'C', 14: 'E', 15: 'D', 16: 'B', 17: 'E', 18: 'C', 19: 'A', 20: 'E',
        21: 'D', 22: 'E', 23: 'E', 24: 'A', 25: 'C', 26: 'C', 27: 'D', 28: 'B', 29: 'D', 30: 'D',
        31: 'A', 32: 'D', 33: 'D', 34: 'B', 35: 'C', 36: 'D', 37: 'B', 38: 'D', 39: 'D', 40: 'D',
        41: 'D', 42: 'E', 43: 'C', 44: 'A', 45: 'D', 46: 'B', 47: 'C', 48: 'A', 49: 'D', 50: 'E'
    }
    
    for i in range(0,150):
        processImg('cart2fuck.png', gabarito)
    logger.info(f"Processing time: {(time.time() - start_time):.2f} seconds")