import cv2
import numpy as np
from typing import List, Union, Optional




def display(input_img):
    cv2.imshow("ig", input_img)
    cv2.waitKey(0)
    
    
def rotateImg(img:np.ndarray, points: list) -> tuple[np.ndarray, list]:
    """
        Realiza a rotação de uma imagem e calcula a nova posição dos pontos de interesse.

        A função calcula o ângulo de rotação com base em dois pontos de referência e gira a imagem de acordo.
        Ela também calcula as novas posições dos pontos após a rotação, retornando a imagem rotacionada
        e as novas coordenadas dos pontos.

        Args:
            img (np.ndarray): A imagem a ser rotacionada. Deve ser uma matriz NumPy representando a imagem.
            points (dict): Dicionário com os pontos de interesse, onde as chaves são índices e os valores são
                        dicionários contendo as coordenadas 'x' e 'y' de cada ponto (ex: {0: {'x': 100, 'y': 150}}).

        Returns:
            tuple: Tupla contendo:
                - img_rotated (np.ndarray): A imagem rotacionada.
                - new_points (list): Lista de dicionários contendo as novas coordenadas dos pontos após a rotação.
    """

    
    x1, y1 = points[0]['x'],points[0]['y']
    x2, y2 = points[1]['x'], points[1]['y']

    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calcular o novo tamanho da imagem após a rotação
    abs_cos = abs(rotation_matrix[0,0]) 
    abs_sin = abs(rotation_matrix[0,1])

    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Ajustar a matriz de transformação
    rotation_matrix[0, 2] += new_width/2 - center[0]
    rotation_matrix[1, 2] += new_height/2 - center[1]

    # Realizar a rotação
    rotated_img = cv2.warpAffine(img, rotation_matrix, (new_width, new_height))



    
    
    
    
        # Calcular os novos pontos após a rotação
    new_points = []
    for point in points:
        # Adicionar a coordenada homogênea (1) ao ponto
        x, y = point['x'], point['y']
        point_hom = np.array([x, y, 1]).reshape(3, 1)
        
        rotated_point = rotation_matrix.dot(point_hom)  # Multiplicação da matriz

        new_points.append({'x': rotated_point[0, 0], 'y': rotated_point[1, 0]})

    return rotated_img, new_points


def cropImg(image: np.ndarray, points: list) -> np.ndarray:
    """
    Realiza o corte (crop) de uma imagem com base em dois pontos fornecidos.

    A função recebe os pontos de interesse (coordenadas) e utiliza esses pontos para
    definir a região da imagem a ser recortada. A imagem recortada é retornada.

    Args:
        image (np.ndarray): A imagem original a ser recortada, no formato NumPy array (BGR).
        points (list): Lista de dicionários, onde cada dicionário contém as coordenadas 'x' e 'y' de um ponto
                       na imagem. A função usa os dois primeiros pontos para calcular o corte.

    Returns:
        np.ndarray: A imagem recortada após aplicar os limites definidos pelos pontos fornecidos.
    """
    x1, y1 = int(points[0]['x']),int(points[0]['y'])
    x2, y2 = int(points[1]['x']),int(points[1]['y'])
 
    # Realizar o corte da imagem com os valores mínimos e máximos
    image = image[y1+12:870,x1+14:x2]
    return image


def countours(image: np.ndarray) -> np.ndarray:
    """
    Encontra os contornos de uma imagem binária.

    Essa função recebe uma imagem em escala de cinza (ou binária), corta a imagem pela metade (ao longo da altura),
    e então utiliza o algoritmo de detecção de contornos do OpenCV para encontrar e retornar os contornos externos presentes na imagem.

    Args:
        image (np.ndarray): Imagem de entrada no formato `np.ndarray`, geralmente uma imagem binária ou em escala de cinza,
                            onde os contornos são visíveis. A imagem deve ter pelo menos duas dimensões (altura, largura).

    Returns:
        np.ndarray: Lista de contornos encontrados na imagem. Cada contorno é representado por uma sequência de pontos 
                    (coordenadas) que definem o contorno na imagem.
                    O formato da saída é um array de contornos, onde cada contorno é um array de pontos.
    """
    altura, largura = image.shape[:2]
    image = image[0:altura//2, ]
    
    
    
    # altura, largura = image.shape[:2]
    # image = image[0:(altura-250)//2, ]
    
    # contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # contours = contours[0] if len(contours) == 2 else contours[1]
    
    # return contours
    
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    return contours




def noNoise(img: np.ndarray) -> np.ndarray:
    """
        Realiza o pré-processamento de uma imagem para redução de ruído e melhoria na detecção de contornos.

        Essa função converte a imagem para escala de cinza, aplica um desfoque (mediana) para redução de ruído,
        em seguida realiza uma limiarização para binarizar a imagem e, finalmente, aplica uma operação morfológica 
        para fechar pequenos buracos e melhorar a conectividade dos objetos.

        Args:
            img (np.ndarray): Imagem de entrada, geralmente no formato colorido (BGR), que será processada.
                            O formato é um array NumPy com 3 dimensões para imagens coloridas ou 2 dimensões para imagens em escala de cinza.

        Returns:
            np.ndarray: Imagem processada após a remoção de ruídos, com bordas mais destacadas e adequada para detecção de contornos.
                A imagem retornada é um array NumPy em formato binário (0s e 255s), preparado para análises subsequentes.
    """


    blur = cv2.medianBlur(img,5)

    thresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    return thresh



def findPoints(cnts: list) -> list:
    """
    Encontra pontos de interesse em uma imagem com base em contornos detectados.

    A função recebe uma lista de contornos detectados na imagem e, para cada contorno com uma área
    dentro de um intervalo específico, calcula o retângulo delimitador (bounding box) e armazena suas
    coordenadas e área. A função retorna um dicionário com as informações de cada ponto de interesse.

    Args:
        cnts (list): Lista de contornos detectados na imagem, normalmente obtida a partir de uma
                     operação como `cv2.findContours`. Cada contorno é representado por uma sequência
                     de pontos (coordenadas) que delimitam uma região da imagem.

    Returns:
        dict: Dicionário contendo as informações dos pontos de interesse encontrados. Para cada ponto, 
              é armazenado um dicionário com as coordenadas do retângulo delimitador (x, y, largura, altura)
              e a área do contorno. As chaves do dicionário são índices (inteiros) que identificam os pontos.
    """
    
    
    
    min_area = 90
    max_area = 108
    points = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            x,y,w,h = cv2.boundingRect(c)
            points.append({"x":x,"y":y,"w":w,"h":h,"area":area})
    
    points = sorted(points, key=lambda x: x['y'])[:2]
    return points






# Função para calcular a distância euclidiana (utilizada para ordenar os círculos)
def dist(a):
    return np.sqrt(a[0]**2 + a[1]**2).astype(int)

def agrupar_por_linhas(rects: list, tolerancia: int = 5) -> list:
    """
    Agrupa os retângulos em linhas com base na proximidade vertical dos retângulos. 
    Retângulos com uma diferença vertical menor que a tolerância (ajustada pela altura) 
    serão considerados parte da mesma linha.

    Args:
        rects (list): Lista de retângulos, onde cada retângulo é representado por uma tupla 
                      (x, y, w, h), sendo (x, y) as coordenadas do canto superior esquerdo 
                      e (w, h) a largura e altura do retângulo.
        tolerancia (int, opcional): Valor que define a tolerância para agrupar os retângulos 
                                     em uma linha. A tolerância é ajustada dinamicamente com base 
                                     na altura dos retângulos. O padrão é 5.

    Returns:
        list: Lista de listas, onde cada sublista contém os retângulos que foram agrupados 
              na mesma linha.
    """
    linhas = []
    linha_atual = [rects[0]]

    for i in range(1, len(rects)):
        x_atual, y_atual, w_atual, h_atual = rects[i]
        x_anterior, y_anterior, w_anterior, h_anterior = linha_atual[-1]

        tolerancia_dinamica = max(h_atual, h_anterior) // 2 + tolerancia
        print(f" {linha_atual[-1]}")
        print(f" {tolerancia_dinamica} ")
        if abs(y_atual - y_anterior) <= tolerancia_dinamica or abs((y_atual + h_atual) - (y_anterior + h_anterior)) <= tolerancia_dinamica:
            linha_atual.append(rects[i])
        else:
            # Finaliza a linha atual e inicia uma nova
            linhas.append(linha_atual)
            linha_atual = [rects[i]]

    # Adiciona a última linha
    if linha_atual:
        linhas.append(linha_atual)

    return linhas





def orderQuestions(cntrRect: list) -> list:
    """
    Ordena as questões de um cartão de resposta com base na posição das caixas delimitadoras 
    (bounding boxes) e na distância relativa entre as questões.

    A função agrupa as questões por linha e, dentro de cada linha, ordena-as pela posição 
    horizontal (da esquerda para a direita). A ordenação é feita calculando a distância
    de cada ponto em relação a um ponto inicial, para garantir que as questões sejam numeradas 
    corretamente de acordo com sua posição.

    Args:
        cntrRect (list): Lista de tuplas, onde cada tupla contém a coordenada (x, y) de um retângulo 
                          delimitador de uma questão. Cada retângulo pode ser representado como uma 
                          tupla (x, y, w, h), onde `x` e `y` são as coordenadas do canto superior esquerdo 
                          e `w` e `h` são a largura e altura do retângulo.

    Returns:
        list: Lista de tuplas onde cada tupla contém as coordenadas do retângulo e o número da questão 
              correspondente, ordenado da esquerda para a direita e de cima para baixo.
    """
    cntrRect = sorted(cntrRect, key=lambda y: y[1])
    linhas = agrupar_por_linhas(cntrRect)
    
    question = 1 
    linhas_order = {} 
    for linha in linhas:
        distancias = {}
        
        for rectQuestion in linha:
            distancia = dist(np.array([rectQuestion[0], rectQuestion[1]]))
            distancias[rectQuestion] = distancia
        
        distancias = sorted(distancias.items(), key=lambda item: item[1])
        
        for dista in distancias:
            linhas_order[dista[0]] = question
            question += 1  
    
    return list(linhas_order.items())


