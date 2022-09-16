#Importar as dependências
import cv2
import time

#Cores das classes
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


#Carrega as classes de objetos
class_name = []
with open ("coco.names", "r") as f:
    class_names =  [cname.strip() for cname in f.readlines()]

#Carregando mídia / Para Webcam, definir o valor em 0
cap = cv2.VideoCapture("teste.mp4")

#Carregar Rede Neural com os Pesos treinados
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")


#Definindo os parâmetros da Rede Neural
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)


#Lendo os frames da mídia
while True:

    #Captura do frame
    _, frame = cap.read()

    start = time.time()

    #Detecção
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)

    end = time.time()

    #Percorre toda a detecção
    for (classid, score, box) in zip (classes, scores, boxes):
        
        #Gera uma cor para cada classe
        color = COLORS[int(classid) % len (COLORS)]

        #Definindo o nome da classe pelo ID e Score da acurancia 
        label = f"{class_names[classid]} : {score * 100}"

        #Desenhando o retângulo de detecção
        cv2.rectangle(frame, box, color, 2)
        
        #Escrevendo texto da ID da classe no retângulo de detecção
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color , 2)

    #Calculando o tempo de detecção
    fps_label = f"FPS: {round((1.0/(end - start)),2)}"
        

    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5)
    cv2.putText(frame, fps_label, (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

    #Mostrar a imagem
    cv2.imshow("Output", frame)

    #ESC para fechar
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
