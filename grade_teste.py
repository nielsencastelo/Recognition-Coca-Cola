import os
import dlib
import cv2
import glob
import numpy as np
import math
import sys


#print(dlib.test_simple_object_detector("Ninho/lata.xml", "Ninho/lata.svm"))

detectorLata = dlib.simple_object_detector("Grades/palete.svm")

for imagem in glob.glob(os.path.join("Grades/testes", "*.JPG")):
    img = cv2.imread(imagem)
    #img = cv2.resize(img, None, fx=0.3, fy=0.3)
    objetosDetectados = detectorLata(img)

    numObj = np.array(objetosDetectados)

    for d in objetosDetectados:
        e, t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
        cv2.rectangle(img, (e,t), (d, b), (0,255,0), 2)

    texto = "{} {:.1f}".format('Total: ', len(numObj))
    cv2.putText(img, texto, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), thickness=2)

    cv2.imshow("Detector de grades e palete", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

'''
pula_quadros = 30
captura = cv2.VideoCapture(0)
contadorQuadros = 0
detector = dlib.simple_object_detector("Ninho/lata.svm")
while captura.isOpened():
    conectado, frame = captura.read()
    #contadorQuadros += 1
    #if contadorQuadros % pula_quadros == 0:
    objetosDetectados = detector(frame, 1)
    for o in objetosDetectados:
        e, t, d, f = (int(o.left()), int(o.top()), int(o.right()), int(o.bottom()))
        cv2.rectangle(frame, (e, t), (d, f), (0, 0, 255), 2)

    cv2.imshow("Preditor de Latas", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

captura.release()
cv2.destroyAllWindows()
sys.exit(0)
'''