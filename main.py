import cv2

def DetectFace(faceNet, frame):
    frameHeight = frame.shape[0] # shape возвращает форму массива, которая определяется числом элементов вдоль каждой оси
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# пути к моделям обнаружения лиц, определения пола и возраста
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# загрузка модели обнаружения лиц, определения пола и возраста с диска используя ранее указанные пути к ним
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

#задаем диапазоны определяемых возрастов и список определяемых полов (мужской, женский)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# открываем готовый файл и фото/видео или выбираем распознавание в режиме реального времени
video = cv2.VideoCapture(0)  # Выбираем устройство видеозахвата

padding = 20

while True:
    ret, frame = video.read()
    frame, bboxs = DetectFace(faceNet, frame)
    for bbox in bboxs:
        # face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        # получаем координаты лиц и  формируем 4-х размерный двоичный объект (blob)
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        # пропускаем blob через модель определения пола и получать уровень доверия для двух классов - Male и Female
        # для какого класса уровень доверия будет больше, тот и будет определенным на изображении полом человека
        # blobFromImage создает 4-мерное пятно из изображения. Опционально изменяет размеры и обрезает изображение по центру,
        # вычитает средние значения, масштабирует значения по коэффициенту

        # предсказание пола
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        # предсказание возраста
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]
        # argmax() возвращает индекс максимального значения для предсказываемого возраста

        label = "{},{}".format(gender, age)
        # добавляем рамку и результаты определения пола и возраста
        cv2.rectangle(frame, (bbox[0], bbox[1] - 30), (bbox[2], bbox[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                    cv2.LINE_AA)
        # LINE_AA сглаживает линии
    cv2.imshow("Age-Gender Recognation", frame)
    k = cv2.waitKey(1)
    # если нажата клавиша `q`, то осуществляется выход из цикла
    if k == ord('q'):
        break
video.release() # закрываем видео файл или устройство захвата изображения
cv2.destroyAllWindows() # очищаем на экране результаты работы программы
