import cv2, os, numpy as np

class FaceEngine:

    def __init__(self):
        self.dataset = "dataset"
        self.face = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        os.makedirs(self.dataset, exist_ok=True)

    def save_face(self, name, frame):
        path = f"{self.dataset}/{name}"
        os.makedirs(path, exist_ok=True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count = len(os.listdir(path))
        cv2.imwrite(f"{path}/{count}.jpg", gray)

    def recognize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return "No Face"

        (x,y,w,h) = faces[0]
        face_img = cv2.resize(gray[y:y+h, x:x+w], (100,100))

        best, min_diff = "Unknown", 999999

        for person in os.listdir(self.dataset):
            for img in os.listdir(f"{self.dataset}/{person}"):
                path = f"{self.dataset}/{person}/{img}"
                saved = cv2.imread(path, 0)
                if saved is None: continue
                saved = cv2.resize(saved,(100,100))

                diff = np.sum(cv2.absdiff(saved, face_img))
                if diff < min_diff:
                    min_diff = diff
                    best = person

        return best if min_diff < 300000 else "Unknown"