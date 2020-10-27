from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import tkinter as tk
from tkinter import *
import tkinter.font as font


def emotions():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    classifier = load_model("emotions.h5")
    gender_labels = ["Man", "Woman"]
    class_labels = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (64, 0, 255), 4)
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y - 15)
                cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (89, 57, 230),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "No Face Found",
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (191, 64, 128),
                    1,
                )
        cv2.imshow("Emotion Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


window = tk.Tk()
window.geometry("500x500")
window.title("Face Recognition Options")
window.configure(bg="gray25")

fonting = font.Font(family="Times New Roman", size=20)
lab1 = Label(
    window,
    text="Face recognition controller",
    bg="gray25",
    fg="ghost white",
    font=("Helventica", 14),
).place(x=50, y=50)


Button(
    window, text="Emotion Detection", command=emotions, fg="ghost white", bg="gray20"
).place(x=50, y=150)

Label(
    text="Click on the camera window and press q to exit.",
    bg="gray25",
    fg="ghost white",
    font=("Helventica", 12),
).place(x=50, y=350)

window.mainloop()
