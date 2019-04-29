from keras.models import load_model
import numpy as np
import cv2 as cv
from trainer import Trainer


class LFED:

    def __init__(self, train=False):
        if train:
            self.train()

        self.emotionsList = ['happy', 'normal', 'sad', 'sleepy', 'surprised', 'wink']
        
        # Face Model
        self.faceDetection = cv.CascadeClassifier('./haarcascade_frontalface_default.xml')
        
        # Trained Model
        self.model = load_model('./models/trainedModel', compile=False)

    def train(self):
        trainer = Trainer(imageSize= 48, imagesPath='./img/', batchSize=5)
        trainer.execute()

    def start(self):
        # Start recording
        cam = cv.VideoCapture(0)

        while True:
            # Get camera frame and create a copy in grayscale
            ret, frame = cam.read()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Detect the face using the face detection model
            faces = self.faceDetection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                        flags=cv.CASCADE_SCALE_IMAGE)

            # Draw rectangle if the face was detected
            if len(faces) > 0:
                # Get the position and width, height of the detected face
                (faceX, faceY, faceWidth, faceHeight) = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]

                # Extract ROI of the face,
                roi = gray[faceY:faceY + faceHeight, faceX:faceX + faceWidth]

                # Resize to 48x48 pixels
                roi = cv.resize(roi, (48, 48))

                # Get all predictions
                predictions = self.model.predict(roi[np.newaxis, :, :, np.newaxis])[0]

                # Get the emotion with the highest probability
                label = self.emotionsList[predictions.argmax()]

                # Draw probabilities and rectangle on face detection
                for (i, (emotion, prob)) in enumerate(zip(self.emotionsList, predictions)):
                    text = f'{emotion} = {prob * 100:.2f}'

                    # Draw probabilities by emotion
                    cv.putText(frame, text, (10, (i * 20) + 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

                    # Draw detected emotion
                    cv.putText(frame, label, (faceX, faceY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (185, 128, 41), 2)

                    # Draw rectangle around the face
                    cv.rectangle(frame, (faceX, faceY), (faceX + faceWidth, faceY + faceHeight), (185, 128, 41), 2)

                    # Debugging probabilities output
                    print(f'{emotion} => {prob * 100:.2f}')

            # If face was not detected draw the following text on the frame
            else:
                cv.putText(frame, 'No Face Detected', (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.45, (185, 128, 41), 2)

            # Show the frame
            cv.imshow('LFED (Live Face Emotion Detection)', frame)

            # Press 'q' to quit the program
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    lfed = LFED(train=False)
    lfed.start()
