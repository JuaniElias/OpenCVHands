import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

colors = {
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'red': (0, 0, 255)
}
rect1 = (200, 10)
rect2 = (400, 200)

# For webcam input:
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Clip inicializado
clip = cv2.VideoWriter('output.mp4',
                       cv2.VideoWriter_fourcc(*'MP4V'),
                       10, size)

secs_to_record = 1
entered_rectangle = False

with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=4,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        h, w, _ = image.shape
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # Entro al rectángulo
            if rect1[0] <= results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w <= \
                    rect2[0] and rect1[1] <= results.multi_hand_landmarks[0].landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h <= \
                    rect2[1]:
                cv2.rectangle(image, rect1, rect2, colors['green'], 2)
                entered_rectangle = True
                # Empieza el timer para grabar
                stop_recording_sec = time.time() + secs_to_record
            else:
                cv2.rectangle(image, rect1, rect2, colors['red'], 2)
        else:
            cv2.rectangle(image, rect1, rect2, colors['blue'], 2)

        if entered_rectangle and time.time() < stop_recording_sec:
            clip.write(image)

        cv2.imshow('MediaPipe Hands', image)

        # Salir con Esc o 'q'
        if cv2.waitKey(5) & 0xFF == 27 or cv2.waitKey(5) & 0xFF == ord('q'):
            break

clip.release()
cap.release()
