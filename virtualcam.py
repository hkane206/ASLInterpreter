import cv2
import numpy as np
import pyvirtualcam
import time
import mediapipe as mp

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

# Initialize virtual webcam. Set the width, height, and FPS to match source webcam.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
i = 0 # Counter for file naming

# Initialize mediapipe hand detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define the code and create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'avc1'
out = None
start_time = None
recording = False

start_time = time.time()

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    print(f'Virtual cam started: {cam.device} ({cam.width}x{cam.height} @ {cam.fps}fps)')

    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        
        while True:
            
            # Capture frame-by-frame from the real webcam
            ret, image = cap.read()

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                print("Hand detected")

                if not recording: # Start recording if hand detected and not already
                    recording = True
                    start_time = time.time()
                    out = cv2.VideoWriter(f'output_{i}.mp4', fourcc, 30, (width, height))

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
            else:
                print("No hand detected")

            if recording and time.time() - start_time >= 2:
                i += 1  # Increment file counter
                recording = False
                out.release()  # Close the current output file

            if recording:       
                out.write(image)

            # Add text to the frame
            text_frame = cv2.putText(image, 'Hello, World!', (540, 1000), cv2.FONT_HERSHEY_SIMPLEX , 4, (0,0,0), 2, cv2.LINE_AA)
            
            # Flip frame 
            text_frame = cv2.flip(text_frame, 1)

            resized_frame = cv2.resize(text_frame, (cam.width, cam.height))
            flipped_frame = cv2.flip(resized_frame, 1)

            # Convert the frame to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            # Send to virtual camera
            cam.send(frame_rgb)

            # Optional: Sleep for a while to sync FPS (use if needed)
            cam.sleep_until_next_frame()

            # Display the resulting frame for debugging
            #cv2.imshow('Real Camera Output', frame)

            # Press 'q' to quit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if out is not None:
    out.release()

cap.release()
cv2.destroyAllWindows()