import cv2
from deepface import DeepFace

# Camera start pannum
cap = cv2.VideoCapture(0)

print("Camera started! Press 'q' to quit...")

while True:
    # Frame capture pannum
    ret, frame = cap.read()
    
    if not ret:
        break
    
    try:
        # Emotion detect pannum
        result = DeepFace.analyze(frame, 
                                   actions=['emotion'],
                                   enforce_detection=False)
        
        # Emotion extract pannum
        emotion = result[0]['dominant_emotion']
        
        # Screen-la text show pannum
        cv2.putText(frame, f'Emotion: {emotion}', 
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Frame display pannum
    cv2.imshow('Face Emotion Detector', frame)
    
    # Q press panna quit aagum
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Camera release pannum
cap.release()
cv2.destroyAllWindows()