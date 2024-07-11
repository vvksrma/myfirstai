import cv2
import mediapipe as mp     #Importing all Needed Modules & Packages
import time

cap = cv2.VideoCapture(0)  #Creating variable to get access to the camera

mpHand = mp.solutions.hands 
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #Defining while loop to read the hand movement 
    results = hands.process(imgRGB)               #and return the landmark valye in terms of x and y.
    #print(results.multi_hand_landmarks)
    
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)  #printing the ID and Landmark of hand movement
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) #creating a for loop to get ID and 
                print(id, cx, cy)                     #Landmark of hand movement in pixels
                

                #use this if function to find the movement through landmark id ex: 1, 2, 4 etc.
                #if id == 4:
                    #cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED) #defining the shape and size of pointer
            
            
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS) #Drawing the landmarks


    cTime = time.time()
    fps = 1/(cTime-pTime) #Explaining how to calculate FPS to the program
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)


    cv2.imshow("Image", img)
    cv2.waitKey(1)