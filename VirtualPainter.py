import cv2  
import time
import numpy as np
from collections import deque
import mediapipe as mp

count_frames = 0
b = [0, 0, 0, 0, 0]
a = deque([[0, 0]], maxlen=5)
color = (255, 255, 255)
brush_sizes = [15, 30, 45, 60]
current_brush_size_index = 0

class HandsDetection():
    
    def __init__(self, num_hands=1, detection_confidence=0.8, 
                static_mode=False, tracking_confidence=0.8,):
        self.num_hands = num_hands
        self.detection_confidence = detection_confidence
        self.static_mode = static_mode
        self.tracking_confidence = tracking_confidence
        
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(max_num_hands=self.num_hands, 
                                        min_detection_confidence=self.detection_confidence, 
                                        static_image_mode=self.static_mode, 
                                        min_tracking_confidence=self.tracking_confidence)
        
    
    def DrawHandsLandmarks(self, frame):
        
        for i in self.results.multi_hand_landmarks:    
            self.mpDraw.draw_landmarks(frame, i, self.mpHands.HAND_CONNECTIONS)
        
        return frame
    

    def SelectColor(self, x, y):
        global color
        
        if ((750 < x < 950) and (50 < y < 150)):
            color = (255, 0, 0)
            return color
            
        elif ((270 < x < 470) and (50 < y < 150)):
            color = (0, 255, 0)
            return color
        
        elif ((520 < x < 720) and (50 < y < 150)):
            color = (0, 0, 255)
            return color
        
        elif ((20 < x < 220) and (50 < y < 150)):
            color = (0, 0, 0)
            return color
        
        elif ((20 < x < 220) and (220 < y < 320)):
            current_brush_size_index = 0
            return current_brush_size_index
        
        elif ((20 < x < 220) and (330 < y < 370)):
            current_brush_size_index = 1
            return current_brush_size_index
        
        elif ((20 < x < 220) and (380 < y < 420)):
            current_brush_size_index = 2
            return current_brush_size_index
        
        elif ((20 < x < 220) and (430 < y < 470)):
            current_brush_size_index = 3
            return current_brush_size_index
        
        return None
    
    
    def DrawOnScreen(self, frame, canvas, x, y):
        global a, count_frames, color, current_brush_size_index
        
        self.SelectColor(x, y)
        
        if ((x, y) != (0, 0) and (a[-1] == [0,0])):    
            a.append([x, y])
            return frame, canvas
        
        if ((x, y) != (0, 0) and (a[-1] != [0,0])):    
            cv2.circle(frame, (x, y), brush_sizes[current_brush_size_index], (200, 255, 200), -1)
            a.append([x, y])
            
            if sum(b) == 1:
                if color == (0, 0, 0):
                    x0, y0 = a[-2]
                    cv2.circle(canvas, (x, y), 7, color, -1)
                    cv2.line(canvas, (x , y), (x0, y0), color, brush_sizes[current_brush_size_index])

                else:
                    x0, y0 = a[-2]
                    cv2.circle(canvas, (x, y), 7, color, -1)
                    cv2.line(canvas, (x , y), (x0, y0), color, brush_sizes[current_brush_size_index])                    
                
        return frame, canvas
    
    
    def DrawBoxes(self, frame):
        global color
        
        cv2.rectangle(frame, (20, 50), (220, 150), (0, 0, 0), 4)
        cv2.putText(frame, 'Eraser', (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        
        cv2.rectangle(frame, (270, 50), (470, 150), (0, 255, 0), -1)
        cv2.putText(frame, 'Green', (320, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        cv2.rectangle(frame, (520, 50), (720, 150), (255, 0, 0), -1)        
        cv2.putText(frame, 'Red', (590, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        cv2.rectangle(frame, (770, 50), (970, 150), (0, 0, 255), -1)        
        cv2.putText(frame, 'Blue', (830, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        cv2.rectangle(frame, (1050, 50), (1250, 150), (0, 0, 0), -1)        
        cv2.putText(frame, 'Clear', (1100, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        if color == (0, 0, 255):
            c = 'Red'
            cv2.rectangle(frame, (518, 45), (718, 155), (255, 0, 255), 8)        

        elif color == (0, 255, 0):
            c = 'Green'
            cv2.rectangle(frame, (268, 45), (468, 155), (255, 0, 255), 8)        

        elif color == (255, 0, 0):
            c = 'Blue'
            cv2.rectangle(frame, (768, 45), (968, 155), (255, 0, 255), 8)        

        elif color == (255, 255, 255):
            c = 'White'
            # cv2.rectangle(frame, (950, 50), (1150, 170), (255, 0, 255)), 6)        
        
        elif color == (0, 0, 0):
            c = 'Eraser'
            cv2.rectangle(frame, (15, 45), (225, 155), (255, 0, 255), 8) 
                   
        if c == 'Clear':
            cv2.rectangle(frame, (1040, 1255), (220, 155), (255, 0, 255), 8) 
            cv2.putText(frame, 'Clear Selected', (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        else:
            cv2.putText(frame, 'Color Selected: ', (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
            cv2.putText(frame, c, (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color[::-1], 3)
        
        # Draw buttons for brush sizes
        cv2.rectangle(frame, (20, 210), (220, 250), (0, 0, 0), -1)
        cv2.putText(frame, 'Brush Size', (55, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (20, 280), (220, 320), (0, 0, 0), -1)
        cv2.putText(frame, '15', (100, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (20, 330), (220, 370), (0, 0, 0), -1)
        cv2.putText(frame, '30', (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (20, 380), (220, 420), (0, 0, 0), -1)
        cv2.putText(frame, '45', (100, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(frame, (20, 430), (220, 470), (0, 0, 0), -1)
        cv2.putText(frame, '60', (100, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame    
    
    
    def Landmarks(self, frame, canvas, draw=True):
        global a, b, color, current_brush_size_index

        self.results = self.hands.process(frame)

        frame = self.DrawBoxes(frame)

        if self.results.multi_hand_landmarks: 
            h, w = frame.shape[0], frame.shape[1]    
            x, y = int(self.results.multi_hand_landmarks[0].landmark[8].x*w), int(self.results.multi_hand_landmarks[0].landmark[8].y*h)

            x2, x4 = self.results.multi_hand_landmarks[0].landmark[2].x*w, self.results.multi_hand_landmarks[0].landmark[4].x*w
            if x4 > x2:
                b[0] = 0
            elif x4 < x2:
                b[0] = 1    

            # Co-ordinates of Index finger:
            y6, y8 = self.results.multi_hand_landmarks[0].landmark[6].y*h, self.results.multi_hand_landmarks[0].landmark[8].y*h
            if y8 < y6:
                b[1] = 1
            elif y8 > y6:
                b[1] = 0
                
            # Check if index finger touches the size buttons
            if ((20 < x < 220) and (280 < y < 320)):
                current_brush_size_index = 0
            elif ((20 < x < 220) and (330 < y < 370)):
                current_brush_size_index = 1
            elif ((20 < x < 220) and (380 < y < 420)):
                current_brush_size_index = 2
            elif ((20 < x < 220) and (430 < y < 470)):
                current_brush_size_index = 3
            elif ((1050 < x < 1250) and (50 < y < 150)):
                canvas = np.zeros_like(frame)
                x, y = 0, 0
                a = deque([[x, y], [x, y]])
                color = (255, 255, 255)
                current_brush_size_index = 0
                cv2.rectangle(frame, (250, 550), (1100, 650), (255, 0, 255), -1)        
                cv2.putText(frame, 'Screen Cleared', (300, 630), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10)
           

            frame, canvas = self.DrawOnScreen(frame, canvas, x, y)

            if draw:
                frame = self.DrawHandsLandmarks(frame)

        return frame, canvas
    


                
def main():
    global count_frames, current_brush_size_index

    s = 0
    t = 0
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    video = cv2.VideoCapture(0)  
    video.set(3, 1280)  # width
    video.set(4, 720)   # height
    
    hands = HandsDetection(num_hands=1, detection_confidence=0.8, 
                        static_mode=False, tracking_confidence=0.8,)
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frame, canvas = hands.Landmarks(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB), canvas, draw=False)

        s = time.time()
        fps = int(1/(s-t))
        t = s
            
        cv2.putText(frame, 'FPS: ' + str(fps), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
        
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        canvas_inv = cv2.cvtColor(canvas_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), canvas_inv)
        output = cv2.bitwise_or(frame, canvas)
        
        cv2.imshow('output', output)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
        
        count_frames += 1
        
    video.release() 
    
    cv2.destroyAllWindows()
    print("Done processing video")
    
    return None



if __name__ == '__main__':
    main()
