import cv2 as cv
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EVENT_MOUSEMOVE = 0
EVENT_LBUTTONDOWN = 1
EVENT_RBUTTONDOWN = 2
EVENT_LBUTTONUP = 4
EVENT_RBUTTONUP = 5
EVENT_LBUTTONDBLCLK = 7


class DrawHandler:
    def __init__(self):
        self.isHolding = False
        self.currentPin = None
        self.drawnPins = []
        self.current_frame = None
        
        
    def getPins(self):
        return self.drawnPins
    
    
    def setFrame(self, frame):
        self.current_frame = frame.copy()
    
    def startPin(self, x, y, flags, _):
        if not self.isHolding:
            self.isHolding = True
            self.currentPin = PinPoint(x, y)
            print(f"Pin Start: {x}, {y}")
            
    def endPin(self, x, y, flags, _):
        if self.isHolding:
            self.isHolding = False
            self.currentPin.setEnd(x, y)
            print(f"Pin End: {x}, {y} | width: {self.currentPin.width}, height: {self.currentPin.height}")
            self.appendPin(self.currentPin)
            self.currentPin = None
            print(self.drawnPins)
            
    def mouseCallback(self, event, x, y, flags, _):
        if event == EVENT_LBUTTONDOWN:
            self.startPin(x, y, flags, _)
        elif event == EVENT_LBUTTONUP:
            self.endPin(x, y, flags, _)
            
    def appendPin(self, pin):
        self.drawnPins.append(pin)
        
    def savePins(self, filename):
        if self.current_frame is None:
            print("No frame to save mask on.")
            return
        
        mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
        for pin in self.drawnPins:
            x1, y1 = pin.x, pin.y
            x2, y2 = pin.x + pin.width, pin.y + pin.height
            cv.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # Fill rectangle with white
        
        cv.imwrite(filename, mask)
        print(f"Mask saved to {filename}")


class PinPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = None
        self.height = None
    
    def setEnd(self, x, y):
        self.width = x - self.x
        self.height = y - self.y
        
window_name = "myWindow"
drawHandler = DrawHandler()

def sliderCallback(num):
    print(f"My Custom Callback {num}")




CAMERA_SRC= os.getenv("CAMERA_SRC")


if (__name__ == "__main__"):
    
    cv.namedWindow(window_name)
    
    cap = cv.VideoCapture(CAMERA_SRC)
    
    # First, buffer the entire video into memory
    frame_buffer = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
    cap.release()
    
    if len(frame_buffer) == 0:
        print("No frames read from video source.")
        exit()
    
    direction = 1  # 1 = forward, -1 = backward
    buffer_index = 0
    
    SLOWDOWN_FACTOR = 2  # 2 = 50% speed, 3 = 33% speed, etc.
    repeat_count = 0

    cv.createTrackbar("slider_name", window_name, 0, 100, sliderCallback)
    cv.setMouseCallback(window_name, drawHandler.mouseCallback)
    
    while True:
        if repeat_count == 0:
            buffer_index += direction
            
            if buffer_index >= len(frame_buffer):
                direction = -1
                buffer_index = len(frame_buffer) - 1
            elif buffer_index < 0:
                direction = 1
                buffer_index = 0
        
        repeat_count = (repeat_count + 1) % SLOWDOWN_FACTOR
        frame = frame_buffer[buffer_index].copy()
        
        drawHandler.setFrame(frame)
        
        for pin in drawHandler.getPins():
            pt1 = (pin.x, pin.y)
            pt2 = (pin.x + pin.width, pin.y + pin.height)
            cv.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        
        cv.putText(frame, "Press 'q' to quit, 'z' to undo, 'c' to clear, 's' to save mask", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.imshow(window_name, frame)
        
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            drawHandler.drawnPins = drawHandler.drawnPins[:-1]
        elif key == ord('c'):
            drawHandler.drawnPins = []
        elif key == ord('s'):
            drawHandler.savePins("mask.jpg")
            break
    
    cv.destroyAllWindows()