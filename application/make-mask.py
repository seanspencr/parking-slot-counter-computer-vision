import cv2 as cv
import numpy as np

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


if (__name__ == "__main__"):
    
    cv.namedWindow(window_name)
    cap = cv.VideoCapture("../parking_1920_1080_loop.mp4")
    
    # Add a slider
    cv.createTrackbar("slider_name", window_name, 0, 100, sliderCallback)
    cv.setMouseCallback(window_name, drawHandler.mouseCallback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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
        elif key == ord('z'):  # Undo last pin
            drawHandler.drawnPins = drawHandler.drawnPins[:-1]
        elif key == ord('c'):  # Clear pins
            drawHandler.drawnPins = []
        elif key == ord('s'):  # Save pins
            drawHandler.savePins("mask.jpg")
            break
    
    cap.release()
    cv.destroyAllWindows()