import cv2 as cv





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
    # Create window
    cv.namedWindow(window_name)
    # Show an image in the window
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
        
        cv.imshow(window_name, frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()