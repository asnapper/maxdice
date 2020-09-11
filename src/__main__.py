import numpy as np
import cv2
import imutils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(1280))
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(720))

cap.set(cv2.CAP_PROP_EXPOSURE, -7.5)
cap.set(cv2.CAP_PROP_SATURATION, 200)

def isRectangle(contour):
    peri = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.009 * peri, True)
    sides = len(vertices)
    return sides == 4

def getPureRectangle(contour):
    peri = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.009 * peri, True)

lastBluePlayArea = []


def getBluePlayArea(frame):
    global lastBluePlayArea
    contrast = cv2.convertScaleAbs(frame, alpha=2, beta=1)
    gray = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 80, 255)  # Determine edges of objects in an image
    (contours, _) = cv2.findContours(edged, cv2.RETR_CCOMP,
                                     cv2.CHAIN_APPROX_NONE)  # Find contours in an image
    largestRectangle = None
    for cnt in contours:
        if isRectangle(cnt):
            if (largestRectangle is None) or (cnt.size > largestRectangle.size):
                largestRectangle = cnt
    if largestRectangle is not None:
        lastBluePlayArea = largestRectangle
        return [largestRectangle]
    elif lastplayArea is not None:
        return [lastBluePlayArea]
    else:
        return []

lastplayArea = None
lastRect = None
def getRedPlayArea(frame):
    global lastplayArea
    global lastRect

    if lastplayArea is not None:
        return [lastplayArea], lastRect
    
    hsv = cv2.convertScaleAbs(frame, alpha=1.2, beta=1.1)
    hsv = cv2.GaussianBlur(hsv, (9, 9), 2, borderType=cv2.BORDER_DEFAULT)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))

    mask = mask1 | mask2
    mask = cv2.bitwise_not(mask)

    edged = cv2.Canny(mask, 80, 255)  # Determine edges of objects in an image
    (contours, _) = cv2.findContours(edged,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    playArea = None
    rect = None
    for cnt in contours:
        if isRectangle(cnt):
            if (lastplayArea is None) or (cnt.size > lastplayArea.size):
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                if (box.max() - box.min()) > 900:
                    lastRect = rect
                    lastplayArea = box
    if playArea is not None:
        lastplayArea = playArea
        return [playArea], rect
    elif lastplayArea is not None:
        return [lastplayArea], lastRect
    else:
        return [], None

def cropArea(frame, area):
    topleft = area[0]
    bottomright = area[0]
    for shape in area:
        if shape[0] > topx:
            topx =  shape[0]
        if shape[1] < bottomy:
            bottomy =  shape[0]

while(True):
    ret, frame = cap.read()
    (redAreaCountours, rect) = getRedPlayArea(frame)
    box = redAreaCountours[0]
    if rect is not None:
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(frame, M, (width, height))

        cv2.imshow('warped', warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
