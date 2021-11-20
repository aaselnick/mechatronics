import cv2 as cv
import numpy as np
import serial
import imutils
import time
from random import seed
from imutils.video import VideoStream

# initialize serial port
j=0
seed(1)

ser = serial.Serial('/dev/ttyUSB0', 38400, timeout=1)
ser.flush()

# grab camera matrix and distortion coefficients from text files
#mtx = np.loadtxt('cam_matrix')
#dist = np.loadtxt('dist_coeff')
#newcameramtx = np.loadtxt('newcameramtx')

# kernel for image operations
kernel = np.ones((5,5),np.uint8)

# distortion parameters
mtx = []
with open('cam_matrix.txt') as f:
    for i in range(1,4):
        line = f.readline()
        x = line.split()
        mtxline = []
        for h in x:
            mtxline.append(np.float32(h.replace("\n","")))
        mtx.append(mtxline)

dist = []
with open('dist_coeff.txt') as f:
    line = f.readline()
    x = line.split()
    distline = []
    for h in x:
        distline.append(np.float32(h.replace("\n","")))
    dist.append(distline)

newcameramtx = []
with open('newcameramtx.txt') as f:
    for i in range(1,4):
        line = f.readline()
        x = line.split()
        newcameramtxline = []
        for h in x:
            newcameramtxline.append(np.float32(h.replace("\n","")))
        newcameramtx.append(newcameramtxline)

roi = []
with open('roi.txt') as f:
    for i in range(1,5):
        line = f.readline()
        roi.append(int(float((line.replace("\n","")))))

# Video capture
cap = VideoStream(src=0).start()

# allow the camera to warm up
time.sleep(2.0)

####################################################################################################### MAIN LOOP
########################################################################################################

print("Starting loop...")

while True:
    e1 = cv.getTickCount()

    # Capture frame-by-frame
    frame = cap.read()

    # undistort
    frame = cv.undistort(frame,np.float32(mtx),np.float32(dist),None,np.float32(newcameramtx))
    
    # crop the image
    x,y,w,h = roi
    frame=frame[y:y+h,x:x+w]
    
    # resize frame, blur it, and convert to HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv.GaussianBlur(frame, (11,11),0)
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # green inRange
    lowerG = np.array([48,176,16])
    upperG = np.array([88, 255, 158])

    # yellow inRange
    lowerY = np.array([0,44,71])
    upperY = np.array([43,255,225])

    # blue inRange
    lowerB = np.array([101,142,7])
    upperB = np.array([160,255,149])
    
    # Threshold HSV images to get only desired colors, then perform a series of dilations and erosions
    # to remove any small blobs left in the mask
    green_mask = cv.inRange(hsv, lowerG, upperG)
    green_mask = cv.erode(green_mask, None, iterations=2)
    green_mask = cv.dilate(green_mask, None, iterations=2)

    yellow_mask = cv.inRange(hsv,lowerY,upperY)
    yellow_mask = cv.erode(yellow_mask, None, iterations=2)
    yellow_mask = cv.dilate(yellow_mask, None, iterations=2)

    blue_mask = cv.inRange(hsv,lowerB,upperB)
    blue_mask = cv.erode(blue_mask, None, iterations=2)
    blue_mask = cv.dilate(blue_mask, None, iterations=2)
    
    # find contours in the mask and initialize the current (x, y) center of the ball
    green_cnts = cv.findContours(green_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    green_cnts = imutils.grab_contours(green_cnts)
    green_center = None

    yellow_cnts = cv.findContours(yellow_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    yellow_cnts = imutils.grab_contours(yellow_cnts)
    yellow_center = None

    blue_cnts = cv.findContours(blue_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blue_cnts = imutils.grab_contours(blue_cnts)
    blue_center = None
    
    yellow_frame = frame.copy()
    green_frame = frame.copy()
    blue_frame = frame.copy()

    # only proceed if at least one contour was found
    if len(green_cnts) > 0:
        gg = max(green_cnts, key=cv.contourArea)
        ((xG,yG), radiusG) = cv.minEnclosingCircle(gg)
        MG = cv.moments(gg)
        centerG = (int(MG["m10"]/MG["m00"]), int(MG["m01"]/MG["m00"]))

        # only proceed if the radius meets a minimum size
        if radiusG > 0:
            cv.circle(frame, (int(xG), int(yG)), int(radiusG),
                    (0,255,255),2)
            cv.circle(frame, centerG, 5, (0,0,255), -1)

    # only proceed if at least one contour was found
    if len(yellow_cnts) > 0:
        yy = max(yellow_cnts, key=cv.contourArea)
        ((xY,yY), radiusY) = cv.minEnclosingCircle(yy)
        MY = cv.moments(yy)
        centerY = (int(MY["m10"]/MY["m00"]), int(MY["m01"]/MY["m00"]))

        # only proceed if the radius meets a minimum size
        if radiusY > 10:
            cv.circle(frame, (int(xY), int(yY)), int(radiusY),
                    (0,255,255),2)
            cv.circle(frame, centerY, 5, (0,0,255),-1)

    # only proceed if at least one contour was found
    if len(blue_cnts) > 0:
        bb = max(blue_cnts, key=cv.contourArea)
        ((xB,yB), radiusB) = cv.minEnclosingCircle(bb)
        MB = cv.moments(bb)
        centerB = (int(MB["m10"]/MB["m00"]), int(MB["m01"]/MB["m00"]))

        # only proceed if the radius meets a minimum size
        if radiusB > 0:
            cv.circle(frame, (int(xB), int(yB)), int(radiusB),
                    (0,255,255),2)
            cv.circle(frame, centerB, 5, (0,0,255),-1)


    # show the frame to our screen
    #cv.imshow("green_mask", green_mask)
    #cv.imshow("yellow_mask", yellow_mask)
    #cv.imshow("blue_mask", blue_mask)
    
    #cv.imshow("green_frame", green_frame)
    #cv.imshow("yellow_frame", yellow_frame)
    #cv.imshow("blue_frame", blue_frame)
    
    cv.imshow("frame", frame)
    
    try:
        xG = int(xG)
        yG = int(yG)
        xY = int(xY)
        yY = int(yY)
        xB = int(xB)
        yB = int(yB)
    except:
        pass


    try:
        print('Green: (' +str(xG) + ' , ' + str(yG) + ')') 
    except:
        pass

    try:
        print('Yellow: (' + str(xY) + ' , ' + str(yY) + ')')
    except:
        pass

    try: 
        print('Blue: (' + str(xB) + ' , ' + str(yB) + ')')
    except:
        pass

    ################################ arduino communication for radio broadcast 
        
    # Green
    try:
        data=bytes("TX#:", 'utf-8')
        ser.write(data)
        ser.write(str(j).encode('utf-8'))
        data=bytes(" G<", 'utf-8')
        ser.write(data)
        ser.write(str(xG).encode('utf-8'))
        data=bytes(", ", 'utf-8')
        ser.write(data)
        ser.write(str(yG).encode('utf-8'))
        data=bytes(">",'utf-8')
        ser.write(data)
        ser.write("\n".encode('utf-8'))
        time.sleep(0.04)
        ser.flush()
    except:
        pass

    # Blue
    try:
        data=bytes("B<", 'utf-8')
        ser.write(data)
        ser.write(str(xB).encode('utf-8'))
        data=bytes(", ", 'utf-8')
        ser.write(data)
        ser.write(str(yB).encode('utf-8'))
        data=bytes(">", 'utf-8')
        ser.write(data)
        time.sleep(0.04)
        ser.flush()
    except:
        pass

    # Yellow
    try:
        data=bytes(" Y<", 'utf-8')
        ser.write(data)
        ser.write(str(xY).encode('utf-8'))
        data=bytes(", ", 'utf-8')
        ser.write(data)
        ser.write(str(yY).encode('utf-8'))
        data=bytes(">", 'utf-8')
        ser.write(data)
        ser.write("\n".encode('utf-8'))
        time.sleep(0.04)
        ser.flush()
        line = ser.readline().decode('utf-8','ignore').rstrip()
        print(line)
    except:
        pass
    
    

    if j<= 98:
        j=j+1
    else:
        j=0
    

    ################################################################################# Performance metrics
    e2 = cv.getTickCount()
    fps1 = (e2-e1)/ cv.getTickFrequency()
    fps2 = 1/fps1
    print(fps2)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

