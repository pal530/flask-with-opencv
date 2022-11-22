# -*- coding: utf-8 -*-

# import cv2
# from flask import Flask, render_template, Response
# import cv2
import os
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import HandTrackingModule as htm
import time
import random
import time
import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()   # create background subtractor

def checkWinner(player, comp):
    if player == comp:
        return None
    elif player == 'rock' and comp == 'scissor':
        return 0
    elif player == 'paper' and comp == 'rock':
        return 0
    elif player == 'scissor' and comp == 'paper':
        return 0
    else:
        return 1
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('home.html')

#code of rock paper scisssor
def gen():
    """Video streaming generator function."""
    # VARIABLES

    moves = ['rock', 'paper', 'scissor']
    scores = [0, 0]  # [player, comp]
    comp, player = None, None
    wCam, hCam = 1280, 720

    # Get feed
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    # Get detector
    detector = htm.handDetector(detectionCon=0.8, maxHands=1)

    # Time variables for fps and time limit
    waitTime = 4
    pTime = 0
    prevTime = time.time()
    newTime = time.time()

    # Read until video is completed
    while (cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret:  # if vid finish repeat
            frame = cv2.VideoCapture(0)
            continue
        if ret:  # if there is a frame continue with code
            img = cv2.flip(frame, 1)
            cv2.line(img, (wCam // 2, 0), (wCam // 2, hCam), (0, 255, 0), 5)
            cv2.rectangle(img, (780, 160), (1180, 560), (0, 0, 255), 2)
            cv2.putText(img, f'{scores[1]}', (320, 640), cv2.FONT_HERSHEY_PLAIN, 5,
                        (0, 250, 0), 3)
            cv2.putText(img, f'{scores[0]}', (960, 640), cv2.FONT_HERSHEY_PLAIN, 5,
                        (0, 250, 0), 3)
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if waitTime - int(newTime) + int(prevTime) < 0:
                cv2.putText(img, '0', (960, 120), cv2.FONT_HERSHEY_PLAIN, 7,
                            (0, 0, 255), 3)
            else:
                cv2.putText(img, f'{waitTime - int(newTime) + int(prevTime)}', (960, 120), cv2.FONT_HERSHEY_PLAIN, 7,(0, 0, 255), 3)
            if len(lmList) != 0:

                if newTime - prevTime >= waitTime:

                    x, y = lmList[0][1:]

                    if 780 < x < 1180 and 160 < y < 560:
                        fingers = detector.fingersUp()
                        totalFingers = fingers.count(1)

                            # Game logic
                        if totalFingers == 0:
                            player = 'rock'
                        elif totalFingers == 2:
                            player = 'scissor'
                        elif totalFingers == 5:
                            player = 'paper'

                        comp = moves[random.randint(0, 2)]

                        winner = checkWinner(player, comp)
                        if winner is not None:
                            scores[winner] = scores[winner] + 1

                        prevTime = time.time()

                # Show computer move
                if comp:
                    img[160:560, 120:520] = cv2.imread(f'Fingers/{comp}.jpg')

                # Show fps
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv2.putText(img, f'FPS: {int(fps)}', (1050, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                newTime = time.time()
                cv2.waitKey(1)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   # time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
           break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img12 tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# code of pedestrians detection using opencv
def gene():
    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)

    # Read until video is completed
    while (cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret:  # if vid finish repeat
            frame = cv2.VideoCapture(0)
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1.5, 1.5)  # resize image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            fgmask = sub.apply(gray)  # uses the background subtraction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minarea = 400
            maxarea = 50000
            for i in range(len(contours)):  # cycles through all contours in current frame
                if hierarchy[
                    0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                    area = cv2.contourArea(contours[i])  # area of contour
                    if minarea < area < maxarea:  # area threshold for contour
                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)
                        # creates a rectangle around contour
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Prints centroid text in order to double check later on
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, .3,
                                    (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,
                                       line_type=cv2.LINE_8)
        # cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
            break


@app.route('/vidfeed')
def vidfeed():
    """Video streaming route. Put this in the src attribute of an img12 tag."""
    return Response(gene(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



#code of kids game for play
def gener():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = FaceMeshDetector(maxFaces=1)
    idList = [0, 17, 78, 292] 


    folderEatable = 'Objects/eatable'
    listEatable = os.listdir(folderEatable)
    eatables = []
    for object in listEatable:
        eatables.append(cv2.imread(f'{folderEatable}/{object}', cv2.IMREAD_UNCHANGED))

    folderNonEatable = 'Objects/noneatable'
    listNonEatable = os.listdir(folderNonEatable)
    nonEatables = []
    for object in listNonEatable:
        nonEatables.append(cv2.imread(f'{folderNonEatable}/{object}', cv2.IMREAD_UNCHANGED))

    currentObject = eatables[0]
    pos = [300, 0]
    speed = 5
    count = 0
    global isEatable
    isEatable = True
    gameOver = False

    def resetObject():
        global isEatable
        pos[0] = random.randint(100, 1180)
        pos[1] = 0
        randNo = random.randint(0, 2)  # change the ratio of eatables/ non-eatables
        if randNo == 0:
            currentObject = nonEatables[random.randint(0, 3)]
            isEatable = False
        else:
            currentObject = eatables[random.randint(0, 3)]
            isEatable = True

        return currentObject


    while (cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret:  # if vid finish repeat
            frame = cv2.VideoCapture(0)
            continue
        if ret:
            img = cv2.flip(frame, 1)
            if gameOver is False:
                img, faces = detector.findFaceMesh(img, draw=False)

                img = cvzone.overlayPNG(img, currentObject, pos)
                pos[1] += speed

                if pos[1] > 520:
                    currentObject = resetObject()

                if faces:
                    face = faces[0]
                    # for idNo,point in enumerate(face):
                    #     cv2.putText(img,str(idNo),point,cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),1)

                    up = face[idList[0]]
                    down = face[idList[1]]

                    for id in idList:
                        cv2.circle(img, face[id], 5, (255, 0, 255), 5)
                    cv2.line(img, up, down, (0, 255, 0), 3)
                    cv2.line(img, face[idList[2]], face[idList[3]], (0, 255, 0), 3)

                    upDown, _ = detector.findDistance(face[idList[0]], face[idList[1]])
                    leftRight, _ = detector.findDistance(face[idList[2]], face[idList[3]])

                    ## Distance of the Object
                    cx, cy = (up[0] + down[0]) // 2, (up[1] + down[1]) // 2
                    cv2.line(img, (cx, cy), (pos[0] + 50, pos[1] + 50), (0, 255, 0), 3)
                    distMouthObject, _ = detector.findDistance((cx, cy), (pos[0] + 50, pos[1] + 50))
                    print(distMouthObject)

                    # Lip opened or closed
                    ratio = int((upDown / leftRight) * 100)
                    # print(ratio)
                    if ratio > 60:
                        mouthStatus = "Open"
                    else:
                        mouthStatus = "Closed"
                    cv2.putText(img, mouthStatus, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

                    if distMouthObject < 100 and ratio > 60:
                        if isEatable:
                            currentObject = resetObject()
                            count += 1
                        else:
                            gameOver = True
                cv2.putText(img, str(count), (1100, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 5)
            else:
                cv2.putText(img, "Game Over", (300, 400), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 10)

            cv2.imshow("Image", img)
            key = cv2.waitKey(1)

            if key == ord('r'):
                resetObject()
                gameOver = False
                count = 0
                currentObject = eatables[0]
                isEatable = True
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
            break



@app.route('/viderfeed')
def viderfeed():
    """Video streaming route. Put this in the src attribute of an img12 tag."""
    return Response(gener(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

    

