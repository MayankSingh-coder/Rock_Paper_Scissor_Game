import numpy as np
import cv2
import random
import math

array = np.array(['p.png','r.png','s.png'])
test_list = [0,1,2]


player =0
comp =0

# Opening Camera
capture = cv2.VideoCapture(1)
prev=-1
while capture.isOpened():

    # taking all frames from cam
    ret, frame = capture.read()
    if not ret:
        continue
    # taking all data of hand from rectangle window
    cv2.rectangle(frame, (50, 350), (50, 350), (0, 255, 0), 0)
    crop_image = frame[50:350, 50:350]

    # Applying Gaussian blur
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    # Show threshold image

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(crop_image.shape, np.uint8)
    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv2.convexHull(contour)
        print(hull)
        # Draw contour

        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Find convexity defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                print(angle)
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)
        random_num = random.choice(test_list)
        h = int(str(random_num))
        img = cv2.imread(array[h], 0)
        img = cv2.resize(img, (120, 120))


        # Print number of fingersq
        #capture.release()
        if prev!=count_defects:
            if count_defects == 0:
                cv2.putText(frame, "ROCK", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
               # cv2.waitKey(0)

                if h==1:
                    cv2.putText(frame, "Draw", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                elif h==2:
                    player = player + 1
                    cv2.putText(frame, "player won ", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                elif h==0:
                    comp = comp + 1
                    cv2.putText(frame, "Computer won", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
               # cv2.waitKey(0)
                #capture = cv2.VideoCapture(0)
            elif count_defects == 1:
                cv2.putText(frame, "SCISSOR", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

                if h==2:
                    cv2.putText(frame, "Draw", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                elif h==0:
                    player = player + 1
                    cv2.putText(frame, "player won ", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                elif h==1:
                    comp = comp + 1
                    cv2.putText(frame, "Computer won", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                #capture = cv2.VideoCapture(0)
               # cv2.waitKey(0)


            elif count_defects >1:
                cv2.putText(frame, "PAPER", (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)


                if h==0:
                    cv2.putText(frame, "Draw", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                elif h==1:
                    player = player + 1
                    cv2.putText(frame, "player won ", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                elif h==2:
                    comp = comp + 1
                    cv2.putText(frame, "Computer won", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                #capture = cv2.VideoCapture(0)
                #cv2.waitKey(0)
            cv2.imshow('img', img)
            prev=count_defects
        else:
            pass

    except:
        pass

# Show required images
    cv2.imshow("Gesture", frame)
    all_image = np.hstack((drawing, crop_image))


# Close the camera if 'q' is pressed
    if cv2.waitKey(1) & 0xFF== ord('q'):
        break



    #img = cv2.imread('blank.png', 0)
    #img = cv2.resize(img, (540, 540))
    #cv2.imshow('img', img)
if player > comp:
    cv2.putText(img, "Player won the game ", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
elif comp > player:
    cv2.putText(img, "Computer won the game ", (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

capture.release()
cv2.destroyAllWindows()
