import pytesseract
import cv2
import imutils
import numpy as np 
import random

# tunning licensce
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]

char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ'

net = cv2.dnn.readNet("yolo.weights", "yolo.cfg")

# Name custom object
classes = ["License-Plate"]
# image path
images_path = ["demo.jpg"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# insert the path of images
random.shuffle(images_path)

# loop through all the images
for img_path in images_path:
    # loading image
    img = cv2.imread(img_path)
    img_tmp = img
    cv2.imshow("demo", img)
    cv2.waitKey(0)
    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # rectangle corrdinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            #cv2.putText(img, label, (x, y+30), font, 3, color, 3)

    license_cropped = img[y:(y+h), x:(x+w), :]

    # convert rgb -> gray
    license_cropped = cv2.cvtColor(license_cropped, cv2.COLOR_BGR2GRAY)
    # # blur
    # license_cropped = cv2.GaussianBlur(license_cropped, (3,3), 0)

    cv2.imshow("license Plate", license_cropped)
    cv2.waitKey(0)

    text = pytesseract.image_to_string(license_cropped, lang='eng', config='--psm 6')
    print("text: ", text)
    cv2.putText(img, text,(50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), lineType=cv2.LINE_AA)
    cv2.imshow("OCR", img)
    cv2.waitKey(0)