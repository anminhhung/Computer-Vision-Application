import cv2
import numpy as np
import glob
import random
import imutils
import OCR

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
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    print(classes[class_ids[0]])
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            #cv2.putText(img, label, (x, y+30), font, 3, color, 3)

    license_cropped = img[y:(y+h), x:(x+w), :]
    # license_cropped = imutils.resize(license_cropped, width=200)
    license_cropped = cv2.bilateralFilter(license_cropped, 11, 61, 40)
    cv2.imshow("License-Plate", license_cropped)
    cv2.waitKey(0)
    cv2.imwrite("license.jpg", license_cropped)
    LP = OCR.recog_LP("license.jpg")
    cv2.putText(img_tmp,LP,(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),1)
    detect_image = cv2.imread("detection.png")
    cv2.imshow("Segment", detect_image)
    cv2.waitKey(0)
    cv2.imshow("Image", img_tmp)
    cv2.waitKey(0)


cv2.destroyAllWindows