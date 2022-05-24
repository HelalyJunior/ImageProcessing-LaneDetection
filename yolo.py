import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

def yolo_init():
    # defining the paths for the configuartion and weights file for the network
    weights_path = "yolov3.weights"
    config_path = "yolov3.cfg"

    # reading the network
    net = cv2.dnn.readNetFromDarknet(config_path,weights_path)

    #getting all the 106 layer names
    names = net.getLayerNames()

    # getting the names of the output layers
    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net,layers_names
    
def run_yolo(net,layers_names,img):
    
    (H,W) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0,(416,416),crop = False,swapRB=False)
    net.setInput(blob)
    start_t = time.time()
    layers_output = net.forward(layers_names)
    print("A forward pass through yolov3 took {}".format(time.time() - start_t))
    boxes =[]
    confidences = []
    classIDs = []
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID =  np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.85:
                box = detection[:4] * np.array([W,H,W,H])
                bx , by, bw, bh = box.astype("int")
                x = int(bx - (bw / 2) )
                y = int(by - (bh / 2) )
                
                
                boxes.append([x,y,int(bw),int(bh)])
                confidences.append(confidence)
                classIDs.append(classID)
                
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,0.8,0.7)
    labels_path = "coco.names"
    labels = open(labels_path).read().strip().split("\n")

    if len(idxs) > 0:
        
        for i in idxs.flatten():
            (x,y) = [boxes[i][0], boxes[i][1]]
            (w,h) = [boxes[i][2], boxes[i][3]]
        
        
            cv2.rectangle(img, (x, y-20), (x + 80, y), (0, 205, 255), -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 205, 255), 3)
            cv2.putText(img,"{}: {}".format(labels[classIDs[i]],round(float(confidences[i]),2)), ( x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, \
                                0.5, (0, 0, 0), 2)
    