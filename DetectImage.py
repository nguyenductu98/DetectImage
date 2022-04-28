import cv2
import numpy as np

net = cv2.dnn.readNet('/media/tund/Data/Rang/yolov4_final.weights', '/media/tund/Data/Rang/yolov4.cfg')
classes = []
with open('obj.names', 'r') as f:
    classes = f.read().splitlines()

count=1


path = "/media/tund/Data/Rang/8.jpg"
img = cv2.imread(path)
height, width, _= img.shape

blob = cv2.dnn.blobFromImage(img, 1/255, (512,512), (0,0,0), swapRB=True, crop= False)
net.setInput(blob)
output_layers_names = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_names)

boxes = []
confidences = []
class_ids = [] 
k=0
for output in layerOutputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x -w/2)
            y = int (center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)
            # crop_img = img[y:y + h, x:x + w]
            # cv2.imwrite("/home/tund/test/that/b"+str(k)+".jpg",crop_img)
            # cv2.waitKey(100)
            # k=k+1


indexes = cv2.dnn.NMSBoxes(boxes,confidences, 0.5 , 0.4)

font = cv2.FONT_HERSHEY_SIMPLEX
colors = np.random.uniform(0,255, size=(len(boxes),3))

if len(indexes)>0:
    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w,y+h),color,2)
        cv2.putText(img,confidence, (x,y+20), font, 0.5, (255,255,255), 2)
        print("{}, {}".format(label, confidence))


cv2.imwrite("rang8.jpg",img)



cv2.destroyAllWindows()
