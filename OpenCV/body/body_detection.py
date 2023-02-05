import cv2
import numpy as np
import sys;
# load the COCO class names
with open('COCO_labels.txt', 'r') as f:
    class_names = f.read().split('\n')
# get a different color array for each of the classes
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
#Nous allons charger le MobileNet modèle SSD à l'aide de la fonction readNet() 
model = cv2.dnn.readNet(model='frozen_inference_graph_V2.pb',
                        config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')
#Ensuite, nous allons lire l'image à partir du disque et de préparer l'entrée de fichier blob(pretraitement).
# read the image from disk
image = cv2.imread(sys.argv[1])
image_height, image_width, _ = image.shape
# create blob from image
# We specify the size to be 300×300 as this the input size that SSD models generally expect in almost all frameworks. 
# We are using the swapRB because the models generally expect the input to be in RGB format.
blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
# set the input to the pre-trained deep learning network and obtain
# the output predicted probabilities for label
# classes
model.setInput(blob)
output = model.forward()
nombre = 0; 

# loop over each of the detection
for detection in output[0, 0, :, :]:
    # extract the confidence of the detection
    confidence = detection[2]
    # draw bounding boxes only if the detection confidence is above a certain threshold, else skip
    if confidence > .6 :
        # get the class id
        class_id = detection[1]
        # map the class id to the class
        class_name = class_names[int(class_id)-1]
        #On detecte que les personnes
        if class_name != "person" :
            continue
        #print(class_name)
        color = COLORS[int(class_id)]
        # get the bounding box coordinates
        box_x = detection[3] * image_width
        box_y = detection[4] * image_height
        # get the bounding box width and height
        box_width = detection[5] * image_width
        box_height = detection[6] * image_height
        # draw a rectangle around each detected object
        cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (240, 50, 50), thickness=3)
        # put the FPS text on top of the frame
        # cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # Nombre de personnes
        nombre += 1   
cv2.putText(image, f"Nombre person: {nombre}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (240, 50, 50), 3)
#cv2.imshow('image', image)
cv2.imwrite('../predicted_age.jpg', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

