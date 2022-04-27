import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np


'''
# Optional if you are using a GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)'''

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

print('test')

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold,bounding_boxes):
    i = 0
    for person in keypoints_with_scores:
        #draw_connections(frame, person, edges, confidence_threshold)
        #draw_keypoints(frame, person, confidence_threshold)
        if person[6][0] > person[8][0] and person[8][0] > person[10][0]:
            draw_box(frame,bounding_boxes[i],confidence_threshold)
        i += 1


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_box(frame,bounding_boxes,confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(bounding_boxes[0], [y,x,y,x,1]))
    ymin, xmin, ymax, xmax, c = shaped
    #print(c)



    if c > confidence_threshold:
        print(c)
        print(int(xmin),int(ymin))
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 3)

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))


    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]



        if (c1 > confidence_threshold) & (c2 > confidence_threshold):

            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

cap = cv2.VideoCapture(0)   #index depending on number of camera chanels
while cap.isOpened():
    ret, frame = cap.read()

    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)  #reshape mult if 32
    input_img = tf.cast(img, dtype=tf.int32)  #transformdatatype

    # Detection section
    results = movenet(input_img)

    bounding_boxes = results['output_0'].numpy()[:,:,51:56].reshape((6,1,5))



    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3)) #increase 51 for boundaries

    # Render keypoints
    loop_through_people(frame, keypoints_with_scores, EDGES, 0.45,bounding_boxes)


    cv2.imshow('Gesture detection', frame)

    if cv2.waitKey(10) & 0xFF==ord('d'):
        cap.release()
        cv2.destroyAllWindows()
        break
cap.release()
cv2.destroyAllWindows()
