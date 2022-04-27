import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np


model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

print('Starting...')

def loop_through_people(frame, keypoints_with_scores, confidence_threshold, bounding_boxes):
    for i, person in enumerate(keypoints_with_scores):
        # Check if arm is raised    1
        if person[6][0] > person[8][0] and person[8][0] > person[10][0]:
            draw_box(frame,bounding_boxes[i],confidence_threshold)


def draw_box(frame,bounding_boxes,confidence_threshold):
    y, x, confidence = frame.shape
    shaped = np.squeeze(np.multiply(bounding_boxes[0], [y,x,y,x,1]))
    ymin, xmin, ymax, xmax, confidence = shaped

    if confidence > confidence_threshold:
        print("Confidence", confidence)
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 3)


capture = cv2.VideoCapture(0)   #index depending on number of camera chanels
while capture.isOpened():
    _, frame = capture.read()

    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)  #reshape mult if 32
    input_img = tf.cast(img, dtype=tf.int32)  #transformdatatype

    # Detection section
    results = movenet(input_img)

    # Take the prediction bounding box coordinates from the results, can detect up to 6 people.
    bounding_boxes = results['output_0'].numpy()[:,:,51:56].reshape((6,1,5))


    # Take the person keypoints and confidence from the output
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3)) #increase 51 for boundaries

    # Render keypoints
    loop_through_people(frame, keypoints_with_scores, 0.45, bounding_boxes)


    cv2.imshow('Gesture detection', frame)

    if cv2.waitKey(10) & 0xFF==ord('d'):
        capture.release()
        cv2.destroyAllWindows()
        break
capture.release()
cv2.destroyAllWindows()
