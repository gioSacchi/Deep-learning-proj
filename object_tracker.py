import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
tf.config.optimizer.set_jit(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet


flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')

def loop_through_people(frame, keypoints_with_scores, confidence_threshold, bounding_boxes):
    for i, person in enumerate(keypoints_with_scores):
        # Check if right arm is raised    1
        if person[6][0] > person[8][0] and person[8][0] > person[10][0]:
            if draw_box(frame,bounding_boxes[i],confidence_threshold):
                return bounding_boxes[i][0]
                


def draw_box(frame,bounding_boxes,confidence_threshold):
    y, x, confidence = frame.shape
    shaped = np.squeeze(np.multiply(bounding_boxes[0], [y,x,y,x,1]))
    ymin, xmin, ymax, xmax, confidence = shaped

    if confidence > confidence_threshold:
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,255,0), 3)
        return True

def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric, max_age=500)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = FLAGS.size
    video_path = FLAGS.video


    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # loading movenet model
    movenet_model = tf.saved_model.load("./checkpoints/movenet_multipose_lightning_1")
    movenet = movenet_model.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)


    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        movenet_frame = frame.copy()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        # print('Frame #: ', frame_num)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # start_time = time.time()

        movenet_frame = tf.image.resize_with_crop_or_pad(tf.expand_dims(movenet_frame, axis=0), 384, 640)
        movenet_frame = tf.cast(movenet_frame, dtype=tf.int32)
        movenet_results = movenet(movenet_frame)
        
        bounding_boxes = movenet_results['output_0'].numpy()[:,:,51:56].reshape((6,1,5))
        keypoints_with_scores = movenet_results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        reinit_bboxes = loop_through_people(frame, keypoints_with_scores, 0.45, bounding_boxes)

        batch_data = tf.constant(image_data)
        # detection_time = time.time()
        pred_bbox = infer(batch_data)
        # print("Detection: ", (time.time() - detection_time))
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        if reinit_bboxes is not None:
            reinit_bboxes = utils.format_boxes([reinit_bboxes[0:4]], original_h, original_w)[0]


        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        # allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = []
        if len(boxs) > 0:
            indices = tf.image.non_max_suppression(boxs, scores, 50, iou_threshold=FLAGS.iou,
                score_threshold=FLAGS.score)
        detections = [detections[i] for i in indices]

        # track_time = time.time()
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        # print("Tracker: ", (time.time() - track_time))

        # update tracks
        for track in tracker.tracks:
            if not track.is_confirmed(): #or track.track_id != 1: # or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            class_name = track.get_class()
            print(reinit_bboxes)
            print(bbox)
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)

        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # calculate frames per second of running detections
        # fps = 1.0 / (time.time() - start_time)
        # print("FPS: %.2f" % fps)

        # print("Runtime: ",(time.time() - start_time))
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # print("\n")
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
