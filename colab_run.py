import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

def initializer(_argv):
    flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
    flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
    flags.DEFINE_integer('size', 416, 'resize images to')
    flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
    flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
    flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
    flags.DEFINE_string('output', None, 'path to output video')
    flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
    flags.DEFINE_float('iou', 0.45, 'iou threshold')
    flags.DEFINE_float('score', 0.50, 'score threshold')
    flags.DEFINE_boolean('dont_show', False, 'dont show video output')
    flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
    flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    return tracker, infer, interpreter, encoder, FLAGS