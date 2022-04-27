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

def initializer(weights='./checkpoints/yolov4-tiny-416', model='yolov4', video=0, tiny=True, size=416):
    FLAGS = {"weights": weights, "framework":'tf', 'size': size, 'tiny': tiny, 'model': model, 'video': video, 'output': None, 'output_format': 'XVID', 
                    'iou': 0.45, 'score': 0.50, 'dont_show': False, 'info': False, 'count': False}

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
    #session = InteractiveSession(config=config)
    #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    #video_path = FLAGS['video']

    # load tflite model if flag is set
    if FLAGS['framework'] == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS['weights'])
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS['weights'], tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
    
    return tracker, infer, encoder, FLAGS