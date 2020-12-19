import argparse
import os
import re
import sys
import subprocess
import requests
import zipfile
import tensorflow as tf
tf.get_logger().setLevel('INFO')

# Get arguments from parser
parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--data_dir', type=str, help='Directory where training data is stored')
parser.add_argument('--checkpoint_dir', type=str, help='Directory where initial checkpoint is stored')
parser.add_argument('--tensorflow_models_dir', type=str, help='Directory where tensorflow model directory is stored')
parser.add_argument('--output_dir', type=str, help='Directory where outputs will be stored')
args = parser.parse_args()

data_dir = args.data_dir
checkpoint_dir = args.checkpoint_dir
tensorflow_models_dir = args.tensorflow_models_dir
output_dir = args.output_dir

# Set paths to tensorflow models folder
research_dir = os.path.abspath('{}/research'.format(tensorflow_models_dir))
slim_dir = os.path.abspath('{}/research/slim'.format(tensorflow_models_dir))
sys.path.insert(0, slim_dir)
sys.path.insert(0, research_dir)

# Install additional libraries
os.system('apt-get update')
os.system('apt-get install ffmpeg libsm6 libxext6  -y')

# Update config file with mounted data locations
config_file = 'model.config'
with open(config_file) as f:
    s = f.read()
with open(config_file, 'w') as f:
    s = re.sub('fine_tune_checkpoint: ".*?"',
               'fine_tune_checkpoint: "{}/model.ckpt"'.format(checkpoint_dir), s)
    s = re.sub(
        '(input_path: ".*?)(train.record)(.*?")', 'input_path: "{}/tf_record/train.record"'.format(data_dir), s)
    s = re.sub(
        '(input_path: ".*?)(val.record)(.*?")', 'input_path: "{}/tf_record/val.record"'.format(data_dir), s)
    s = re.sub(
        'label_map_path: ".*?"', 'label_map_path: "{}/annotations/label_map.pbtxt"'.format(data_dir), s)
    f.write(s)

# Run training script
train_script = '{}/research/object_detection/model_main.py'.format(tensorflow_models_dir)
sys.argv = ['--logtostderr', 
            '--pipeline_config_path', config_file, 
            '--model_dir', output_dir]
exec(open(train_script).read())

