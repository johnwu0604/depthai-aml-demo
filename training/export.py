import os
import sys
import subprocess
import requests
import zipfile
import tensorflow as tf
tf.get_logger().setLevel('INFO')

print('Installing additional libraries .....')
os.system('apt-get update')
os.system('apt-get install ffmpeg libsm6 libxext6  -y')
print('Done. \n')

print('Cloning tensorflow models repo .....')
os.system('git clone https://github.com/tensorflow/models tensorflow-models')
print('Done. \n')

root_dir = os.path.abspath('.')
research_dir = os.path.abspath('tensorflow-models/research')
slim_dir = os.path.abspath('tensorflow-models/research/slim')

sys.path.insert(0, slim_dir)
sys.path.insert(0, research_dir)

print('Compiling tensorflow models proto files .....')
os.chdir(research_dir)
os.system('protoc object_detection/protos/*.proto --python_out=.')
print('Done. \n')

print('Exporting model .....')
last_checkpoint = max([f for f in os.listdir('outputs')])
train_script = os.path.abspath('tensorflow-models/research/object_detection/export_inference_graph.py')
sys.argv = ['--input_type', 'image_tensor', 
            '--pipeline_config_path', 'model.config', 
            '--trained_checkpoint_prefix', 'outputs/{}'.format(last_checkpoint.split('.m')[0]),
            '--output_directory', 'outputs/final-model']
exec(open(train_script).read())
print('Done. \n')
