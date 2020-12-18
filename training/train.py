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

# print('Downloading masks dataset .....')
# os.chdir(root_dir)
# data_dir = 'data'
# if not os.path.exists(data_dir):
#     zip_file_name = 'data.zip'
#     if not os.path.exists(zip_file_name):
#         url = 'https://johndatasets.blob.core.windows.net/masks/data.zip?sp=r&st=2020-12-17T00:18:43Z&se=2022-03-01T08:18:43Z&spr=https&sv=2019-12-12&sr=b&sig=SfcpEhbQ1vYhWelmz0Ow873Ska%2B4mB5W43CdOQ3H6LI%3D'
#         r = requests.get(url, allow_redirects=True)
#         with open(zip_file_name, 'wb') as f:
#             f.write(r.content)
#     with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
#         zip_ref.extractall('.')
# print('Done. \n')

# print('Downloading pretrained weights for SSD Mobile Net V2 COCO model .....')
# checkpoints_dir = 'checkpoints'
# if not os.path.exists(checkpoints_dir):
#     zip_file_name = 'checkpoints.tar.gz'
#     if not os.path.exists(zip_file_name):
#         url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz'
#         r = requests.get(url, allow_redirects=True)
#         with open(zip_file_name, 'wb') as f:
#             f.write(r.content)
#     os.mkdir(checkpoints_dir)
#     os.system('tar --extract --file {} --strip-components=1 --directory {}'.format(zip_file_name, checkpoints_dir))
# print('Done. \n')

print('Training model .....')
os.chdir(root_dir)
train_script = 'tensorflow-models/research/object_detection/model_main.py'
sys.argv = ['--logtostderr', 
            '--pipeline_config_path', 'model.config', 
            '--model_dir', 'outputs']
exec(open(train_script).read())
print('Done. \n')

