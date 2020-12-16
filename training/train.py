import os
import sys
import subprocess
import requests

root_dir = os.path.abspath('.')
research_dir = os.path.abspath('tensorflow-models/research')
slim_dir = os.path.abspath('tensorflow-models/research/slim')

sys.path.insert(0, slim_dir)
sys.path.insert(0, research_dir)

print('Compiling tensorflow models proto files .....')
os.chdir(research_dir)
os.system('protoc object_detection/protos/*.proto --python_out=.')
print('Done. \n')

print('Running tensorflow models test script .....')
test_script = os.path.abspath('object_detection/builders/model_builder_test.py')
exec(open(test_script).read())
print('Done. \n')

print('Downloading pretrained weights for SSD Mobile Net V2 COCO model .....')
os.chdir(root_dir)
checkpoints_dir = 'checkpoints'
if not os.path.exists(checkpoints_dir):
    zip_file_name = 'checkpoints.tar.gz'
    if not os.path.exists(zip_file_name):
        url = 'http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz'
        r = requests.get(url, allow_redirects=True)
        with open(zip_file_name, 'wb') as f:
            f.write(r.content)
    os.mkdir(checkpoints_dir)
    os.system('tar --extract --file {} --strip-components=1 --directory {}'.format(zip_file_name, checkpoints_dir))
print('Done. \n')