import os
import sys
import subprocess

root_dir = os.path.abspath('.')
research_dir = os.path.abspath('tensorflow-models/research')
slim_dir = os.path.abspath('tensorflow-models/research/slim')

# Add to python path
sys.path.insert(0, slim_dir)
sys.path.insert(0, research_dir)

print('Compiling proto files .....')
os.chdir(research_dir)
os.system('protoc object_detection/protos/*.proto --python_out=.')
print('Done. \n')

# # print('Building object detection library .....')
# # os.system('python setup.py build')
# # os.system('python setup.py install')
# # print('Done. \n')

print('Running test script .....')
os.system('python object_detection/builders/model_builder_test.py')
print('Done. \n')
