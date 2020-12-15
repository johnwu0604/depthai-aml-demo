import os
import sys
import pathlib

root_dir = pathlib.Path().absolute()
research_dir = pathlib.Path('tensorflow-models/research').absolute()
object_detection_dir = pathlib.Path('tensorflow-models/research/object_detection').absolute()
slim_dir = pathlib.Path('tensorflow-models/research/slim').absolute()

# Add to python path
sys.path.append(research_dir)
sys.path.append(object_detection_dir)
sys.path.append(slim_dir)

# Compile proto files
os.chdir(research_dir)
os.system('protoc object_detection/protos/*.proto --python_out=.')

os.system('python setup.py build')
os.system('python setup.py install')
os.system('python object_detection/builders/model_builder_test.py')
