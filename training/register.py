import os
import sys
import argparse
from azureml.core import Model
from azureml.core import Workspace

parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--converted_model_dir', type=str, help='Directory where converted model is stored')
args = parser.parse_args()

converted_model_dir = args.converted_model_dir

workspace = Workspace.from_config()
model = Model.register(workspace = workspace,
                       model_path ='{}/mobilenet-ssd-face-mask.blob'.format(converted_model_dir),
                       model_name = 'face-mask-detector',
                       description = 'A face mask detecting model in compiled OpenVINO IR format for deploying on opencv AI kit')