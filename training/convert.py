import os
import sys
import argparse
import subprocess
import requests
import time

# Get arguments from parser
parser = argparse.ArgumentParser(description='Training arg parser')
parser.add_argument('--exported_model_dir', type=str, help='Directory where final model is exported to')
parser.add_argument('--output_dir', type=str, help='Directory where outputs will be stored')
args = parser.parse_args()

exported_model_dir = args.exported_model_dir
output_dir = args.output_dir

# Install tools
print('Installing dependencies...')
os.system('apt-get update')
os.system('apt-get install pciutils cpio -y')
os.system('apt autoremove -y')

print('Downloading OpenVINO...')
os.system('wget http://registrationcenter-download.intel.com/akdlm/irc_nas/16345/l_openvino_toolkit_p_2020.1.023.tgz')
os.system('tar xf l_openvino_toolkit_p_2020.1.023.tgz')
os.chdir('l_openvino_toolkit_p_2020.1.023')

print('Installing OpenVINO dependencies...')
os.system('ls .')
os.system('./install_openvino_dependencies.sh')
os.system('sed -i "s/decline/accept/g" silent.cfg')
os.system('./install.sh --silent silent.cfg')

print('Modifying JSON...')
os.chdir('/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/')

with open('ssd_v2_support.json', 'r') as f:
  filedata = f.read()

filedata = filedata.replace('"Postprocessor/ToFloat"', '"Postprocessor/Cast_1"')
with open('ssd_v2_support.json', 'w') as f:
  f.write(filedata)

# Convert model
print('Converting Model ...')
export_script = os.path.abspath('/opt/intel/openvino/deployment_tools/model_optimizer/mo.py')
os.chdir(exported_model_dir)

print('Sourcing setup ...')
os.system('echo "dash dash/sh boolean false" | debconf-set-selections')
os.system('mkdir -p /usr/share/man/man1')
os.system('dpkg-reconfigure -p critical dash')

print('Sourcing script ...')
os.system('source /opt/intel/openvino/bin/setupvars.sh')

print('Running  ...')
os.system('python {} --input_model frozen_inference_graph.pb \
                     --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json \
                     --tensorflow_object_detection_api_pipeline_config pipeline.config \
                     --reverse_input_channels \
                     --output_dir {}\
                     --data_type FP16'.format(export_script, output_dir))

# Compile model 
print('Compiling Model ...')
xmlfile = '{}/frozen_inference_graph.xml'.format(output_dir)
binfile = '{}/frozen_inference_graph.bin'.format(output_dir)
url = 'http://69.164.214.171:8080'

payload = {'compiler_params': '-ip U8 -VPU_MYRIAD_PLATFORM VPU_MYRIAD_2480 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4'}
files = [
  ('definition', open(xmlfile,'rb')),
  ('weights', open(binfile,'rb'))
]

output_file = '{}/mobilenet-ssd-face-mask.blob'.format(output_dir)
response = requests.request('POST', url, data=payload, files=files)
with open(output_file, 'wb') as f:
  f.write(response.content)
