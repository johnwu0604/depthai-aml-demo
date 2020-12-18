from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.runconfig import RunConfiguration 
from azureml.data.data_reference import DataReference
from azureml.pipeline.steps import PythonScriptStep

# Get workspace, datastores, and compute targets
print('Connecting to Workspace ...')
workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()
compute_target = workspace.compute_targets['v100cluster']

# Get dataset and checkpoints
dataset = workspace.datasets['mask-data']
checkpoint = workspace.datasets['ssd-mobilenet-v2-checkpoint']

# Create run environment
env = Environment.from_conda_specification(name='mask-detector', file_path='env.yml')
env.docker.enabled = True
env.docker.base_image = 'mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04'

run_config = RunConfiguration()
run_config.environment = env

# Step: Train Model
output_dir = PipelineData(
    name='outputs', 
    pipeline_output_name='outputs',
    datastore=datastore,
    output_mode='mount',
    is_directory=True)

train_step = PythonScriptStep(name='Train Model',
                        source_directory='.',
                        script_name='train.py', 
                        compute_target=compute_target, 
                        inputs=[dataset.as_named_input('data'), checkpoint.as_named_input('checkpoint')],
                        outputs=[output_dir],
                        runconfig=run_config)

# Submit pipeline
print('Submitting pipeline ...')
pipeline = Pipeline(workspace=workspace, steps=[train_step])
pipeline_run = Experiment(workspace, 'mask-detector').submit(pipeline)