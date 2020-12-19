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
dataset = workspace.datasets['mask-data'].as_named_input('data').as_mount()
checkpoint = workspace.datasets['ssd-mobilenet-v2-checkpoint'].as_named_input('checkpoint').as_mount()
tensorflow_models = workspace.datasets['tensorflow-models'].as_named_input('tensorflowmodel').as_mount()

# Create run environment
env = Environment.from_conda_specification(name='mask-detector', file_path='env.yml')
env.docker.enabled = True
env.docker.base_image = 'mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04'

run_config = RunConfiguration()
run_config.environment = env

# Step 1: Train Model
train_output_dir = PipelineData(
    name='train_output', 
    pipeline_output_name='train_output',
    datastore=datastore,
    output_mode='mount',
    is_directory=True)

train_step = PythonScriptStep(name='Train Model',
                        source_directory='.',
                        script_name='train.py', 
                        compute_target=compute_target, 
                        arguments=['--data_dir', dataset, 
                                   '--checkpoint_dir', checkpoint,
                                   '--tensorflow_models_dir', tensorflow_models,
                                   '--output_dir', train_output_dir],
                        inputs=[dataset, checkpoint, tensorflow_models],
                        outputs=[train_output_dir],
                        runconfig=run_config)

# Step 2: Export Model
export_output_dir = PipelineData(
    name='export_output', 
    pipeline_output_name='export_output',
    datastore=datastore,
    output_mode='mount',
    is_directory=True)

export_step = PythonScriptStep(name='Export Model',
                               source_directory='.',
                               script_name='export.py', 
                               compute_target=compute_target, 
                               arguments=['--data_dir', dataset, 
                                          '--checkpoint_dir', checkpoint,
                                          '--tensorflow_models_dir', tensorflow_models,
                                          '--training_results_dir', train_output_dir,
                                          '--output_dir', export_output_dir],
                                inputs=[dataset, checkpoint, tensorflow_models, train_output_dir],
                                outputs=[export_output_dir],
                                runconfig=run_config)

# Step 3: Convert Model To OpenVINO format
convert_output_dir = PipelineData(
    name='convert_output_dir', 
    pipeline_output_name='convert_output_dir',
    datastore=datastore,
    output_mode='mount',
    is_directory=True)

convert_step = PythonScriptStep(name='Convert Model',
                                source_directory='.',
                                script_name='convert.py', 
                                compute_target=compute_target, 
                                arguments=['--exported_model_dir', export_output_dir, 
                                           '--output_dir', convert_output_dir],
                                inputs=[export_output_dir],
                                outputs=[convert_output_dir],
                                runconfig=run_config)


# Submit pipeline
print('Submitting pipeline ...')
pipeline = Pipeline(workspace=workspace, steps=[train_step, export_step, convert_step])
pipeline_run = Experiment(workspace, 'mask-detector').submit(pipeline)