# Depth AI - Azure ML Demo

The following repo demonstrates an example a custom object detection model being trained in Azure ML and then deployed onto the OpenCV AI Kit. 

## Prerequisites 

Build and activate a new conda enviornment using the `env.yml` file as follows:

```
conda create --file env.yml
conda activate depthai-aml
```

## Training

The `training` folder contains all the source code used for training the model using pipelines in Azure ML. The pipeline consists of the following steps:

1. Trains a model using Tensorflow to detect when a person is wearing a face mask, not wearing a face mask, or wearing a face mask incorrectly.
2. Exports the model into a frozen inference graph.
3. Converts the inference graph into an OpenVINO format and registers it in the Azure ML workspace.

To train the model on Azure ML:

1. Make sure you have an Azure ML workspace created, along with a compute target created.
2. Change the `config.json` file to match the properties of your workspace.
3. Change into the train directory and run `python submit-pipeline.py --compute-target <NAME OF COMPUTE TARGET TO USE>`.
4. Go to https://ml.azure.com to check on progress of the run.

## Inference

The `inference` folder contains all the source code used for running the trained model on an OpenCV AI kit. To run the model:

1. Make sure OpenCV AI camera is plugged in
2. Change into the inference directory and run `python run.py`
