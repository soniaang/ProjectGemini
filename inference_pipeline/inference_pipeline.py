import json, os

from azureml.core import Workspace, Experiment, Datastore, Dataset, ComputeTarget, RunConfiguration, Environment, Model
from azureml.core.runconfig import CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.contrib.pipeline.steps import ParallelRunStep, ParallelRunConfig
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.pipeline.steps import PythonScriptStep


def create_experiment_config(workspace):
    ########################################
    ### Creating data prep Pipeline Step ###
    ########################################

    # Load settings
    print("Loading settings")
    data_prep_step_path = os.path.join("steps", "data_prep")
    with open(os.path.join(data_prep_step_path, "step.json")) as f:
        data_prep_settings = json.load(f)
    
    # Setup datasets - Create PipelineParameter for dynamic pipeline input
    print("Setting up datasets with dynamic input")
    data_prep_input_path = DataPath(
        datastore=Datastore(workspace=workspace, name=data_prep_settings.get("datastore_input_name", "workspaceblobstore")),
        path_on_datastore="golden/Atlantis/PAX1/15-Mar-2020-23-37-50-279971/PAX1.parquet/"
    )
    data_prep_input_path_pipeline_parameter = PipelineParameter(
        name="input_path",
        default_value=data_prep_input_path
    )
    data_prep_input = (data_prep_input_path_pipeline_parameter, DataPathComputeBinding(mode="mount"))
    data_prep_output = PipelineData(
        name=data_prep_settings.get("dataset_output_name", None),
        datastore=Datastore(workspace=workspace, name=data_prep_settings.get("datastore_output_name", "workspaceblobstore")),
        output_mode="mount"
    ).as_dataset()
    # Uncomment next lines, if you want to register intermediate dataset
    #data_prep_output.register(
    #    name=data_prep_settings.get("dataset_output_name", None),
    #    create_new_version=True
    #)

    # Create conda dependencies
    print("Creating conda dependencies")
    data_prep_dependencies = CondaDependencies.create(
        pip_packages=data_prep_settings.get("pip_packages", []),
        conda_packages=data_prep_settings.get("conda_packages", []),
        python_version=data_prep_settings.get("python_version", "3.6.2")
    )

    # Create run configuration
    print("Creating RunConfiguration")
    data_prep_run_config = RunConfiguration(
        conda_dependencies=data_prep_dependencies,
        framework=data_prep_settings.get("framework", "Python")
    )

    # Loading compute target 
    print("Loading ComputeTarget")
    data_prep_compute_target = ComputeTarget(
        workspace=workspace,
        name=data_prep_settings.get("compute_target_name", None)
    )

    # Create python step
    print("Creating Step")
    data_prep = PythonScriptStep(
        name=data_prep_settings.get("step_name", None),
        script_name=data_prep_settings.get("script_name", None),
        arguments=data_prep_settings.get("arguments", []) + ["--input-datapath", data_prep_input],
        compute_target=data_prep_compute_target,
        runconfig=data_prep_run_config,
        inputs=[data_prep_input],
        outputs=[data_prep_output],
        params=data_prep_settings.get("parameters", []),
        source_directory=data_prep_step_path,
        allow_reuse=data_prep_settings.get("allow_reuse", True),
        version=data_prep_settings.get("version", None),
    )

    ############################################
    ### Creating inference Parallel Run Step ###
    ############################################

    # Load settings
    print("Loading settings")
    batch_inference_step_path = os.path.join("steps", "batch_inference")
    with open(os.path.join(batch_inference_step_path, "step.json")) as f:
        batch_inference_settings = json.load(f)

    # Setup datasets of first step
    print("Setting up datasets")
    batch_inference_input = data_prep_output.as_named_input(name=batch_inference_settings.get("dataset_input_name", None))
    batch_inference_output = PipelineData(
        name=batch_inference_settings.get("dataset_output_name", None),
        datastore=Datastore(workspace=workspace, name=batch_inference_settings.get("datastore_output_name", None)),
        output_mode="mount",
    ).as_dataset()
    # Uncomment next lines, if you want to register intermediate dataset
    #batch_inference_output.register(
    #    name=batch_inference_settings.get("dataset_output_name", None),
    #    create_new_version=True
    #)

    # Create conda dependencies
    print("Creating conda dependencies")
    batch_inference_dependencies = CondaDependencies.create(
        pip_packages=batch_inference_settings.get("pip_packages", []),
        conda_packages=batch_inference_settings.get("conda_packages", []),
        python_version=batch_inference_settings.get("python_version", "3.6.2")
    )

    # Create run configuration
    print("Creating RunConfiguration")
    data_prep_run_config = RunConfiguration(
        conda_dependencies=batch_inference_dependencies,
        framework=batch_inference_settings.get("framework", "Python")
    )

    # Loading compute target 
    print("Loading ComputeTarget")
    batch_inference_compute_target = ComputeTarget(
        workspace=workspace,
        name=batch_inference_settings.get("compute_target_name", None)
    )

    # Create python step
    print("Creating Step")
    batch_inference = PythonScriptStep(
        name=batch_inference_settings.get("step_name", None),
        script_name=batch_inference_settings.get("script_name", None),
        arguments=batch_inference_settings.get("arguments", []),
        compute_target=batch_inference_compute_target,
        runconfig=data_prep_run_config,
        inputs=[batch_inference_input],
        outputs=[batch_inference_output],
        params=batch_inference_settings.get("parameters", []),
        source_directory=batch_inference_step_path,
        allow_reuse=batch_inference_settings.get("allow_reuse", True),
        version=batch_inference_settings.get("version", None),
    )
    
    #########################
    ### Creating Pipeline ###
    #########################

    # Create Pipeline
    print("Creating Pipeline")
    pipeline = Pipeline(
        workspace=workspace,
        steps=[batch_inference],
        description="Batch Inference Pipeline",
    )

    return pipeline


if __name__ == "__main__":
    # Load workspace
    print("Load Workspace")
    ws = Workspace.from_config()

    # Load experiment config
    print("Loading experiment config")
    config = create_experiment_config(workspace=ws)

    # Load experiment
    print("Loading experiment")
    experiment = Experiment(
        workspace=ws,
        name="myexperiment_inference"
    )

    # Submit experiment config
    print("Submitting experiment config")
    run = experiment.submit(
        config=config,
        tags={}
    )
    run.wait_for_completion(show_output=True)