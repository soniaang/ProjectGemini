import os
import json

from azureml.core import Workspace, Datastore, ComputeTarget, Experiment
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import DatabricksStep


def create_experiment_config(workspace):
    ########################################
    ### Creating data load Pipeline Step ###
    ########################################

    # Load settings
    print("Loading settings")
    data_load_step_path = os.path.join("steps", "data_load")
    with open(os.path.join(data_load_step_path, "step.json")) as f:
        data_load_settings = json.load(f)

    # Setup of pipeline parameter
    print("Setting up pipeline parameters")
    data_load_environment = PipelineParameter(
        name="environment",
        default_value="golden"
    )
    data_load_start_date = PipelineParameter(
        name="start_date",
        default_value="2019-01-01"
    )
    data_load_end_date = PipelineParameter(
        name="end_date",
        default_value="2019-01-31"
    )
    data_load_system = PipelineParameter(
        name="system",
        default_value="PAX 1"
    )
    data_load_platform = PipelineParameter(
        name="platform",
        default_value="Atlantis"
    )

    # Loading compute target 
    print("Loading ComputeTarget")
    data_load_compute_target = ComputeTarget(
        workspace=workspace,
        name=data_load_settings.get("compute_target_name", None)
    )

    # Create Databricks step
    print("Creating Step")
    data_load = DatabricksStep(
        name=data_load_settings.get("step_name", None),
        existing_cluster_id=data_load_settings.get("existing_cluster_id", None),
        inputs=[],
        outputs=[],
        compute_target=data_load_compute_target,
        notebook_path=data_load_settings.get("notebook_path", None),
        notebook_params={"environment": data_load_environment, "start_date": data_load_start_date, "end_date": data_load_end_date, "system": data_load_system, "platform": data_load_platform},
        run_name=data_load_settings.get("step_name", None),
        allow_reuse=data_load_settings.get("allow_reuse", True),
        version=data_load_settings.get("version", None),
    )

    #########################
    ### Creating Pipeline ###
    #########################

    # Create Pipeline
    print("Creating Pipeline")
    pipeline = Pipeline(
        workspace=workspace,
        steps=[data_load],
        description="Training Pipeline",
    )

    # Validate pipeline
    print("Validating pipeline")
    pipeline.validate()

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
        name="myexperiment_data_load"
    )

    # Submit experiment config
    print("Submitting experiment config")
    run = experiment.submit(
        config=config,
        tags={}
    )
    run.wait_for_completion(show_output=True)