import os
import json

from azureml.core import Workspace, Datastore, Dataset, ComputeTarget, RunConfiguration, Experiment
from azureml.core.runconfig import CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep, HyperDriveStep, EstimatorStep
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal
from azureml.train.dnn import Chainer, PyTorch, TensorFlow
from azureml.train.estimator import Estimator
from utils import get_distributed_backend, get_parameter_distribution, get_parameter_sampling, get_policy


def create_experiment_config(workspace):
    ########################################
    ### Creating data prep Pipeline Step ###
    ########################################

    # Load settings
    print("Loading settings")
    data_prep_step_path = os.path.join("steps", "data_prep")
    with open(os.path.join(data_prep_step_path, "step.json")) as f:
        data_prep_settings = json.load(f)

    # Setup datasets of first step
    print("Setting up datasets")
    data_prep_input = Dataset.get_by_name(
        workspace=workspace,
        name=data_prep_settings.get("dataset_input_name", None)
    ).as_named_input(data_prep_settings.get("dataset_input_name", None)).as_mount()
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
        arguments=data_prep_settings.get("arguments", []),
        compute_target=data_prep_compute_target,
        runconfig=data_prep_run_config,
        inputs=[data_prep_input],
        outputs=[data_prep_output],
        params=data_prep_settings.get("parameters", []),
        source_directory=data_prep_step_path,
        allow_reuse=data_prep_settings.get("allow_reuse", True),
        version=data_prep_settings.get("version", None),
    )

    
    ###############################################
    ### Creating data model train Pipeline Step ###
    ###############################################

    # Load settings
    print("Loading settings")
    model_train_step_path = os.path.join("steps", "model_train")
    with open(os.path.join(model_train_step_path, "step.json")) as f:
        model_train_settings = json.load(f)
    hyperparameter_sampling_settings = model_train_settings.get("hyperparameter_sampling", {})

    # Setup datasets of first step
    print("Setting up datasets")
    model_train_input = data_prep_output.as_named_input(name=model_train_settings.get("dataset_input_name", None))
    model_train_output = PipelineData(
        name=model_train_settings.get("dataset_output_name", None),
        datastore=Datastore(workspace=workspace, name=model_train_settings.get("datastore_output_name", None)),
        output_mode="mount",
    ).as_dataset()
    # Uncomment next lines, if you want to register intermediate dataset
    #model_train_output.register(
    #    name=model_train_settings.get("dataset_output_name", None),
    #    create_new_version=True
    #)

    # Create conda dependencies
    print("Creating conda dependencies")
    model_train_dependencies = CondaDependencies.create(
        pip_packages=model_train_settings.get("pip_packages", []),
        conda_packages=model_train_settings.get("conda_packages", []),
        python_version=model_train_settings.get("python_version", "3.6.2")
    )

    # Create run configuration
    print("Creating RunConfiguration")
    model_train_run_config = RunConfiguration(
        conda_dependencies=model_train_dependencies,
        framework=model_train_settings.get("framework", "Python")
    )

    # Loading compute target 
    print("Loading ComputeTarget")
    model_train_compute_target = ComputeTarget(
        workspace=workspace,
        name=model_train_settings.get("compute_target_name", None)
    )

    # Create distributed training backend
    print("Creating distributed training backend")
    distributed_training_backend = get_distributed_backend(
        backend_name=model_train_settings.get("distributed_backend", None)
    )

    # Create Estimator for Training
    print("Creating Estimator for training")
    model_train_estimator = Estimator(
        source_directory=model_train_step_path,
        entry_script=model_train_settings.get("script_name", None),
        environment_variables=model_train_settings.get("parameters", None),
        compute_target=model_train_compute_target,
        node_count=model_train_settings.get("node_count", None),
        distributed_training=distributed_training_backend,
        conda_packages=model_train_settings.get("conda_packages", None),
        pip_packages=model_train_settings.get("pip_packages", None),
    )

    try:
        # Create parameter sampling
        print("Creating Parameter Sampling")
        parameter_dict = {}
        parameters = hyperparameter_sampling_settings.get("parameters", {}) if "parameters" in hyperparameter_sampling_settings else {}
        for parameter_name, parameter_details in parameters.items():
            parameter_distr = get_parameter_distribution(
                distribution=parameter_details.get("distribution", None),
                **parameter_details.get("settings", {})
            )
            parameter_dict[f"--{parameter_name}"] = parameter_distr
        model_train_ps = get_parameter_sampling(
            sampling_method=hyperparameter_sampling_settings.get("method", None),
            parameter_dict=parameter_dict
        )

        # Get Policy definition
        policy_settings = hyperparameter_sampling_settings.get("policy", {})
        kwargs = {key:value for key, value in policy_settings.items() if key not in ["policy_method", "evaluation_interval", "delay_evaluation"]}
        
        # Create termination policy
        print("Creating early termination policy")
        model_train_policy = get_policy(
            policy_method=policy_settings.get("method", ""),
            evaluation_interval=policy_settings.get("evaluation_interval", None),
            delay_evaluation=policy_settings.get("delay_evaluation", None),
            **kwargs
        )

        # Create HyperDriveConfig
        print("Creating HyperDriveConfig")
        model_train_hyperdrive_config = HyperDriveConfig(
            estimator=model_train_estimator,
            hyperparameter_sampling=model_train_ps,
            policy=model_train_policy,
            primary_metric_name=hyperparameter_sampling_settings.get("primary_metric", None),
            primary_metric_goal=PrimaryMetricGoal.MINIMIZE if "min" in hyperparameter_sampling_settings.get("primary_metric_goal", None) else PrimaryMetricGoal.MAXIMIZE,
            max_total_runs=hyperparameter_sampling_settings.get("max_total_runs", 1),
            max_concurrent_runs=hyperparameter_sampling_settings.get("max_concurrent_runs", 1),
            max_duration_minutes=hyperparameter_sampling_settings.get("max_duration_minutes", None)
        )

        # Create HyperDriveStep
        print("Creating HyperDriveStep")
        model_train = HyperDriveStep(
            name=model_train_settings.get("step_name", None),
            hyperdrive_config=model_train_hyperdrive_config,
            estimator_entry_script_arguments=model_train_settings.get("arguments", None),
            inputs=[model_train_input],
            outputs=[model_train_output],
            allow_reuse=model_train_settings.get("allow_reuse", True),
            version=model_train_settings.get("version", True)
        )
    except:
        print("Not all required parameters specified for HyperDrive step")

        # Create EstimatorStep
        print("Creating EstimatorStep")
        model_train = EstimatorStep(
            name=model_train_settings.get("step_name", None),
            estimator=model_train_estimator,
            estimator_entry_script_arguments=model_train_settings.get("arguments", None),
            inputs=[model_train_input],
            outputs=[model_train_output],
            compute_target=model_train_compute_target,
            allow_reuse=model_train_settings.get("allow_reuse", True),
            version=model_train_settings.get("version", True)
        )
    

    #########################
    ### Creating Pipeline ###
    #########################

    # Create Pipeline
    print("Creating Pipeline")
    pipeline = Pipeline(
        workspace=workspace,
        steps=[model_train],
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
        name="myexperiment"
    )

    # Submit experiment config
    print("Submitting experiment config")
    run = experiment.submit(
        config=config,
        tags={}
    )
    run.wait_for_completion(show_output=True)