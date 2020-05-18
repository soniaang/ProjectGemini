import argparse

from azureml.core import Workspace
from azureml.core.compute import DatabricksCompute, ComputeTarget
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.exceptions import ComputeTargetException

def main(args, workspace):
    # Connect Databricks to AzureML
    print("Connecting Databricks to AzureML")
    try:
        databricks_compute = DatabricksCompute(
            workspace=workspace,
            name=args.compute_name
        )
        print(f"Compute target {databricks_compute.name} already exists")
    except ComputeTargetException as exception:
        print(f"Databricks compute target not found: {exception}")
        print("Attaching Databricks to Azure ML")
        databricks_config = DatabricksCompute.attach_configuration(
            resource_group=args.db_resource_group,
            workspace_name=args.db_workspace_name,
            access_token=args.db_access_token
        )
        databricks_compute = ComputeTarget.attach(
            workspace=workspace,
            name=args.compute_name,
            attach_configuration=databricks_config
        )
    databricks_compute.wait_for_completion(
        show_output=True
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Databricks arguments")
    parser.add_argument("--compute-name", dest="compute_name", type=str, default="databricks", help="Name of Databricks Compute in AML")
    parser.add_argument("--db-resource-group", dest="db_resource_group", type=str, help="Resource group of Databricks workspace")
    parser.add_argument("--db-workspace-name", dest="db_workspace_name", type=str, help="Name of Databricks workspace")
    parser.add_argument("--db-access-token", dest="db_access_token", type=str, help="Access token of Databricks workspace")
    parser.add_argument("--aml-resource-group", dest="aml_resource_group", type=str, help="Resource group of Azure Machine Learning")
    parser.add_argument("--aml-subscription-id", dest="aml_subscription_id", type=str, help="Subscription ID of Azure Machine Learning")
    parser.add_argument("--aml-workspace-name", dest="aml_workspace_name", type=str, help="Name of Azure Machine Learning workspace")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Load arguments
    print("Loading arguments")
    args = parse_args()

    # Load workspace
    print("Load Workspace")
    interactive_auth = InteractiveLoginAuthentication()
    ws = Workspace(
        subscription_id=args.aml_subscription_id,
        resource_group=args.aml_resource_group,
        workspace_name=args.aml_workspace_name,
        auth=interactive_auth
    )

    # Attach ADB as remote compute
    print("Attaching ADB as remote compute")
    main(args=args, workspace=ws)