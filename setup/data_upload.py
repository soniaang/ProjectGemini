from azureml.core import Workspace, Datastore, Dataset


def main(args):
    # Load workspace
    print("Loading Workspace")
    workspace = Workspace.from_config()
    print(
        f"Workspace name: {workspace.name}", 
        f"Azure region: {workspace.location}", 
        f"Subscription id: {workspace.subscription_id}", 
        f"Resource group: {workspace.resource_group}",
        sep="\n"
    )

    # Printing all datastores
    print("Printing all datastores")
    for name, datastore in workspace.datastores.items():
        print(name, datastore.datastore_type, sep="\t")
    
    # Load datastore
    print("Loading datastore")
    datastore = Datastore(
        workspace=workspace,
        name=args.datastore_name
    )

    # Upload dataset
    print("Uploading dataset")
    datastore.upload_files(
        files=["./train_dataset/iris.csv"],
        target_path="train_dataset/iris.csv",
        overwrite=True,
        show_progress=True
    )

    # Register dataset
    file_dataset = Dataset.File.from_files(
        
    )


def parse_args():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--datastore_name", default="workspaceblobstore", type=str, help="Name of datastore")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)