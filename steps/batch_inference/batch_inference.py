import pickle
import pandas as pd
import argparse
import joblib

from sklearn.ensemble import IsolationForest
from azureml.core import Model, Run
run = Run.get_context()


def main(args):
    # Load model
    print("Loading model")
    model_path = Model.get_model_path(
        model_name=args.model_name,
        version=args.model_version
    )
    with open(model_path, "rb") as model_file:
        my_model = joblib.load(filename=model_file)

    # Load mounted dataset path
    print("Get file dataset mount paths")
    input_path = os.environ.get(args.input_dataset, None)
    output_path = os.environ.get(f"AZUREML_DATAREFERENCE_{args.output_dataset}", None)
    print(f"Input mount path: {input_path}")
    print(f"Output mount path: {output_path}")

    # Create output path
    print("Creating output path and /outputs folder")
    os.makedirs(
        name=output_path,
        exist_ok=True
    )

    # Get input file paths list
    print("Get input file paths")
    paths = []
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            if ".parquet" in filename:
                path = os.path.join(root, filename)
                paths.append(path)
    print(f"Path List: {paths}")
    
    # Create one large dataframe from all files
    if len(paths) > 0:
        print("Creating one large pandas dataframe")
        df = pd.read_parquet(
            path=paths.pop(),
            engine="auto"
        )
        for path in paths:
            df_temp = pd.read_parquet(
                path=path,
                engine="auto"
            )
            df.append(df_temp)
    else:
        print("File dataset does not include any files")
        return

    # Score data
    print("Scoring data in dataframe")
    num_rows = df.shape
    predictions = my_model.predict(df) #.reshape((num_rows, 1))
    result = df
    result["predictions"] = predictions

    # Save parquet for training
    print("Saving Parquet file for training")
    result.to_parquet(
        path=os.path.join(output_path, "result.parquet")
    )


def parse_args():
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--input-dataset", dest="input_dataset", type=str, help="Name of input dataset")
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Name of output dataset")
    parser.add_argument("--model-name", dest="model_name", type=str, help="Name of model")
    parser.add_argument("--model-version", dest="model_version", default=None, type=str, help="Version of model")
    args, unknown_args = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args= parse_args()
    main(args=args)