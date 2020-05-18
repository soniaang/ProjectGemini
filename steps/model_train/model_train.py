import os
import argparse
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from azureml.core import Run

# Get Run object to log values
run = Run.get_context()


def main(args):
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
    os.makedirs(
        name="outputs",
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
    
    # Train an IsolationForest model
    print("Training an IsolationForest model")
    clf = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples if args.max_samples else "auto",
        contamination=args.contamination if args.contamination else "auto",
        max_features=args.max_features,
        bootstrap=args.bootstrap,
        random_state=args.random_state
    )
    clf.fit(df)

    # Save model
    print("Saving model to outputs")
    joblib.dump(
        value=clf,
        filename=os.path.join("outputs", args.model_name)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="model_train_args")
    parser.add_argument("--input-dataset", dest="input_dataset", type=str, help="Name of input dataset")
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Name of output dataset")
    parser.add_argument("--model-name", dest="model_name", type=str, default="model.pkl", help="Name of the model file")
    parser.add_argument("--n-estimators", dest="n_estimators", type=int, default=100, help="Number of base estimators in the ensemble of the IsolationForest model")
    parser.add_argument("--max-samples", dest="max_samples", type=int, help="Number of samples to draw from X to train each base estimator of the IsolationForest model")
    parser.add_argument("--contamination", dest="contamination", type=int, help="Amount of contamination of the data set, i.e. the proportion of outliers in the data set")
    parser.add_argument("--max-features", dest="max_features", type=int, default=1.0, help="Number of features to draw from X to train each base estimator of the IsolationForest model")
    parser.add_argument("--bootstrap", dest="bootstrap", type=bool, default=False, help="Fit tree on random subsets of the training data sampled with replacement")
    parser.add_argument("--random-state", dest="random_state", type=int, default=42, help="Random state for the IsolationForest model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args=args)