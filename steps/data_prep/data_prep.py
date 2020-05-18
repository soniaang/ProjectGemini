import argparse, os
import pandas as pd

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

    # Check input path and load other parameter if no dataset was provided
    if not input_path:
        input_path = args.input_datapath
    if not input_path:
        print("No input data provided")
        raise ValueError()

    # Create output path
    print("Creating output path")
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

    # Pre-process data
    print("Pre-processing data")
    df_final = pd.DataFrame()
    for path in paths:
        tag_name = os.path.split(path)[-2]
        # Read data
        print(f"Reading parquet file: {tag_name}")
        df = pd.read_parquet(
            path=path,
            engine="auto"
        )

        # Reset index
        print("Resetting index")
        df.set_index("ts_utc", inplace=True)

        # Resample data
        print(f"Resampling data of tag: {tag_name}")
        tag_resampled = df["tag_value"].resample(
            rule="30S",
            axis=0,
            label="left"
        ).agg("mean")

        # Merge data
        print("Merging data to final dataframe")
        df_tag_resampled = pd.DataFrame(data={tag_name: tag_resampled})
        if len(df_final) == 0:
            df_final = df_tag_resampled
        else:
            df_final = df_final.merge(
                df_tag_resampled,
                left_index=True,
                right_index=True
            )
    
    # Fill NaN
    print("Fill NaN")
    df_final.fillna(
        method="ffill",
        inplace=True
    )
    
    # Save parquet for training
    print("Saving Parquet file for training")
    df_final.to_parquet(
        path=os.path.join(output_path, "train.parquet")
    )


def parse_args():
    parser = argparse.ArgumentParser(description="data_prep_args")
    parser.add_argument("--input-dataset", dest="input_dataset", type=str, help="Name of dataset")
    parser.add_argument("--input-datapath", dest="input_datapath", type=str, help="Datapath for variable input for inference")
    parser.add_argument("--output-dataset", dest="output_dataset", type=str, help="Name of dataset")
    parser.add_argument("--output-path", dest="output_path", type=str, help="Name of dataset")
    parser.add_argument("--environment", dest="environment", type=str, help="Name of dataset")
    parser.add_argument("--system", dest="system", type=str, help="Name of dataset")
    parser.add_argument("--platform", dest="platform", type=str, help="Name of dataset")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
