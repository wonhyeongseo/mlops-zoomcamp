import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
mlflow.set_experiment("week2_initial_baseline") # choose a name for your experiment

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--model_path",
    default="./models/lin_reg.bin",
    help="Location where the trained model will be saved"
)
def run_train(data_path: str, model_path: str):
    mlflow.autolog()
    with mlflow.start_run():
        mlflow.set_tag("developer", "wonhseo")

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        with open(model_path, 'wb') as f_out:
            dv = load_pickle(os.path.join(data_path, "dv.pkl"))
            try:
                # Pickle both the dictionary vectorizer and the random forest model
                pickle.dump((dv, rf), f_out)
                print("Model successfully pickled.")
            except Exception as e:
                print("Error occurred while pickling the model:", str(e))

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)
        print(rmse)

        mlflow.log_artifact(model_path)

if __name__ == '__main__':
    run_train()