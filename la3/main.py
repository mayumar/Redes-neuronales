# TODO: Load the necessary libraries

from rbf import RBFNN
import click
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split


@click.command()
@click.option(
    "--dataset_filename",
    "-d",
    default=None,
    required=True,
    help="Name of the file with training data.",
    type=str,
)
@click.option(
    "--standarize",
    "-s",
    is_flag=True,
    help="Standardize input variables.",
)
@click.option(
    "--classification",
    "-c",
    is_flag=True,
    help="Use classification instead of regression.",
)
@click.option(
    "--ratio rbf",
    "-r",
    default=0.1,
    show_default=True,
    help="Ratio of RBFs with respect to the total number of patterns.",
    type=float,
)
@click.option(
    "--l2",
    "-l",
    is_flag=True,
    help="Use L2 regularization for logistic regression.",
)
@click.option(
    "--eta",
    "-e",
    default=0.01,
    show_default=True,
    help="Value of the regularization factor for logistic regression.",
    type=float,
)
@click.option(
    "--fairness",
    "-f",
    is_flag=True,
    help="Calculate fairness metrics.",
)
@click.option(
    "--outputs",
    "-o",
    default=1,
    show_default=True,
    help="Number of output columns in the dataset (always at the end).",
    type=int,
)
@click.option(
    "--logisticcv",
    "-v",
    is_flag=True,
    help="Use LogisticRegressionCV.",
)
@click.option(
    "--seeds",
    "-n",
    default=5,
    show_default=True,
    help="Number of seeds to use.",
    type=int,
)
@click.option(
    "--model_filename",
    "-m",
    default="",
    show_default=True,
    help="Directory name to save the models (or name of the file to load the model, if "
    "the prediction mode is active).",
    type=str,
)  # KAGGLE
@click.option(
    "--pred",
    "-p",
    default=None,
    show_default=True,
    help="Specifies the seed used to predict the output of the dataset_filename.",
    type=int,
)  # KAGGLE
def main(
    dataset_filename: str,
    standarize: bool,
    classification: bool,
    ratio_rbf: float,
    l2: bool,
    eta: float,
    fairness: bool,
    logisticcv: bool,
    seeds: int,
    model_filename: str,
    pred: int,
):
    """
    Run several executions of RBFNN training and testing.

    RBF neural network based on hybrid supervised/unsupervised training. Every run uses
    a different seed for the random number generator. The results of the training and
    testing are stored in a pandas DataFrame.

    Parameters
    ----------
    dataset_filename: str
        Name of the data file
    standarize: bool
        True if we want to standarize input variables (and output ones if
          it is regression)
    classification: bool
        True if it is a classification problem
    ratio_rbf: float
        Ratio (as a fraction of 1) indicating the number of RBFs
        with respect to the total number of patterns
    l2: bool
        True if we want to use L2 regularization for logistic regression
        False if we want to use L1 regularization for logistic regression
    eta: float
        Value of the regularization factor for logistic regression
    fairness: bool
        False. If set to true, it will calculate fairness metrics on the prediction
    logisticcv: bool
        True if we want to use LogisticRegressionCV
    seeds: int
        Number of seeds to use
    model_filename: str
        Name of the directory where the models will be written. Note that it will create
        a directory with the name of the dataset file and the seed number
    pred: int
        If used, it will predict the output of the dataset_filename using the model
        stored in model_filename with the seed indicated in this parameter
    """

    # check that when logisticcv is set to True, eta is not included
    if logisticcv and eta != 0.01:
        raise ValueError("You cannot use eta when logisticcv is set to True.")

    # TODO: Complete with at least 10 checks to ensure that the parameters are correct

    # Validaciones de parámetros
    if not 0 < ratio_rbf <= 1:
        raise ValueError("The ratio_rbf parameter must be between 0 and 1.")

    if seeds <= 0:
        raise ValueError("The number of seeds (--seeds) must be greater than 0.")

    if fairness and not classification:
        raise ValueError("Fairness evaluation is only applicable to classification problems.")

    if pred is not None and not model_filename:
        raise ValueError("Model filename (-m) must be specified for prediction mode.")

    if pred is not None and (pred < 0 or pred >= seeds):
        raise ValueError("Pred seed must be within the range of 0 to (seeds - 1).")

    if eta <= 0:
        raise ValueError("The eta parameter (--eta) must be greater than 0.")

    if not dataset_filename.endswith(".csv"):
        raise ValueError("The dataset file must be a CSV.")


    results = []

    seeds_list = range(seeds)
    dataset_name = dataset_filename.split("/")[-1].split(".")[0]

    if pred is not None:
        seeds_list = [pred]

    for random_state in seeds_list:
        print(f"Running on {dataset_name} - seed: {random_state}.")
        np.random.seed(random_state)

        # TODO: Read the data
        data = read_data(dataset_filename, standarize, random_state, classification, fairness, pred)

        if not fairness and pred is None:
            X_train, y_train, X_test, y_test = data
        elif pred is not None:
            X_train, y_train, X_test, y_test, X_test_kaggle = data
        elif fairness:
            (
                X_train,
                y_train,
                X_test,
                y_test,
                X_train_disc,
                X_test_disc,
            ) = data

        if pred is None:  # Train the model
            # TODO: Create the object

            # TODO: Train the model

            if model_filename:
                dir_name = f"{model_filename}/{dataset_name}/{random_state}.p"
                save(rbf, dir_name)
                print(f"Model saved in {dir_name}")

        else:  # Load the model from file
            dir_name = f"{model_filename}/{dataset_name}/{random_state}.p"
            rbf = load(dir_name, random_state)

        # TODO: Predict the output using the trained model

        if pred is not None:
            preds_kaggle = rbf.predict(X_test_kaggle)
            dir_name = f"{model_filename}/{dataset_name}/predictions_{pred}.csv"
            # include index in the first column from 0 to length preds_test
            preds_kaggle_with_index = np.column_stack(
                (np.arange(1, len(preds_kaggle)+1), preds_kaggle)
            )
            np.savetxt(
                dir_name,
                preds_kaggle_with_index,
                delimiter=",",
                header="ID,survived",
                comments="",
                fmt="%d",
            )
            print(f"Predictions saved in {dir_name}.")

        train_results_per_seed = {
            "seed": random_state,
            "partition": "Train",
            "MSE": mean_squared_error(y_train, preds_train),
        }
        test_results_per_seed = {
            "seed": random_state,
            "partition": "Test",
            "MSE": mean_squared_error(y_test, preds_test),
        }

        if classification:
            train_results_per_seed["CCR"] = accuracy_score(y_train, preds_train) * 100
            test_results_per_seed["CCR"] = accuracy_score(y_test, preds_test) * 100

        # Fairness evaluation
        if fairness:

            # TODO: Calculate the first fairness metric
            train_results_per_seed["FN0"] = train_fn.by_group["White"] * 100
            train_results_per_seed["FN1"] = train_fn.by_group["Black"] * 100
            test_results_per_seed["FN0"] = test_fn.by_group["White"] * 100
            test_results_per_seed["FN1"] = test_fn.by_group["Black"] * 100

            # TODO: Calculate the second fairness metric
            train_results_per_seed["FP0"] = train_fp.by_group["White"] * 100
            train_results_per_seed["FP1"] = train_fp.by_group["Black"] * 100
            test_results_per_seed["FP0"] = test_fp.by_group["White"] * 100
            test_results_per_seed["FP1"] = test_fp.by_group["Black"] * 100

        results.append(train_results_per_seed)
        results.append(test_results_per_seed)

    results = pd.DataFrame(results)
    if pred is None:
        metrics = results.columns[2:]

        mean_std = []

        # TODO: Calculate the mean and standard deviation of the metrics and add them to the
        #  results DataFrame

    results.set_index(["seed", "partition"], inplace=True)

    print("******************")
    print("Summary of results")
    print("******************")

    print(results)


def save(
    model: RBFNN,
    dir_name: str,
) -> None:
    """
    Save the model to a file

    Parameters
    ----------
    model: RBFNN
        Model to be saved
    dir_name: str
        Name of the file where the model will be saved
    """

    dir = os.path.dirname(dir_name)
    if not os.path.isdir(dir):
        os.makedirs(dir)

    with open(dir_name, "wb") as f:
        pickle.dump(model, f)


def load(dir_name: str, random_state: int) -> RBFNN:
    """
    Load the model from the file

    Parameters
    ----------
    dir_name: str
        Name of the model file
    random_state: int
        Seed for the random number generator

    Returns
    -------
    model: RBFNN
        Model loaded from the file
    """

    if not os.path.exists(dir_name):
        raise ValueError(
            f"The model file {dir_name} does not exist.\n"
            f"You can create it by firstly using the parameter (n = {random_state}) and"
            f" removing the flag P (for pred) to train the model."
        )
    with open(dir_name, "rb") as f:
        self = pickle.load(f)
    return self


def read_data(
    dataset_filename: str,
    standarize: bool,
    random_state: int,
    classification: bool = False,
    fairness: bool = False,
    prediction_mode: bool = False,
) -> (
    tuple[np.array, np.array, np.array, np.array]
    | tuple[np.array, np.array, np.array, np.array, np.array, np.array]
    | tuple[np.array, np.array, np.array, np.array, np.array]
):
    """
    Read the input data

    It receives the name of the dataset file and returns the corresponding matrices.

    Parameters
    ----------
    dataset_filename: str
        Name of the dataset file
    standarize: bool
        True if we want to standarize input variables (and output ones if
          it is regression)
    random_state: int
        Seed for the random number generator
    classification: bool
        True if it is a classification problem
    fairness: bool
        True if we want to calculate fairness metrics. The discriminative attribute
        is assumed to be the last column of the input data.
    prediction_mode: bool
        True if we are in prediction mode. This is to load the data for Kaggle.

    Returns
    -------
    X_train: array, shape (n_train_patterns,n_inputs)
        Matrix containing the inputs for the training patterns
    y_train: array, shape (n_train_patterns,n_outputs)
        Matrix containing the outputs for the training patterns
    X_test: array, shape (n_test_patterns,n_inputs)
        Matrix containing the inputs for the test patterns
    y_test: array, shape (n_test_patterns,n_outputs)
        Matrix containing the outputs for the test patterns
    X_test_kaggle: array, shape (n_test_patterns_kaggle,n_inputs)
        Matrix containing the inputs for the test patterns (only if prediction_mode is
        set to True)
    X_train_disc: array, shape (n_train_patterns,)
        Array containing the discriminative attribute for the training patterns (only
        if fairness is set to True)
    X_test_disc: array, shape (n_test_patterns,)
        Array containing the discriminative attribute for the testing patterns (only
        if fairness is set to True)
    """
    data = pd.read_csv(dataset_filename, header=None)

    n_features = data.shape[1] - 1
    X = data.iloc[:, :n_features].values
    y = data.iloc[:, n_features:].values

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y if classification else None
    )

    if fairness:
        # Group label (we assume it is in the last column of X)
        # 1 women / 0 men
        lu = np.unique(X_train[:, -1])
        X_train_disc = np.where(X_train[:, -1] == lu[1], "Black", "White")
        X_test_disc = np.where(X_test[:, -1] == lu[1], "Black", "White")

        return X_train, y_train, X_test, y_test, X_train_disc, X_test_disc

    if prediction_mode is not None:
        return X_train, y_train, X_test, y_test, X_test_kaggle  # KAGGLE

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    main()
