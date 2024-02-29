# Module with only necessary functions/classes for running the code on IT4Innovations,
# without importing any packages which are not needed


import os
import re
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import optuna
import warnings
from sklearn.model_selection import KFold
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


warnings.filterwarnings("ignore")


def fp_from_smiles(list_smiles):
    list_fingerprint = []
    for smi in list_smiles:
        mol = Chem.MolFromSmiles(smi)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=124)
        vector = np.array(fingerprint)
        list_fingerprint.append(vector)
    # takes a list of smiles strings,output is a corresponding Morgan fingerprint as a list
    return list_fingerprint


def parse_jazzy_df(df):
    cols = df.columns
    data = {}  # all data from csv file (i.e. mol indexes, smiles, half-lives and features)
    for col in cols:
        data[col] = list(df[col])
    nan_idxs = np.argwhere(np.isnan(data["dgtot"]))
    nan_idxs = [int(idx) for idx in nan_idxs]
    data_clumped = []  # same as data, but in the form [[idx1, smi1, thalf1, fts1], [idx2, smi2, thalf2, fts2],...]]
    for col in cols:
        for i, foo in zip(range(len(data[col])), data[col]):
            if len(data_clumped) < i+1:
                data_clumped.append([])
            data_clumped[i].append(foo)

    # remove all mols for which Jazzy features generation wasn't successful
    num_pops = 0
    for nan_idx in nan_idxs:
        data_clumped.pop(nan_idx - num_pops)
        num_pops += 1
        print(f"     removed index {nan_idx} corresponding to NaN")
    print(f"     {len(data_clumped)}, {data_clumped[0]}")

    # filter out only the features
    mol_features = np.array([feature[3:11] for feature in data_clumped])
    halflives = np.array([feature[2] for feature in data_clumped])
    smiles = np.array([feature[1] for feature in data_clumped])
    contains_nan = np.any(np.isnan(mol_features))

    return smiles, mol_features, halflives, contains_nan

def optuna_trial_logging(log_csv_path, trial_number, parameters, rmsd, predictions, std):
    # Check if the CSV file exists
    is_new_file = not os.path.isfile(log_csv_path)

    # Open the CSV file in append mode
    with open(log_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['Trial Number', 'Parameters', 'RMSD', 'Predictions', 'Std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file is newly created, write the header
        if is_new_file:
            writer.writeheader()

        # Write the data for the current trial
        writer.writerow({
            'Trial Number': trial_number,
            'Parameters': parameters,
            'RMSD': rmsd,
            'Predictions': predictions.tolist(),  # Convert numpy array to a list for CSV
            'Std': std.tolist()  # Convert numpy array to a list for CSV
        })

    # Extract only the filename using regular expressions
    file_name_match = re.search(r'[^\\/:*?"<>|\r\n]+$', log_csv_path)
    file_name = file_name_match.group() if file_name_match else log_csv_path

    # Print appropriate success message with the filename
    if is_new_file:
        print(f"Successfully created {file_name} with results of trial {trial_number} as the first entry")
    else:
        print(f"Successfully updated {file_name} with results of trial {trial_number}")

class HyperparamTuner():
    def __init__(self, log_csv_path, model_identifier, X_train, y_train, X_test, y_test):
        self.log_csv_path = log_csv_path
        self.model_identifier = model_identifier
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def sample_params(self, trial: optuna.Trial, model_identifier):
        if model_identifier == 'linear':
            fit_intercept = trial.set_user_attr("fit_intercept", True)
            alpha = trial.suggest_float('alpha', 1e-5, 1e-1)
            l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
            return {
                "fit_intercept": fit_intercept,
                "alpha": alpha,
                "l1_ratio": l1_ratio
            }, ElasticNet()

        if model_identifier == 'KRR':
            alpha = trial.suggest_float("alpha", 1e-4, 1)
            gamma = trial.suggest_float("gamma", 0, 1e-14)
            kernel = trial.suggest_categorical("kernel", ["linear", "laplacian", "rbf"])
            return {
                "alpha": alpha,
                "gamma": gamma,
                "kernel": kernel
            }, KernelRidge()

        if model_identifier == 'GB':
            n_estimators = trial.suggest_categorical("n_estimators", [10, 20, 50, 200, 500])
            learning_rate = trial.suggest_float("learning_rate", 0.005, 1)
            max_depth = trial.suggest_categorical("max_depth", [1, 2, 3, 4, 5])
            return {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth
            }, GradientBoostingRegressor()

        if model_identifier == 'RF':
            n_estimators = trial.suggest_categorical("n_estimators", [10, 20, 50, 200, 500])
            max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
            max_depth = trial.suggest_categorical("max_depth", [None, 2, 3, 4, 5, 10])
            return {
                "n_estimators": n_estimators,
                "max_features": max_features,
                "max_depth": max_depth
            }, RandomForestRegressor()

        if model_identifier == 'ANN':
            learning_rate_init = trial.suggest_float("learning_rate_init", 0.001, 0.1)
            hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes",
                                                           [[5], [10], [20], [50], [5]*2, [10]*2, [20]*2, [50]*2, [5]*3, [10]*3, [50]*3])
            return {
            "learning_rate_init": learning_rate_init,
            "hidden_layer_sizes": hidden_layer_sizes
            }, MLPRegressor()

    def cross_validation_splits(self, X_train, X_test, y_train, y_test, cv_splits=5):
        """
        Splits the data into cv_splits different combinations for cross-validation.

        Parameters:
        - X_train: Training data features
        - X_test: Testing data features
        - y_train: Training data labels
        - y_test: Testing data labels
        - cv_splits: Number of cross-validation splits

        Returns:
        - List of tuples, where each tuple contains (X_train_fold, X_test_fold, y_train_fold, y_test_fold)
        """
        # Initialize StratifiedKFold with the desired number of splits
        kf = KFold(n_splits=cv_splits, shuffle=True)  # random_state=42)

        # Initialize an empty list to store the data splits
        data_splits = []

        # Loop through the cross-validation splits
        for train_index, test_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

            # Append the current split to the list
            data_splits.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

        # Append the original test data to the list
        data_splits.append((X_train, X_test, y_train, y_test))

        return data_splits

    def evaluate(self, model, X_test, y_test, return_predictions=False):
        predictions = model.predict(X_test)
        rmsd = mean_squared_error(y_test, predictions, squared=False)
        return rmsd, predictions

    def train_test_return(self, parameters, model, trial_number):
        runs = 3
        # average over all runs
        runs_results = []
        y_tests_predicted = []

        for run in range(runs):
            validation_splits = self.cross_validation_splits(self.X_train, self.X_test, self.y_train, self.y_test)
            # average over all splits in a given run
            cv_fold_results = []
            y_test_predicted = []
            fold_num = 0

            # cross-validation
            for (X_train_val, X_test_val, y_train_val, y_test_val) in validation_splits:
                fold_num += 1
                
                # train the model on the given validation split
                model.fit(X_train_val, y_train_val)
                cv_fold_rmsd, validation_predictions = self.evaluate(model, X_test_val, y_test_val)
                
                # and save the result of that split
                cv_fold_results.append(cv_fold_rmsd)
                
                # after all five folds, append the final predictions
                if fold_num == 6:
                    y_test_predicted.append(validation_predictions)

            runs_results.append(np.mean(cv_fold_results))
            y_tests_predicted.append(y_test_predicted)

        # calculate the standard deviation of predictions
        y_tests_predicted = np.array(y_tests_predicted)
        std = np.std(y_tests_predicted, axis=0)[0]
        
        average_predictions = np.average(y_tests_predicted, axis=0)[0]
        average_result = np.mean(runs_results)
        
        # write the result and hyperparameters of a run to csv file
        optuna_trial_logging(self.log_csv_path, trial_number, parameters, average_result, average_predictions, std)

        return average_result

    def objective(self, trial=None):
        parameters, model = self.sample_params(trial, self.model_identifier)
        return self.train_test_return(parameters, model, trial.number)