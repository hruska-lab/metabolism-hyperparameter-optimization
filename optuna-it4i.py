# file for executing optuna code on it4i
import optuna
import joblib
import numpy as np
import pandas as pd
import time
from optuna.storages import JournalStorage, JournalFileStorage
from project_resources.it4i_resources import fp_from_smiles, parse_jazzy_df, HyperparamTuner


types = ["morgan", "jazzy"]
splitters = ["rand", "scaff", "time"]
model_identifiers = ["linear", "KRR", "GB", "RF", "ANN"]
isozymes = ["3A4", "RLM", "HLC"]
data_splits = ["train", "test"]
rel_paths = {
    "morgan_3A4_train_scaff": r"project_resources/base_splits/scaffold_splitter/3A4_train.csv",
    "morgan_3A4_train_rand": r"project_resources/base_splits/random/3A4_train.csv",
    "morgan_3A4_train_time": r"project_resources/base_splits/time_split/3A4_train.csv",
    "morgan_RLM_train_scaff": r"project_resources/base_splits/scaffold_splitter/RLM_train.csv",
    "morgan_RLM_train_rand": r"project_resources/base_splits/random/RLM_train.csv",
    "morgan_RLM_train_time": r"project_resources/base_splits/time_split/RLM_train.csv",
    "morgan_HLC_train_scaff": r"project_resources/base_splits/scaffold_splitter/HLC_train.csv",
    "morgan_HLC_train_rand": r"project_resources/base_splits/random/HLC_train.csv",
    "morgan_HLC_train_time": r"project_resources/base_splits/time_split/HLC_train.csv",

    "morgan_3A4_test_scaff": r"project_resources/base_splits/scaffold_splitter/3A4_test.csv",
    "morgan_3A4_test_rand": r"project_resources/base_splits/random/3A4_test.csv",
    "morgan_3A4_test_time": r"project_resources/base_splits/time_split/3A4_test.csv",
    "morgan_RLM_test_scaff": r"project_resources/base_splits/scaffold_splitter/RLM_test.csv",
    "morgan_RLM_test_rand": r"project_resources/base_splits/random/RLM_test.csv",
    "morgan_RLM_test_time": r"project_resources/base_splits/time_split/RLM_test.csv",
    "morgan_HLC_test_scaff": r"project_resources/base_splits/scaffold_splitter/HLC_test.csv",
    "morgan_HLC_test_rand": r"project_resources/base_splits/random/HLC_test.csv",
    "morgan_HLC_test_time": r"project_resources/base_splits/time_split/HLC_test.csv",

    "jazzy_3A4_train_scaff": r"project_resources/jazzy_splits/scaffold_splitter/3A4_train.csv",
    "jazzy_3A4_train_rand": r"project_resources/jazzy_splits/random/3A4_train.csv",
    "jazzy_3A4_train_time": r"project_resources/jazzy_splits/time_split/3A4_train.csv",
    "jazzy_RLM_train_scaff": r"project_resources/jazzy_splits/scaffold_splitter/RLM_train.csv",
    "jazzy_RLM_train_rand": r"project_resources/jazzy_splits/random/RLM_train.csv",
    "jazzy_RLM_train_time": r"project_resources/jazzy_splits/time_split/RLM_train.csv",
    "jazzy_HLC_train_scaff": r"project_resources/jazzy_splits/scaffold_splitter/HLC_train.csv",
    "jazzy_HLC_train_rand": r"project_resources/jazzy_splits/random/HLC_train.csv",
    "jazzy_HLC_train_time": r"project_resources/jazzy_splits/time_split/HLC_train.csv",

    "jazzy_3A4_test_scaff": r"project_resources/jazzy_splits/scaffold_splitter/3A4_test.csv",
    "jazzy_3A4_test_rand": r"project_resources/jazzy_splits/random/3A4_test.csv",
    "jazzy_3A4_test_time": r"project_resources/jazzy_splits/time_split/3A4_test.csv",
    "jazzy_RLM_test_scaff": r"project_resources/jazzy_splits/scaffold_splitter/RLM_test.csv",
    "jazzy_RLM_test_rand": r"project_resources/jazzy_splits/random/RLM_test.csv",
    "jazzy_RLM_test_time": r"project_resources/jazzy_splits/time_split/RLM_test.csv",
    "jazzy_HLC_test_scaff": r"project_resources/jazzy_splits/scaffold_splitter/HLC_test.csv",
    "jazzy_HLC_test_rand": r"project_resources/jazzy_splits/random/HLC_test.csv",
    "jazzy_HLC_test_time": r"project_resources/jazzy_splits/time_split/HLC_test.csv"
}
# sampler - a method used to generate new sets of hyperparameters in each iteration of the optimization process
samplers = {
    'RandomSampler': optuna.samplers.RandomSampler,          # Sampler that selects hyperparameters randomly from the search space.
    'GridSampler': optuna.samplers.GridSampler,              # Sampler that performs a grid search over the hyperparameter space.
    'TPESampler': optuna.samplers.TPESampler,                # Sampler that uses a tree-structured Parzen estimator to model the objective function and sample new points from the search space.
    'CmaEsSampler': optuna.samplers.CmaEsSampler,            # Sampler that uses the Covariance Matrix Adaptation Evolution Strategy algorithm to efficiently search the hyperparameter space.
    'NSGAIISampler': optuna.samplers.NSGAIISampler,          # Multi-objective evolutionary algorithm that generates new samples using non-dominated sorting and crowding distance selection.
    'QMCSampler': optuna.samplers.QMCSampler,                # Quasi-Monte Carlo sampler that uses low-discrepancy sequences to sample the search space in a more efficient and evenly distributed way than random sampling.
    'BoTorchSampler': optuna.integration.BoTorchSampler,     # Sampler that leverages the BoTorch library for Bayesian optimization and can handle both continuous and categorical hyperparameters.
    'BruteForceSampler': optuna.samplers.BruteForceSampler,  # Sampler that exhaustively evaluates all possible combinations of hyperparameters in the search space.
}
# pruner - a technique used to eliminate unpromising trials during the course of hyperparameter optimization.
pruners = {
    'BasePruner': optuna.pruners.BasePruner,                            # This is the base class for all pruning strategies in Optuna. It provides a skeleton for implementing custom pruning strategies.
    'MedianPruner': optuna.pruners.MedianPruner,                        # A pruner that prunes unpromising trials that have median objective values, as determined in previous steps.
    'SuccessiveHalvingPruner': optuna.pruners.SuccessiveHalvingPruner,  # This pruner repeatedly splits trials into halves, discarding the lower performing half at each iteration.
    'HyperbandPruner': optuna.pruners.HyperbandPruner,                  # This pruner implements the Hyperband algorithm, which selects promising trials and runs them with different resource allocation schemes to determine the best one.
    'PercentilePruner': optuna.pruners.PercentilePruner,                # A pruner that prunes unpromising trials based on their percentile rank relative to all completed trials.
    'NopPruner': optuna.pruners.NopPruner,                              # A pruner that does nothing and does not prune any trials.
    'ThresholdPruner': optuna.pruners.ThresholdPruner,                  # This pruner prunes trials that have not reached a certain level of performance (i.e., objective value).
    'PatientPruner': optuna.pruners.PatientPruner,                      # This pruner prunes trials that do not show improvement over a certain number of steps (or epochs).
}
smiles = {}
halflives = {}
features = {}


# load smiles used for ML with Morgan features
smiles["morgan"] = {}
halflives["morgan"] = {}
for splitter in splitters:
    smiles["morgan"][splitter] = {}
    halflives["morgan"][splitter] = {}
    for isozyme in isozymes:
        smiles["morgan"][splitter][isozyme] = {}
        halflives["morgan"][splitter][isozyme] = {}
        for split in data_splits:
            df = pd.read_csv(rel_paths[f"morgan_{isozyme}_{split}_{splitter}"])
            df_smiles = list(df["smiles"])
            df_halflives = list(df["half-life"])
            smiles["morgan"][splitter][isozyme][split] = df_smiles
            halflives["morgan"][splitter][isozyme][split] = df_halflives

# smiles to Morgan fingerprint
features["morgan"] = {}  # need to destinguish between Jazzy and Morngan since Jazzy ommits some mols
for splitter in splitters:
    features["morgan"][splitter] = {}
    for isozyme in isozymes:
        features["morgan"][splitter][isozyme] = {}
        for data_split in data_splits:
            fps = fp_from_smiles(smiles["morgan"][splitter][isozyme][data_split])
            features["morgan"][splitter][isozyme][data_split] = np.array(fps)

# load Jazzy features from csv files and their corresponding smiles
smiles["jazzy"] = {}
halflives["jazzy"] = {}
features["jazzy"] = {}
for splitter in splitters:
    features["jazzy"][splitter] = {}
    smiles["jazzy"][splitter] = {}
    halflives["jazzy"][splitter] = {}
    for isozyme in isozymes:
        features["jazzy"][splitter][isozyme] = {}
        smiles["jazzy"][splitter][isozyme] = {}
        halflives["jazzy"][splitter][isozyme] = {}
        for split in data_splits:
            df = pd.read_csv(rel_paths[f"jazzy_{isozyme}_{split}_{splitter}"])
            jazzy_smiles, df_features, thalfs, contains_nan = parse_jazzy_df(df)
            smiles["jazzy"][splitter][isozyme][split] = jazzy_smiles
            features["jazzy"][splitter][isozyme][split] = df_features
            halflives["jazzy"][splitter][isozyme][split] = thalfs

sampler = samplers['TPESampler']
pruner = pruners["BasePruner"]
t_end = time.time() + (60 * 60 * 24)
while time.time() < t_end:
    # while loop is needed; if instead n_trials was large only one model would be trained
    n_trials = 5
    for _type in types:
        for splitter in splitters:
            if splitter == "rand":
                splitter_name = "random"
            elif splitter == "scaff":
                splitter_name = "scaffold_splitter"
            else:
                splitter_name = "time_split"

            for isozyme in isozymes:
                X_train = features[_type][splitter][isozyme]["train"]
                y_train = np.array(halflives[_type][splitter][isozyme]["train"])
                X_test = features[_type][splitter][isozyme]["test"]
                y_test = np.array(halflives[_type][splitter][isozyme]["test"])

                for model_identifier in model_identifiers:
                    print(splitter_name, isozyme, model_identifier)
                    lock_obj = optuna.storages.JournalFileOpenLock(f"./project_resources/optuna/{_type}/{splitter_name}/{isozyme}/{model_identifier}_journal.log")

                    storage = JournalStorage(
                        JournalFileStorage(f"./project_resources/optuna/{_type}/{splitter_name}/{isozyme}/{model_identifier}_journal.log", lock_obj=lock_obj)
                    )
                    study = optuna.create_study(study_name=model_identifier, directions=['minimize'], pruner=pruner,
                                                storage=storage, load_if_exists=True)
                    tuner = HyperparamTuner(model_identifier, X_train, y_train, X_test, y_test)
                    study.optimize(tuner.objective, n_trials=n_trials, n_jobs=-1)  # catch=(ValueError,)
                    joblib.dump(study, f"./project_resources/optuna/{_type}/{splitter_name}/{isozyme}/{model_identifier}.pkl")