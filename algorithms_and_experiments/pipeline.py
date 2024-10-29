import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from alignment_predictor import NUM_OF_ARTISTS
from conformalized_selection import *
from evaluation import *

# End-to-end experiment pipeline

def string_to_numpy_array(s):
    s = s.strip('[]')
    return np.fromstring(s, sep=' ')


def prepare_df_for_train(df, target_column):
    # split vectors into seperate columns
    df['ground_truth_artist_vector'] = df['ground_truth_artist_vector'].apply(string_to_numpy_array)
    df['predicted_prob_artists_vector'] = df['predicted_prob_artists_vector'].apply(string_to_numpy_array)
    gtav_split = pd.DataFrame(df['ground_truth_artist_vector'].tolist(), columns=[f'gtav_{i+1}' for i in range(NUM_OF_ARTISTS)])
    ppav_split = pd.DataFrame(df['predicted_prob_artists_vector'].tolist(), columns=[f'ppav_{i+1}' for i in range(NUM_OF_ARTISTS + 1)])

    X_data = pd.concat([gtav_split, ppav_split, df['shared_elements_count']], axis=1)
    y_data = df[[target_column]]
    return X_data, y_data 


def experiment_pipeline(alpha, c, seed, target_column):
    table_path = "data.csv"
    df = pd.read_csv(table_path)
    
    # data preparation
    X_data, y_data = prepare_df_for_train(df, target_column)
    X_ref, X_test, y_ref, y_test = train_test_split(X_data, y_data, test_size=0.45, random_state=seed)
    X_train, X_calib, y_train, y_calib = train_test_split(X_ref, y_ref, test_size=0.6, random_state=seed)
    dtrain_reg = xgb.DMatrix(X_train, y_train)
    dcalib_reg = xgb.DMatrix(X_calib, y_calib)
    dtest_reg = xgb.DMatrix(X_test, y_test)

    # model training
    hyperparams = {"objective": "reg:logistic"}
    alignment_predictor = xgb.train(params=hyperparams, dtrain=dtrain_reg, num_boost_round=100)

    # predict A_hat
    calib_set_A_hat = alignment_predictor.predict(dcalib_reg)
    test_set_A_hat = alignment_predictor.predict(dtest_reg)

    # compute p-values and perform BH selection
    rejection_set_indices = conformalized_selection(y_calib.to_numpy().flatten(), calib_set_A_hat, test_set_A_hat, alpha, c)

    # evaluation
    fdp_value = fdp(y_test.to_numpy().flatten(), rejection_set_indices, c)
    power_value = power(y_test.to_numpy().flatten(), rejection_set_indices, c)
    selected_indices = X_test.iloc[rejection_set_indices]
    selected_indices = selected_indices.index.values.tolist()
    total_aligned_selected_units = df.iloc[selected_indices][target_column].to_numpy()
    total_aligned_selected_units = np.where(total_aligned_selected_units > c, 1, 0).sum()
    total_aligned_test_units = y_test.to_numpy().flatten()
    total_aligned_test_units = np.where(total_aligned_test_units > c, 1, 0).sum()
    mse = mean_squared_error(y_test, test_set_A_hat, squared=False)

    print(f"Alpha: {alpha}, FDP: {fdp_value}, Power: {power_value}")
    print(f"Selected units: {rejection_set_indices.sum()} out of {y_test.shape[0]}")
    print(f"Selected indices: {selected_indices}")
    print(f"{total_aligned_selected_units} truely aligned units out of {total_aligned_test_units}")
    print(f"Alignment predictor test error: {mse:.3f}")

    return selected_indices, fdp_value, power_value
