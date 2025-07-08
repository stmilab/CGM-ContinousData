"""
This is a copy of the CGMacros Joint Emedding dataset loader
"""

import pandas as pd
import os, pdb, copy
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import socket
import warnings

warnings.filterwarnings("ignore")


def get_dir_path_by_hostname():
    hostname = socket.gethostname()
    if hostname == "vita5.engr.tamu.edu":
        CGMacros_dir_path = "/scratch/CGMacros/cgmacros/anurag_corrected/CGMacros/"
    elif hostname == "csce-stmi-s1.engr.tamu.edu":
        CGMacros_dir_path = "/home/grads/a/atkulkarni/CGM-ContinousData/CGMacros-2/"
        fitbit_dir_path = "/home/grads/a/atkulkarni/CGM-ContinousData/intensityData/"
    else:
        raise ValueError(f"Unknown hostname: {hostname}")
    return CGMacros_dir_path, fitbit_dir_path


CGMacros_dir_path, fitbit_dir_path = get_dir_path_by_hostname()

meal_event_cols = [
    "Calories",
    "Carbs",
    "Protein",
    "Fat",
    "Fiber",
    "Image path",
    "Meal Type",
]
time_series_cols = ["Libre GL", "Dexcom GL", "HR"]
img_size: tuple = (112, 112)
tqdm.write(CGMacros_dir_path)


# NOTE: Not used yet, as this is 24H experiment
def get_subject_a1c_and_fg(
    subject_id: int,
    csv_path: str = f"{CGMacros_dir_path}/bio.csv",
) -> pd.DataFrame:
    demographics_df = pd.read_csv(csv_path, index_col=None)
    demographics_df = demographics_df[demographics_df["subject"] == subject_id]
    a1c = demographics_df["A1c PDL (Lab)"].item()
    fasting_glucose = demographics_df["Fasting GLU - PDL (Lab)"].item()
    return a1c, fasting_glucose


def load_CGMacros(
    subject_id: int,
    csv_dir: str = CGMacros_dir_path,
) -> pd.DataFrame:
    if type(subject_id) != int:
        tqdm.write("subject_id should be an integer")
        raise ValueError
    subject_path = f"CGMacros-{subject_id:03d}/CGMacros-{subject_id:03d}.csv"
    subject_file = os.path.join(csv_dir, subject_path)
    if not os.path.exists(subject_file):
        tqdm.write(f"File {subject_file} not found")
        raise FileNotFoundError
    dataset_df = pd.read_csv(subject_file, index_col=None)
    tqdm.write(str(sorted(dataset_df.columns)))
    dataset_df["Timestamp"] = pd.to_datetime(dataset_df["Timestamp"])
    dataset_df = dataset_df.drop("Unnamed: 0", axis=1, errors="ignore")
    return dataset_df.set_index("Timestamp")


def get_valid_meal_idx(
    dataset_df: pd.DataFrame,
    meal_types: list = ["breakfast", "lunch"],
    ignore_x_hours: int = 0,  # could be configured to skip first meal
    col_names=meal_event_cols,  # columns to check for NaN values
) -> pd.DatetimeIndex:
    # NOTE: only timestamps with valid meal types and Image paths are considered
    ts_df = copy.deepcopy(dataset_df).dropna(subset=col_names)
    # NOTE: making the all meal types lowercases
    ts_df["Meal Type"] = ts_df["Meal Type"].str.lower()
    meal_types = [meal.lower() for meal in meal_types]
    if ignore_x_hours > 0:
        start_time = ts_df.index[0] + pd.Timedelta(hours=ignore_x_hours)
        ts_df = ts_df[start_time:]
    valid_meal_idx = ts_df[ts_df["Meal Type"].isin(meal_types)].index
    return valid_meal_idx


def extract_meal_trace(
    dataset_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    pre_hours: int = 0,
    post_hours: int = 3,
) -> pd.DataFrame:
    start_time = timestamp - pd.Timedelta(hours=pre_hours) + pd.Timedelta(minutes=1)
    end_time = timestamp + pd.Timedelta(hours=post_hours)
    trace_df = dataset_df[start_time:end_time]
    return trace_df


def get_image(
    img_filename: str,
    subject_id: int,
    target_size: tuple,
    cgmacros_path: str = CGMacros_dir_path,
) -> np.ndarray:
    subject_path = f"CGMacros-{subject_id:03d}/"
    img_path = f"{cgmacros_path}{subject_path}{img_filename}"
    if not os.path.exists(img_path):
        tqdm.write(f"File {img_path} not found")
        raise FileNotFoundError
    # Loading names out
    img_data = cv2.resize(
        cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
        target_size,
        interpolation=cv2.INTER_LANCZOS4,
    )
    return img_data


def load_img_data(
    img_df: pd.DataFrame,
    subject_id: int,
    target_size: tuple = (112, 112),
):
    img_data = np.zeros((len(img_df), target_size[0], target_size[1], 3))
    for j, img_filename in enumerate(img_df["Image path"]):
        img_data[j] = get_image(img_filename, subject_id, target_size)
    return img_data


def get_split_pids(
    unique_subjects: list,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 2025,
):
    # First split into train+val and test
    train_val_pids, test_pids = train_test_split(
        unique_subjects,
        test_size=test_size,
        random_state=random_state,
    )

    # Further split train+val into train and validation
    train_pids, val_pids = train_test_split(
        train_val_pids,
        test_size=val_size
        / (1 - test_size),  # Adjust validation size relative to train+val
        random_state=random_state,
    )
    return train_pids, val_pids, test_pids


def get_subset_by_pids(
    time_series_trace_df: pd.DataFrame,
    meal_event_df: pd.DataFrame,
    heatmap_fnames: np.ndarray,
    pids: list,
):
    # Split the data into train, validation, and test sets
    trace = time_series_trace_df[time_series_trace_df["PID"].isin(pids)]
    events = meal_event_df[meal_event_df["PID"].isin(pids)]
    fnames = heatmap_fnames[pids]
    return trace, events, fnames


def generate_24H_CGMacros_dataset(
    save_dir: str = "continuous24H_dataset/",
    regenerate: bool = False,
):
    if os.path.exists(f"{save_dir}cgmacros_time_series_trace.csv") and not regenerate:
        time_series_trace_df = pd.read_csv(
            f"{save_dir}cgmacros_time_series_trace.csv", index_col=0
        )
        meal_event_df = pd.read_csv(
            f"{save_dir}cgmacros_meal_event.csv",
            index_col=0,
        )
        return time_series_trace_df, meal_event_df
    time_series_trace_df = pd.DataFrame()
    meal_event_df = pd.DataFrame()
    unique_subjects = [int]
    for i in tqdm(
        range(1, 50, 1),
        desc="Processing Subjects",
        ascii=True,
    ):  # IDs are from 1 to 49
        try:
            dataset_df = load_CGMacros(subject_id=i)  # loading which subject
        except FileNotFoundError:
            tqdm.write(f"Skipping Subject {i:03d}")
            continue
        unique_subjects.append(i)
        # NOTE: getting the list of timestamps of valid meals
        valid_meal_idx = get_valid_meal_idx(
            dataset_df, meal_types=["breakfast"], ignore_x_hours=0
        )
        # NOTE: Extracting macronutritient labels
        meal_event_series = dataset_df.loc[valid_meal_idx][meal_event_cols]
        meal_event_series["PID"] = i  # Adding subject ID to the labels
        # NOTE: Extracting the time-series trace entirely
        time_series_trace_series = dataset_df[time_series_cols]
        time_series_trace_series["PID"] = i  # Adding subject ID to the labels
        tqdm.write(
            f"Subject ID: {i}, 24H Features: {time_series_trace_series}, "
            f"Label Data: {meal_event_series.shape}"
        )
        time_series_trace_df = pd.concat(
            [time_series_trace_df, time_series_trace_series]
        )
        meal_event_df = pd.concat([meal_event_df, meal_event_series])

    tqdm.write(f"All 24H Features: {time_series_trace_df.shape}")
    tqdm.write(f"All meal event Data: {meal_event_df.shape}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    time_series_trace_df.to_csv(f"{save_dir}cgmacros_time_series_trace.csv")
    meal_event_df.to_csv(f"{save_dir}cgmacros_meal_event.csv")
    tqdm.write(f"Saved to {save_dir}")
    return time_series_trace_df, meal_event_df


def main():
    generate_24H_CGMacros_dataset(regenerate=True)


if __name__ == "__main__":
    main()
