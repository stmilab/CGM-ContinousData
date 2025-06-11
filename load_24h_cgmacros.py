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

label_cols = ["Calories", "Carbs", "Protein", "Fat", "Fiber"]
img_cols = ["Image path"]


# TODO: complete this function
class CGMacros24HDataset(Dataset):
    def __init__(
        self,
        cgm_trace,  # CGM features in multimendional array
        znorm_cgm,  # z-normalized CGM features
        img_arr,
        a1c,  # A1C values
        fasting_glucose,  # fasting glucose values
        label_arr,  # macronutrient labels
        transform=None,  # transform function to apply on CGM features
    ):
        self.cgm_trace = cgm_trace
        self.znorm_cgm = znorm_cgm
        self.img_arr = img_arr
        self.a1c = a1c
        self.fasting_glucose = fasting_glucose
        self.label_arr = label_arr
        self.label_cols = label_cols
        if transform is None:
            self.transform = self.zscore_transform
        else:
            self.transform = transform

    def __len__(self):
        return len(self.cgm_trace)

    def __getitem__(self, idx):
        cgm_trace = self.cgm_trace[idx]
        znorm_cgm = self.znorm_cgm[idx]
        img_arr = self.img_arr[idx]
        a1c = self.a1c[idx]
        fasting_glucose = self.fasting_glucose[idx]
        label_arr = self.label_arr[idx]

        if self.transform:
            # cgm_trace = self.transform(cgm_trace)
            img_arr = self.transform(img_arr)
        return (
            img_arr,
            cgm_trace,
            znorm_cgm,
            a1c,
            fasting_glucose,
            label_arr,
            self.label_cols,
        )

    def zscore_transform(self, orig_feat):
        mean = np.nanmean(orig_feat, axis=1, keepdims=True)
        std = np.nanstd(orig_feat, axis=1, keepdims=True) + 1e-8
        norm_feat = (orig_feat - mean) / (std + 1e-8)
        return norm_feat


# NOTE: Not used yet, as this is 24H experiment
def get_subject_a1c_and_fg(
    subject_id: int,
    csv_path: str = "/scratch/CGMacros/cgmacros/1.0.0/CGMacros/bio.csv",
) -> pd.DataFrame:
    demographics_df = pd.read_csv(csv_path, index_col=None)
    demographics_df = demographics_df[demographics_df["subject"] == subject_id]
    a1c = demographics_df["A1c PDL (Lab)"].item()
    fasting_glucose = demographics_df["Fasting GLU - PDL (Lab)"].item()
    return a1c, fasting_glucose


def load_CGMacros(
    subject_id: int,
    csv_dir: str = "/scratch/CGMacros/cgmacros/1.0.0/CGMacros/",
) -> pd.DataFrame:
    if type(subject_id) != int:
        print("subject_id should be an integer")
        raise ValueError
    subejct_path = f"CGMacros-{subject_id:03d}/CGMacros-{subject_id:03d}.csv"
    subject_file = os.path.join(csv_dir, subejct_path)
    if not os.path.exists(subject_file):
        tqdm.write(f"File {subject_file} not found")
        raise FileNotFoundError
    dataset_df = pd.read_csv(subject_file, index_col=None)
    dataset_df["Timestamp"] = pd.to_datetime(dataset_df["Timestamp"])
    return dataset_df.set_index("Timestamp")


def get_valid_meal_idx(
    dataset_df: pd.DataFrame,
    meal_types: list = ["breakfast", "lunch"],
    ignore_x_hours: int = 0,  # could be configured to skip first meal
    col_names=[
        "Meal Type",
        "Image path",
        "Calories",
        "Carbs",
        "Protein",
        "Fat",
        "Fiber",
    ],  # columns to check for NaN values
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


def load_cgm_trace(
    dataset_df: pd.DataFrame,
    valid_meal_idx: pd.DatetimeIndex,
    prev_hours: int = 0,
    post_hours: int = 3,
    cgm_cols=["Dexcom GL", "Libre GL"],  # Whether to select both of either one
):
    cgm_len = 60 * (prev_hours + post_hours)
    cgm_trace = np.zeros((len(valid_meal_idx), len(cgm_cols), cgm_len))
    for i, ts_idx in enumerate(valid_meal_idx):
        # NOTE: Extracting the time-series trace for meals
        trace_df = extract_meal_trace(dataset_df, ts_idx, prev_hours, post_hours).copy()
        cgm_arr = trace_df[cgm_cols].T.to_numpy()
        # NOTE: Padding the CGM features with NaNs
        if cgm_arr.shape[1] < cgm_len:
            padding = np.full((len(cgm_cols), cgm_len - cgm_arr.shape[1]), np.nan)
            cgm_arr = np.concatenate((cgm_arr, padding), axis=1)
        # NOTE: Adding features to samples
        cgm_trace[i] = cgm_arr
    return cgm_trace


def get_image(
    img_filename: str,
    subject_id: int,
    target_size: tuple,
    cgmacros_path: str = "/scratch/CGMacros/cgmacros/1.0.0/CGMacros/",
) -> np.ndarray:
    subject_path = f"CGMacros-{subject_id:03d}/"
    img_path = f"{cgmacros_path}{subject_path}{img_filename}"
    if not os.path.exists(img_path):
        print(f"File {img_path} not found")
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


def split_dataset(
    cgm_trace: np.ndarray,
    znorm_cgm: np.ndarray,
    img_arr: np.ndarray,
    a1c: np.ndarray,
    fasting_glucose: np.ndarray,
    label_arr: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 2025,
):
    # NOTE: Drop rows with NaN values in cgm_trace, img_arr, or label_arr
    valid_idx = ~(
        np.isnan(cgm_trace).any(axis=(1, 2))
        | np.isnan(znorm_cgm).any(axis=(1, 2))
        | np.isnan(img_arr).any(axis=(1, 2, 3))
        | np.isnan(a1c).any(axis=1)
        | np.isnan(fasting_glucose).any(axis=1)
        | np.isnan(label_arr).any(axis=1)
    )
    cgm_trace = cgm_trace[valid_idx]
    znorm_cgm = znorm_cgm[valid_idx]
    img_arr = img_arr[valid_idx]
    a1c = a1c[valid_idx]
    fasting_glucose = fasting_glucose[valid_idx]
    label_arr = label_arr[valid_idx]

    # First split into train+val and test
    train_val_idx, test_idx = train_test_split(
        np.arange(len(cgm_trace)),
        test_size=test_size,
        random_state=random_state,
    )

    # Further split train+val into train and validation
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size
        / (1 - test_size),  # Adjust validation size relative to train+val
        random_state=random_state,
    )

    # Split the data into train, validation, and test sets
    train_cgm_trace = cgm_trace[train_idx]
    val_cgm_trace = cgm_trace[val_idx]
    test_cgm_trace = cgm_trace[test_idx]

    train_znorm_cgm = znorm_cgm[train_idx]
    val_znorm_cgm = znorm_cgm[val_idx]
    test_znorm_cgm = znorm_cgm[test_idx]

    train_img_arr = img_arr[train_idx]
    val_img_arr = img_arr[val_idx]
    test_img_arr = img_arr[test_idx]

    train_a1c = a1c[train_idx]
    val_a1c = a1c[val_idx]
    test_a1c = a1c[test_idx]

    train_fasting_glucose = fasting_glucose[train_idx]
    val_fasting_glucose = fasting_glucose[val_idx]
    test_fasting_glucose = fasting_glucose[test_idx]

    train_label_arr = label_arr[train_idx]
    val_label_arr = label_arr[val_idx]
    test_label_arr = label_arr[test_idx]

    # Create datasets
    train_dataset = CGMacrosDataset(
        cgm_trace=train_cgm_trace,
        znorm_cgm=train_znorm_cgm,
        img_arr=train_img_arr,
        a1c=train_a1c,
        fasting_glucose=train_fasting_glucose,
        label_arr=train_label_arr,
    )
    val_dataset = CGMacrosDataset(
        cgm_trace=val_cgm_trace,
        znorm_cgm=val_znorm_cgm,
        img_arr=val_img_arr,
        a1c=val_a1c,
        fasting_glucose=val_fasting_glucose,
        label_arr=val_label_arr,
    )
    test_dataset = CGMacrosDataset(
        cgm_trace=test_cgm_trace,
        znorm_cgm=test_znorm_cgm,
        img_arr=test_img_arr,
        a1c=test_a1c,
        fasting_glucose=test_fasting_glucose,
        label_arr=test_label_arr,
    )
    return train_dataset, val_dataset, test_dataset


def normalize_each_cgm_point(subject_cgm_arr: np.ndarray):
    """
    normalize_each_cgm_point normalizes the CGM data point-wise

    That is, at each time point, the CGM values are normalized within a single subject across meals

    Args:
        subject_cgm_arr (np.ndarray): CGM data for a single subject
    """
    normalized_arr = np.zeros_like(subject_cgm_arr)
    for i in range(subject_cgm_arr.shape[1]):  # each CGM sensor (2 total)
        for j in range(subject_cgm_arr.shape[2]):  # each time point (180 total)
            # NOTE: Normalizing each CGM point
            mean = np.nanmean(subject_cgm_arr[:, i, j])
            std = np.nanstd(subject_cgm_arr[:, i, j]) + 1e-8
            normalized_arr[:, i, j] = (subject_cgm_arr[:, i, j] - mean) / std
    return normalized_arr


def generate_CGMacrosDataset(
    img_size: tuple = (112, 112),
    save_dir: str = "dataset/",
):
    all_cgm_trace = []
    all_img_data = []
    all_label_arr = []
    all_znorm_cgm = []
    for i in tqdm(
        range(1, 50, 1),
        desc="Processing Subjects",
        ascii=True,
    ):  # IDs are from 1 to 49
        try:
            dataset_df = load_CGMacros(subject_id=i)  # loading which subject
            # NOTE: not used yet, as this is 24H experiment
            # a1c, fg = get_subject_a1c_and_fg(subject_id=i)
        except FileNotFoundError:
            tqdm.write(f"Skipping Subject {i:03d}")
            continue
        # NOTE: getting the list of timestamps of valid meals
        valid_meal_idx = get_valid_meal_idx(
            dataset_df, meal_types=["breakfast"], ignore_x_hours=0
        )

        # NOTE: Extracting macronutritient labels
        label_arr = dataset_df.loc[valid_meal_idx][label_cols].to_numpy()

        # NOTE: Extracting associated images
        img_df = dataset_df.loc[valid_meal_idx][img_cols]
        img_data = load_img_data(img_df, i, target_size=img_size)

        # NOTE: Extracting the time-series trace for each valid meal
        cgm_trace = load_cgm_trace(dataset_df, valid_meal_idx)
        znorm_cgm = normalize_each_cgm_point(cgm_trace)
        # NOTE: Extracting the a1c and fasting glucose values as array
        a1c_arr = np.full((len(valid_meal_idx), 1), a1c)
        fg_arr = np.full((len(valid_meal_idx), 1), fg)
        tqdm.write(
            f"Subject ID: {i}, CGM Features: {cgm_trace.shape}, "
            f"Image Data: {img_data.shape}, Label Data: {label_arr.shape}"
        )
        all_cgm_trace.append(cgm_trace)
        all_znorm_cgm.append(znorm_cgm)
        all_img_data.append(img_data)
        all_a1c.append(a1c_arr)
        all_fasting_glucose.append(fg_arr)
        all_label_arr.append(label_arr)

    all_cgm_trace = np.concatenate(all_cgm_trace, axis=0)
    all_znorm_cgm = np.concatenate(all_znorm_cgm, axis=0)
    all_img_data = np.concatenate(all_img_data, axis=0)
    all_a1c = np.concatenate(all_a1c, axis=0)
    all_fasting_glucose = np.concatenate(all_fasting_glucose, axis=0)
    all_label_arr = np.concatenate(all_label_arr, axis=0)

    print(f"All CGM Features: {all_cgm_trace.shape}")
    print(f"All Z-Norm CGM Features: {all_znorm_cgm.shape}")
    print(f"All Image Data: {all_img_data.shape}")
    print(f"All Label Data: {all_label_arr.shape}")
    train_dataset, val_dataset, test_dataset = split_dataset(
        all_cgm_trace,
        all_znorm_cgm,
        all_img_data,
        all_a1c,
        all_fasting_glucose,
        all_label_arr,
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(train_dataset, f"{save_dir}cgmacros_train_dataset.pt")
    torch.save(val_dataset, f"{save_dir}cgmacros_val_dataset.pt")
    torch.save(test_dataset, f"{save_dir}cgmacros_test_dataset.pt")
    print(f"Saved to {save_dir}")
    return train_dataset, val_dataset, test_dataset


def main():
    # load_CGMacros()
    generate_CGMacrosDataset()


if __name__ == "__main__":
    main()
