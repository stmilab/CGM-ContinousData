import pandas as pd
import os, pdb, copy
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from data_types import SubjectDaySubset,DailyTracesDataset



def custom_collate(batch):
    """Handles variable-length nutrition data, images and timestamp vectors"""


    def fix_nans(array):
        """Replace NaNs with median (per channel)"""
        median_vals = np.nanmedian(array, axis=1, keepdims=True)
        return np.where(np.isnan(array), median_vals, array)

    # Fix NaNs before converting to tensors
    for i, x in enumerate(batch):
        x['cgm_data'] = fix_nans(x['cgm_data'])
        x['activity_data'] = fix_nans(x['activity_data'])
        if 'meal_timing_features' in x and len(x['meal_timing_features']) > 0:
            x['meal_timing_features'] = fix_nans(x['meal_timing_features'])

    return {
        'subject_ids': torch.tensor([x['subject_id'] for x in batch]),
        'days': [x['day'] for x in batch],
        
        'cgm_data': torch.stack([torch.tensor(x['cgm_data'], dtype=torch.float32) for x in batch]),
        'activity_data': torch.stack([torch.tensor(x['activity_data'], dtype=torch.float32) for x in batch]),
        'images': [x['images'] for x in batch],  # List of lists
        'nutrition': [x['nutrition'] for x in batch],  # List of lists
        'subject_day_pairs': [x['subject_day_pair'] for x in batch],
        'timestamps': [x.get('timestamps', []) for x in batch],  # New: timestamps for each data point
        'meal_timing_features': [torch.tensor(x.get('meal_timing_features', np.zeros((5, 1))), 
                                              dtype=torch.float32) if len(x.get('meal_timing_features', [])) > 0 
                                 else torch.zeros((5, 1)) for x in batch],  # New: meal timing features,
        'demographics': torch.stack([x['demographics'] for x in batch]),
       'intensity_minute': torch.stack([x['minute_intensity'] for x in batch]),
       'intensity_hour': torch.stack([x['hourly_intensity'] for x in batch]),
    }


def split_dataset_by_subject_day(dataset, test_size=0.2, random_state=2025):
    """
    Split the dataset based on subject-day pairs to ensure all data from
    the same subject and day stays together in either training or testing set.
    
    Args:
        dataset (DailyTracesDataset): The dataset to split
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_indices, test_indices)
    """
    # Get unique subject-day pairs
    subject_day_df = pd.DataFrame(dataset.subject_day_pairs, columns=['subject_id', 'day_id'])
    unique_pairs = subject_day_df.drop_duplicates()
    
    # Split the unique subject-day pairs
    train_pairs, test_pairs = train_test_split(
        unique_pairs, 
        test_size=test_size,
        random_state=random_state
    )
    
    # Convert to sets for faster lookup
    train_pairs_set = set(zip(train_pairs['subject_id'], train_pairs['day_id']))
    test_pairs_set = set(zip(test_pairs['subject_id'], test_pairs['day_id']))
    
    # Create masks for train and test indices
    train_indices = []
    test_indices = []
    
    for i, (subject_id, day_id) in enumerate(dataset.subject_day_pairs):
        if (subject_id, day_id) in train_pairs_set:
            train_indices.append(i)
        elif (subject_id, day_id) in test_pairs_set:
            test_indices.append(i)
    
    return train_indices, test_indices



def get_train_test_datasets(data_dir, subject_ids=None, test_size=0.2, random_state=2025, transform=None):
    """
    Get train and test datasets split by subject-day pairs.
    
    Args:
        data_dir (str): Directory containing processed data
        subject_ids (list): List of subject IDs to include. If None, include all available.
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        transform (callable): Optional transform to apply to the data
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Create the full dataset
    full_dataset = DailyTracesDataset(data_dir, subject_ids, transform)
    
    # Split by subject-day pairs
    train_indices, test_indices = split_dataset_by_subject_day(full_dataset, test_size, random_state)
    
    # Create train and test subsets
    train_dataset = SubjectDaySubset(full_dataset, train_indices)
    test_dataset = SubjectDaySubset(full_dataset, test_indices)
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def filter_samples_by_nutrition(dataset):
    """
    Filters samples that don't meet nutrition quality requirements:
    - At least 2 distinct meals out of breakfast, lunch, dinner
    - Non-zero total calories
    - Total calories >= 300
    """
    valid_meals = {"breakfast", "lunch", "dinner","snack","snacks"}
    filtered = []
    
    for sample in dataset:
        nutrition = sample.get("nutrition", [])
        if not nutrition or not isinstance(nutrition, list):
            continue

        meal_types_present = {meal.get("MealType", "").lower() for meal in nutrition}
        relevant_meals = meal_types_present & valid_meals

        total_calories = sum(meal.get("calories", 0) for meal in nutrition)
        calories_are_valid = total_calories > 0 and total_calories >= 300

        if len(relevant_meals) >= 2 and calories_are_valid:
            filtered.append(sample)

    return filtered
