import pandas as pd
import os, pdb, copy
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm


# class DailyTracesDataset(Dataset):
#     def __init__(self, data_dir, subject_ids=None, transform=None, skip_days=[1]):
#         # Initialization remains the same
#         self.data_dir = data_dir
#         self.transform = transform
#         self.skip_days = skip_days or [1]
#         transform = None
        
#         # Find relevant subject files
#         if subject_ids is None:
#             self.data_files = [
#                 f for f in os.listdir(data_dir) 
#                 if f.startswith("subject_") and f.endswith("_daily_data.pt")
#             ]
#         else:
#             self.data_files = [
#                 f"subject_{sid:03d}_daily_data.pt" 
#                 for sid in subject_ids 
#                 if os.path.exists(os.path.join(data_dir, f"subject_{sid:03d}_daily_data.pt"))
#             ]
        
#         # Build indices accounting for skip_days
#         self.indices = []
#         self.subject_day_pairs = []
        
#         for file_idx, fname in enumerate(self.data_files):
#             data = torch.load(os.path.join(data_dir, fname), weights_only=False)
#             subject_id = data['subject_id']
            
#             for day_idx, day_str in enumerate(data['days']):
#                 day_num = int(day_str.split('-')[2])
#                 if day_num not in self.skip_days:
#                     self.indices.append((file_idx, day_idx))
#                     self.subject_day_pairs.append((subject_id, day_num))

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         file_idx, day_idx = self.indices[idx]
#         data = torch.load(os.path.join(self.data_dir, self.data_files[file_idx]), weights_only=False)
#         day = data['days'][day_idx]
        
#         # Apply transforms if specified
#         def _apply_transform(x):
#             return self.transform(x) if self.transform else x
        
#         demographics_path = 'demographicPCA.csv'
#         inetensityMinute = 'intensityDataMinutes.csv'
#         pca_df = pd.read_csv(demographics_path)
#         subject_data = pca_df[pca_df['SubjectID'] == data['subject_id']]
#         pca_values = subject_data[['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']].values

#         # Load minute-level intensity data
#         intensity_minute_df = pd.read_csv(inetensityMinute)

#         intensity_row = intensity_minute_df[
#             (intensity_minute_df['subject_id'] == data['subject_id']) &
#             (intensity_minute_df['day'] == day_idx)
#         ]

#         if not intensity_row.empty:
#             intensity_array_str = intensity_row.iloc[0]['intensity_array']
#             minute_intensity = torch.tensor(eval(intensity_array_str), dtype=torch.float32)
#         else:
#             print(f"Could not find intensity row for {data['subject_id']} on day {day_idx}")
#             minute_intensity = torch.zeros(1440, dtype=torch.float32)

#         # Include the new features in the returned item
#         return {
#             'subject_id': data['subject_id'],
#             'day': day,
#             'cgm_data': _apply_transform(data['cgm_data'][day_idx]),
#            'demographics':  torch.tensor(pca_values.flatten(), dtype=torch.float32),
#             'activity_data': _apply_transform(data['activity_data'][day_idx]),
#             'images': [_apply_transform(img['image']) for img in data['image_data'].get(day, [])],
#             'nutrition': data['nutrition_data'].get(day, []),
#             'subject_day_pair': self.subject_day_pairs[idx],
#             'timestamps': data.get('timestamp_vectors', {}).get(day, []),  # New: timestamps
#             'meal_timing_features': data.get('meal_timing_features', {}).get(day, []),  # New: meal timing features
#             'minute_intensity': minute_intensity
#         }


import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class DailyTracesDataset(Dataset):
    def __init__(self, data_dir, subject_ids=None, transform=None, skip_days=[0]):
        self.data_dir = data_dir
        self.transform = transform
        self.skip_days = skip_days or [0]
        self.indices = []
        self.subject_day_pairs = []

        self.day_counter_lookup = {}    # (subject_id, date) -> day_counter
        self.reverse_day_lookup = {}    # (subject_id, day_counter) -> date

        # Load demographic PCA features
        self.pca_df = pd.read_csv('demographicPCA.csv')

        # Load both minute-wise and hourly intensity data
        self.intensity_minute_df = pd.read_csv('intensityDataMinutes.csv')
        self.intensity_hourly_df = pd.read_csv('intensityDataHourly.csv')

        # Load subject data files
        if subject_ids is None:
            self.data_files = [
                f for f in os.listdir(data_dir)
                if f.startswith("subject_") and f.endswith("_daily_data.pt")
            ]
        else:
            self.data_files = [
                f"subject_{sid:03d}_daily_data.pt"
                for sid in subject_ids
                if os.path.exists(os.path.join(data_dir, f"subject_{sid:03d}_daily_data.pt"))
            ]

        # Build indices and mappings
        for file_idx, fname in enumerate(self.data_files):
            data_path = os.path.join(data_dir, fname)
            data = torch.load(data_path, weights_only=False)
            subject_id = data['subject_id']

            sorted_days = sorted(data['days'])
            for day_counter, day_str in enumerate(sorted_days):
                if day_counter in self.skip_days:
                    continue

                self.day_counter_lookup[(subject_id, day_str)] = day_counter
                self.reverse_day_lookup[(subject_id, day_counter)] = day_str

                day_idx = data['days'].index(day_str)
                self.indices.append((file_idx, day_idx))
                self.subject_day_pairs.append((subject_id, day_counter))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, day_idx = self.indices[idx]
        data_path = os.path.join(self.data_dir, self.data_files[file_idx])
        data = torch.load(data_path, weights_only=False)

        subject_id = data['subject_id']
        day = data['days'][day_idx]

        def _apply_transform(x):
            return self.transform(x) if self.transform else x

        # PCA features
        subject_data = self.pca_df[self.pca_df['SubjectID'] == subject_id]
        pca_values = subject_data[['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']].values

        # Day counter lookup
        day_counter = self.day_counter_lookup.get((subject_id, day), None)

        # Load minute-wise intensity
        if day_counter is not None:
            minute_row = self.intensity_minute_df[
                (self.intensity_minute_df['subject_id'] == subject_id) &
                (self.intensity_minute_df['day'] == day_counter)
            ]
            minute_intensity = torch.tensor(eval(minute_row.iloc[0]['intensity_array']), dtype=torch.float32) \
                if not minute_row.empty else torch.zeros(1440, dtype=torch.float32)
        else:
            minute_intensity = torch.zeros(1440, dtype=torch.float32)

        # Load hourly intensity
        if day_counter is not None:
            hourly_row = self.intensity_hourly_df[
                (self.intensity_hourly_df['subject_id'] == subject_id) &
                (self.intensity_hourly_df['day'] == day_counter)
            ]
            hourly_intensity = torch.tensor(eval(hourly_row.iloc[0]['intensity_array']), dtype=torch.float32) \
                if not hourly_row.empty else torch.zeros(24, dtype=torch.float32)
        else:
            hourly_intensity = torch.zeros(24, dtype=torch.float32)

        return {
            'subject_id': subject_id,
            'day': day,
            'cgm_data': _apply_transform(data['cgm_data'][day_idx]),
            'demographics': torch.tensor(pca_values.flatten(), dtype=torch.float32),
            'activity_data': _apply_transform(data['activity_data'][day_idx]),
            'images': [_apply_transform(img['image']) for img in data['image_data'].get(day, [])],
            'nutrition': data['nutrition_data'].get(day, []),
            'subject_day_pair': self.subject_day_pairs[idx],
            'timestamps': data.get('timestamp_vectors', {}).get(day, []),
            'meal_timing_features': data.get('meal_timing_features', {}).get(day, []),
            'minute_intensity': minute_intensity,
            'hourly_intensity': hourly_intensity
        }


    def get_day_counter(self, subject_id, day_str):
        """
        Helper function to get the day counter for a given subject and day string.
        """
        return self.day_counter_lookup.get((subject_id, day_str), None)

    def get_day_str_from_counter(self, subject_id, day_counter):
        """
        Helper function to get the day string from the day counter for a given subject.
        """
        return self.reverse_day_lookup.get((subject_id, day_counter), None)

    

class SubjectDaySubset(Dataset):
    """
    Subset of DailyTracesDataset based on indices.
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]