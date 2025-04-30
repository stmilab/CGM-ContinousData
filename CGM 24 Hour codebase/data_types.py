import pandas as pd
import os, pdb, copy
import numpy as np
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm


class DailyTracesDataset(Dataset):
    def __init__(self, data_dir, subject_ids=None, transform=None, skip_days=[1]):
        # Initialization remains the same
        self.data_dir = data_dir
        self.transform = transform
        self.skip_days = skip_days or [1]
        transform = None
        
        # Find relevant subject files
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
        
        # Build indices accounting for skip_days
        self.indices = []
        self.subject_day_pairs = []
        
        for file_idx, fname in enumerate(self.data_files):
            data = torch.load(os.path.join(data_dir, fname), weights_only=False)
            subject_id = data['subject_id']
            
            for day_idx, day_str in enumerate(data['days']):
                day_num = int(day_str.split('-')[2])
                if day_num not in self.skip_days:
                    self.indices.append((file_idx, day_idx))
                    self.subject_day_pairs.append((subject_id, day_num))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, day_idx = self.indices[idx]
        data = torch.load(os.path.join(self.data_dir, self.data_files[file_idx]), weights_only=False)
        day = data['days'][day_idx]
        
        # Apply transforms if specified
        def _apply_transform(x):
            return self.transform(x) if self.transform else x
        
        demographics_path = 'demographicPCA.csv'
        pca_df = pd.read_csv(demographics_path)
        subject_data = pca_df[pca_df['SubjectID'] == data['subject_id']]
        pca_values = subject_data[['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5']].values

        # Include the new features in the returned item
        return {
            'subject_id': data['subject_id'],
            'day': day,
            'cgm_data': _apply_transform(data['cgm_data'][day_idx]),
           'demographics':  torch.tensor(pca_values.flatten(), dtype=torch.float32),
            'activity_data': _apply_transform(data['activity_data'][day_idx]),
            'images': [_apply_transform(img['image']) for img in data['image_data'].get(day, [])],
            'nutrition': data['nutrition_data'].get(day, []),
            'subject_day_pair': self.subject_day_pairs[idx],
            'timestamps': data.get('timestamp_vectors', {}).get(day, []),  # New: timestamps
            'meal_timing_features': data.get('meal_timing_features', {}).get(day, [])  # New: meal timing features
        }
    

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