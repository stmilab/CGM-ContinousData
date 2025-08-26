import pandas as pd
import os, pdb, copy
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

import numpy as np

from data_types import DailyTracesDataset , SubjectDaySubset





def clean_time_series_data(df, cgm_cols=["Dexcom GL", "Libre GL"], activity_cols=["HR", "METs"]):
    """Clean and validate time-series data"""
    # Copy to avoid modifying original
    df = df.copy()
    
    # 1. Handle missing values
    for col in cgm_cols + activity_cols:
        if col in df.columns:
            # Linear interpolation with 5-minute window
            df[col] = df[col].interpolate(method='linear', limit=5)
            # Forward/backward fill remaining
            df[col] = df[col].ffill().bfill()
    
    return df

def load_daily_traces(
    dataset_df: pd.DataFrame, 
    subject_id: int,
    cgm_cols=["Dexcom GL", "Libre GL"],
    activity_cols=["HR", "METs"],
    img_size=(112, 112),
    start_hour=4  # Start at 6 AM
):
    """
    Enhanced version with full-day (1440 minute) time features.
    """
    # Cleaning functions remain the same
    def clean_series(series):
        series = series.interpolate(method='linear', limit=5).ffill().bfill()
        try:
            from scipy.signal import savgol_filter
            series = savgol_filter(series, window_length=15, polyorder=2)
        except ImportError:
            series = series.rolling(window=15, min_periods=1, center=True).mean()
        return series

    # Apply cleaning
    cleaned_df = dataset_df.copy()
    for col in cgm_cols + activity_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = clean_series(cleaned_df[col])
    
    # Resample to 1-minute frequency
    resampled_df = cleaned_df.resample('1min').ffill(limit=5)
    
    # Fill NaNs with median
    for col in cgm_cols + activity_cols:
        if col in resampled_df.columns:
            resampled_df[col] = resampled_df[col].fillna(resampled_df[col].median())
    
    # Statistics for normalization
    cgm_stats = {
        'mean': resampled_df[cgm_cols].mean().values if all(col in resampled_df.columns for col in cgm_cols) else np.zeros(len(cgm_cols)),
        'std': resampled_df[cgm_cols].std().values if all(col in resampled_df.columns for col in cgm_cols) else np.ones(len(cgm_cols))
    }
    activity_stats = {
        'mean': resampled_df[activity_cols].mean().values if all(col in resampled_df.columns for col in activity_cols) else np.zeros(len(activity_cols)),
        'std': resampled_df[activity_cols].std().values if all(col in resampled_df.columns for col in activity_cols) else np.ones(len(activity_cols))
    }

    # Initialize arrays and dictionaries
    days = pd.Series(resampled_df.index.date).unique()
    days_list = [str(day) for day in days]
    cgm_daily_data = np.full((len(days), len(cgm_cols), 1440), np.nan)
    activity_daily_data = np.full((len(days), len(activity_cols), 1440), np.nan)
    image_data_by_day = {}
    nutrition_data_by_day = {}
    timestamp_vectors = {}  # Store full day timestamps
    meal_timing_features = {}  # Store meal timing features for full day
    
    # Time window parameters
    minutes_after_last_meal = 6 * 60  # 6 hours after last meal

    # Process each day
    for i, day in enumerate(days):
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        day_data = resampled_df.loc[day_start:day_end]
        
        # Extract meal information for this day
        meal_rows = dataset_df.loc[day_start:day_end].dropna(subset=['Meal Type'])
        meal_times = meal_rows.index.sort_values() if not meal_rows.empty else []
        
        # Create full day timestamp vector (1440 minutes)
        full_day_timestamps = [day_start + pd.Timedelta(minutes=m) for m in range(1440)]
        timestamp_vectors[str(day)] = full_day_timestamps
        
        # Initialize full day meal timing features
        meal_timing = np.zeros((5, 1440))  # 5 features for 1440 minutes
        
        # Set default values for all minutes
        meal_timing[0, :] = -1  # Minutes since most recent meal
        meal_timing[1, :] = -1  # Minutes until next meal
        meal_timing[2, :] = 0   # Is within 2 hours after meal
        meal_timing[3, :] = 0   # Count of previous meals
        meal_timing[4, :] = np.arange(1440)  # Minutes since start of day
        
        # Determine time window for processing (we'll still store full day)
        if len(meal_times) == 0:
            # If no meals, use default window (6 AM to midnight)
            window_start = day_start + pd.Timedelta(hours=start_hour)
            window_end = day_end
        else:
            # Start at 6 AM
            window_start = day_start + pd.Timedelta(hours=start_hour)
            
            # End 6 hours after the last meal or at day end, whichever is earlier
            last_meal_time = meal_times[-1]
            last_meal_plus_6h = last_meal_time + pd.Timedelta(minutes=minutes_after_last_meal)
            window_end = min(last_meal_plus_6h, day_end)
        
        # Filter data to our window
        window_data = day_data.loc[window_start:window_end]
        
        # Calculate window minutes for validation/debugging
        window_minutes = len(window_data)
        
        if not window_data.empty:
            # Store CGM and activity data for the window
            for j, col in enumerate(cgm_cols):
                if col in window_data.columns:
                    # Get minutes of day for each window timestamp
                    minutes_of_day = (window_data.index.hour * 60 + window_data.index.minute).values
                    
                    vals = window_data[col].values
                    vals = np.nan_to_num(vals, nan=window_data[col].median())
                    cgm_daily_data[i, j, minutes_of_day] = vals
            
            for j, col in enumerate(activity_cols):
                if col in window_data.columns:
                    # Get minutes of day for each window timestamp
                    minutes_of_day = (window_data.index.hour * 60 + window_data.index.minute).values
                    
                    vals = window_data[col].values
                    vals = np.nan_to_num(vals, nan=window_data[col].median())
                    activity_daily_data[i, j, minutes_of_day] = vals
            
            # Calculate meal timing features for each timestamp in the full day
            for minute in range(1440):
                timestamp = day_start + pd.Timedelta(minutes=minute)
                
                # 1. Minutes since most recent meal
                prev_meals = [m for m in meal_times if m <= timestamp]
                if prev_meals:
                    meal_timing[0, minute] = (timestamp - prev_meals[-1]).total_seconds()/60
                # else keep default -1
                
                # 2. Minutes until next meal
                next_meals = [m for m in meal_times if m > timestamp]
                if next_meals:
                    meal_timing[1, minute] = (next_meals[0] - timestamp).total_seconds()/60
                # else keep default -1
                
                # 3. Boolean: Is this within 2 hours after a meal?
                meal_timing[2, minute] = 1 if (meal_timing[0, minute] >= 0 and meal_timing[0, minute] <= 120) else 0
                
                # 4. Count of previous meals for the day
                meal_timing[3, minute] = len(prev_meals)
                
                # 5. Minutes since start of day is already set to minute number
        
        # Store meal timing features (full day)
        meal_timing_features[str(day)] = meal_timing
        
        # Process nutrition data with more detailed information
        day_str = str(day)
        original_day_data = dataset_df.loc[day_start:day_end]
        
        nutrition_rows = original_day_data.dropna(subset=['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber'], how='all')
        day_nutrition = []
        
        # Enhanced meal information
        meal_counter = {}  # Track meal numbers by type
        
        # Process meals in chronological order
        for ts, row in nutrition_rows.iterrows():
            meal_type = row['Meal Type'] if pd.notna(row['Meal Type']) else 'Unknown'
            
            # Increment meal counter for this type
            if meal_type not in meal_counter:
                meal_counter[meal_type] = 1
            else:
                meal_counter[meal_type] += 1
                
            # Calculate meal timing within day
            minutes_since_day_start = (ts - day_start).total_seconds() / 60
            hour_of_day = ts.hour + ts.minute/60
            
            nutrition = {
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'MealType': meal_type,
                'MealNumber': meal_counter[meal_type],  # Which breakfast/lunch/dinner/snack is this?
                'MinuteOfDay': int(minutes_since_day_start),
                'HourOfDay': hour_of_day,
                'calories': row['Calories'] if pd.notna(row['Calories']) else 0,
                'carbs': row['Carbs'] if pd.notna(row['Carbs']) else 0,
                'protein': row['Protein'] if pd.notna(row['Protein']) else 0,
                'fat': row['Fat'] if pd.notna(row['Fat']) else 0,
                'fiber': row['Fiber'] if pd.notna(row['Fiber']) else 0,
                'has_image': pd.notna(row['Image path'])
            }
            day_nutrition.append(nutrition)
        
        # Add meal sequence information
        if day_nutrition:
            # Sort by timestamp
            day_nutrition = sorted(day_nutrition, key=lambda x: x['MinuteOfDay'])
            
            # Add meal sequence number and intervals
            for k in range(len(day_nutrition)):
                day_nutrition[k]['MealSequence'] = k + 1  # 1-based meal sequence for the day
                
                # Time to next meal
                if k < len(day_nutrition) - 1:
                    day_nutrition[k]['MinutesToNextMeal'] = day_nutrition[k+1]['MinuteOfDay'] - day_nutrition[k]['MinuteOfDay']
                else:
                    day_nutrition[k]['MinutesToNextMeal'] = -1  # No next meal
                
                # Time since previous meal
                if k > 0:
                    day_nutrition[k]['MinutesSincePrevMeal'] = day_nutrition[k]['MinuteOfDay'] - day_nutrition[k-1]['MinuteOfDay']
                else:
                    day_nutrition[k]['MinutesSincePrevMeal'] = -1  # No previous meal
        
        nutrition_data_by_day[day_str] = day_nutrition
        
        # Image data processing remains largely the same
        image_rows = original_day_data.dropna(subset=['Image path'])
        day_images = []
        for ts, row in image_rows.iterrows():
            try:
                img_data = get_image(row['Image path'], subject_id, img_size)
                # Calculate timing features for this image/meal
                minutes_since_day_start = (ts - day_start).total_seconds() / 60
                
                metadata = {
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'minute_of_day': int(minutes_since_day_start),
                    'meal_type': row['Meal Type'] if 'Meal Type' in row else None,
                    'calories': row['Calories'] if 'Calories' in row else None,
                    'carbs': row['Carbs'] if 'Carbs' in row else None,
                    'protein': row['Protein'] if 'Protein' in row else None,
                    'fat': row['Fat'] if 'Fat' in row else None,
                    'fiber': row['Fiber'] if 'Fiber' in row else None
                }
                day_images.append({'image': img_data, 'metadata': metadata})
            except FileNotFoundError:
                continue
        image_data_by_day[day_str] = day_images if day_images else []

    # Window metadata
    window_metadata = {
        'start_hour': start_hour,
        'hours_after_last_meal': minutes_after_last_meal / 60,
        'full_day_length': 1440
    }

    return (
        days_list,
        cgm_daily_data,
        activity_daily_data,
        image_data_by_day,
        nutrition_data_by_day,
        cgm_stats,
        activity_stats,
        window_metadata,
        timestamp_vectors,        # Now: full day timestamps
        meal_timing_features      # Now: full day meal timing features
    )

def load_CGMacros(
    subject_id: int,
    csv_dir: str = "CGMacros-2",
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
    dataset_df = clean_time_series_data(dataset_df)  # Add cleaning step
    return dataset_df.set_index("Timestamp")

def get_image(
    img_filename: str,
    subject_id: int,
    target_size: tuple,
    cgmacros_path: str = "CGMacros-2/",
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
def create_daily_dataset(
    subject_id: int,
    csv_dir: str = "CGMacros 2",
    cgm_cols=["Dexcom GL", "Libre GL"],
    activity_cols=["HR","METs"],
    img_size=(112, 112),
    start_hour=6,
    verbose=False
):
    try:
        # Load data with column validation
        dataset_df = load_CGMacros(subject_id, csv_dir)
        
        
        if verbose:
            print("Available columns:", dataset_df.columns.tolist())
        
        # Handle missing columns gracefully
        available_activity_cols = [col for col in activity_cols 
                                 if col in dataset_df.columns]
        if len(available_activity_cols) < len(activity_cols):
            print(f"Warning: Missing activity columns. Using {available_activity_cols} for subject {subject_id}")
        
        # Process data with validated columns and custom time window
        result = load_daily_traces(
            dataset_df, subject_id, 
            cgm_cols=cgm_cols,
            activity_cols=available_activity_cols,
            img_size=img_size,
            start_hour=start_hour
        )
        
        return (subject_id,) + result  # Return all elements with subject_id
    
    except FileNotFoundError:
        print(f"Data for subject {subject_id} not found.")
        return None
    except Exception as e:
        print(f"Error processing subject {subject_id}: {str(e)}")
        return None
    
def process_multiple_subjects(
    subject_ids=None,
    csv_dir="CGMacros-2",
    demographics_path = "demographicPCA.csv",
    save_dir="processed_data/",
    cgm_cols=["Dexcom GL","Libre GL"],
    activity_cols=["HR","METs"],
    img_size=(112, 112),
    start_hour=6
):
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if subject_ids is None:
        subject_ids = range(1, 51)  # Try subjects 1-50
    
    summary = {
        'processed_subjects': [],
        'total_days': 0,
        'total_images': 0,
        'total_meals': 0
    }
    
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        result = create_daily_dataset(subject_id, csv_dir, start_hour=start_hour)
        if result is None:
            continue
            
        # Unpack all return values
        (subject_id, days, cgm, activity, images, nutrition, 
         cgm_stats, activity_stats, window_metadata, 
         timestamp_vectors, meal_timing_features) = result
        
        pca_df = pd.read_csv(demographics_path)
        pca_vector = pca_df[pca_df['SubjectID'] == subject_id].iloc[0, 1:].values.astype("float32")

        
        # Save data with all new features
        subject_data = {
            'subject_id': subject_id,
            'days': days,
            'cgm_data': cgm,
            'activity_data': activity,
            'image_data': images,
            'nutrition_data': nutrition,
            'cgm_stats': cgm_stats,
            'activity_stats': activity_stats,
            'demographics':pca_vector,
            'window_metadata': window_metadata,
            'timestamp_vectors': timestamp_vectors,  # New: store timestamps
            'meal_timing_features': meal_timing_features  # New: meal timing features
        }
        torch.save(subject_data, os.path.join(save_dir, f"subject_{subject_id:03d}_daily_data.pt"))
        
        # Update summary counts
        summary['processed_subjects'].append(subject_id)
        summary['total_days'] += len(days)
        summary['total_images'] += sum(len(imgs) for imgs in images.values())
        summary['total_meals'] += sum(len(meals) for meals in nutrition.values())
    return summary


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

