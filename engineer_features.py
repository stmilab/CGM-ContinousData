import numpy as np
import pandas as pd
import pdb
from tqdm import tqdm
from load_24h_cgmacros import generate_24H_CGMacros_dataset
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)


def get_wakeup_idx(intensity_series):
    # Filter intensity index to between 6AM and 10AM
    mask = (intensity_series.index.time >= pd.to_datetime("06:00").time()) & (
        intensity_series.index.time <= pd.to_datetime("10:00").time()
    )
    morning_intensity = intensity_series[mask]
    candidate_idx = morning_intensity[morning_intensity != morning_intensity.min()]
    wakeup_idx = (
        candidate_idx.index[0]
        if not candidate_idx.empty
        else (morning_intensity.index[0] if not morning_intensity.empty else None)
    )
    return wakeup_idx


def get_avg_biomarker_until_wakeup(group, wakeup_idx, col_name):
    biomarker = group[col_name].dropna()
    biomarker_until_wakeup = biomarker.loc[:wakeup_idx]
    return biomarker_until_wakeup.mean() if not biomarker_until_wakeup.empty else np.nan


def get_avg_sleep_biomarker_until_wakeup(group, col_name):
    sleep = group[col_name]
    # Sleep is when sleep_value is not NaN
    sleep_period = sleep[sleep.notna()]
    if sleep_period.empty:
        return np.nan
    # Find the last sleep index (i.e., just before wake)
    wakeup_idx = sleep_period.index[-1]
    biomarker_until_wakeup = group.loc[:wakeup_idx, col_name].dropna()
    return biomarker_until_wakeup.mean() if not biomarker_until_wakeup.empty else np.nan


# NOTE: Extracting up with a list of features: fasting biomarkers (Libre and Dexcom)
def estimate_daily_fasting_biomarkers(ts_df: pd.DataFrame) -> pd.DataFrame:
    pid = ts_df["PID"].unique()[0]
    # Ensure datetime index
    ts_df = ts_df.copy()
    if not isinstance(ts_df.index, pd.DatetimeIndex):
        ts_df.index = pd.to_datetime(ts_df.index)

    results = []
    for date, group in ts_df.groupby(ts_df.index.date):
        # Find first fasting for Libre GL

        intensity = group["METs"].dropna()
        if intensity.empty:
            # tqdm.write(f"No intensity data available for PID:{pid} {date}, skipping...")
            continue  # Skip if no intensity data available
        wakeup_idx = get_wakeup_idx(intensity)
        if wakeup_idx is None:
            # tqdm.write(f"No fasting intensity found for PID:{pid} {date}, skipping...")
            continue
        # avg_libre = get_avg_biomarker_until_wakeup(group, wakeup_idx, "Libre GL")
        # avg_dexcom = get_avg_biomarker_until_wakeup(group, wakeup_idx, "Dexcom GL")
        # avg_HR = get_avg_biomarker_until_wakeup(group, wakeup_idx, "HR")
        avg_libre = get_avg_sleep_biomarker_until_wakeup(group, "Libre GL")
        avg_dexcom = get_avg_sleep_biomarker_until_wakeup(group, "Dexcom GL")
        avg_HR = get_avg_sleep_biomarker_until_wakeup(group, "HR")
        results.append(
            {
                "date": pd.to_datetime(date),
                "wakeup_time": wakeup_idx,
                "wakeup_intensity": intensity[wakeup_idx],
                "avg_fasting_libre": avg_libre,
                "avg_fasting_dexcom": avg_dexcom,
                "avg_fasting_HR": avg_HR,
            }
        )
    result_df = pd.DataFrame(results).set_index("date")
    return result_df


def calc_daily_intakes(meal_df: pd.DataFrame) -> pd.DataFrame:
    # Calculate daily intakes for each subject
    meal_df.index = pd.to_datetime(meal_df.index)
    meal_df["date"] = meal_df.index.date
    daily_intakes = (
        meal_df.groupby(["PID", "date"])
        .agg(
            total_calories=("Calories", "sum"),
            total_carbs=("Carbs", "sum"),
            total_protein=("Protein", "sum"),
            total_fat=("Fat", "sum"),
            total_fiber=("Fiber", "sum"),
        )
        .reset_index()
    )
    daily_intakes["total_ratio"] = daily_intakes["total_carbs"] / (
        daily_intakes["total_carbs"]
        + daily_intakes["total_protein"]
        + daily_intakes["total_fat"]
        + daily_intakes["total_fiber"]
    )
    daily_intakes = daily_intakes.set_index("date")
    return daily_intakes


def estimate_dining_periods(ts_df: pd.DataFrame, meal_df) -> pd.DataFrame:
    # Ensure datetime index
    ts_df = ts_df.copy()
    if not isinstance(ts_df.index, pd.DatetimeIndex):
        ts_df.index = pd.to_datetime(ts_df.index)

    results = []
    for date, group in ts_df.groupby(ts_df.index.date):
        # Calculate AUC for each 15-minute interval in this day
        auc_per_15min = []
        for interval_start, interval_group in group.groupby(group.index.floor("15T")):
            values = interval_group["Libre GL"].dropna()
            if len(values) > 1:
                auc = np.trapz(values, dx=1)
            else:
                auc = 0
            auc_per_15min.append((interval_start, auc))
        # Compute the average AUC across all 15-minute intervals
        if auc_per_15min:
            avg_auc = np.mean([auc for _, auc in auc_per_15min])
        else:
            avg_auc = 0
        results.append(
            {
                "date": pd.to_datetime(date),
                "top10_avg_auc": avg_auc // 10,
            }
        )
    result_df = pd.DataFrame(results).set_index("date")
    return result_df


def estimate_dining_periods_accuracy(
    ts_df: pd.DataFrame,
    meal_df,
    interval: str = "15T",
    top_n: int = 10,
    meal_window_hours: float = 3.0,
) -> pd.DataFrame:
    """
    Estimate dining periods accuracy by comparing top-N AUC intervals to meal times.

    Args:
        ts_df (pd.DataFrame): Time series dataframe.
        meal_df (pd.DataFrame): Meal dataframe.
        interval (str): Interval for grouping (e.g., '15T', '1H').
        top_n (int): Number of top intervals to consider.
        meal_window_hours (float): Window (in hours) after meal time to consider as aligned.

    Returns:
        pd.DataFrame: Results with alignment info.
    """
    ts_df = ts_df.copy()
    if not isinstance(ts_df.index, pd.DatetimeIndex):
        ts_df.index = pd.to_datetime(ts_df.index)

    results = []
    for date, group in ts_df.groupby(ts_df.index.date):
        # Calculate AUC for each interval in this day
        auc_per_interval = []
        for interval_start, interval_group in group.groupby(
            group.index.floor(interval)
        ):
            values = interval_group["Libre GL"].dropna()
            if len(values) > 1:
                auc = np.trapz(values, dx=1)
            else:
                auc = 0
            auc_per_interval.append((interval_start, auc))
        # Sort by AUC descending and take top N
        auc_per_interval_sorted = sorted(
            auc_per_interval, key=lambda x: x[1], reverse=True
        )[:top_n]
        # Prepare meal times for this date (+window)
        if meal_df is not None and not meal_df.empty:
            meal_times = meal_df.index
        else:
            meal_times = pd.to_datetime([])
        for interval_start, auc in auc_per_interval_sorted:
            aligned = False
            for meal_time in meal_times:
                if (
                    meal_time
                    <= interval_start
                    <= meal_time + pd.Timedelta(hours=meal_window_hours)
                ):
                    aligned = True
                    break
            if meal_times.date not in meal_df.index.date:
                aligned = True
            results.append(
                {
                    "date": pd.to_datetime(date),
                    "interval_start": interval_start,
                    "auc": auc,
                    "align_with_dining_time": aligned,
                }
            )
    result_df = pd.DataFrame(results).set_index("date")
    return result_df


def calc_nrmse(y_true, y_pred):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) for each label.
    Returns an array of NRMSE values, one for each label/column.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    n_labels = y_true.shape[1]
    nrmse = []
    for i in range(n_labels):
        mse = np.mean((y_true[:, i] - y_pred[:, i]) ** 2)
        norm = np.mean(y_true[:, i] ** 2) + 1e-10
        nrmse.append(np.sqrt(mse / norm))
    return np.array(nrmse)


def data_split(merged_df, test_size=0.2, random_state=2025, approach="per person"):
    # Train/test split
    if approach == "random":
        train_df, test_df = train_test_split(
            merged_df, test_size=test_size, random_state=random_state
        )
    elif approach == "per person":
        # Split by unique PIDs
        unique_pids = merged_df["PID"].unique()
        train_pids, test_pids = train_test_split(
            unique_pids, test_size=test_size, random_state=random_state
        )
        train_df = merged_df[merged_df["PID"].isin(train_pids)]
        test_df = merged_df[merged_df["PID"].isin(test_pids)]
    elif approach == "per date":
        # Split by unique dates
        unique_dates = merged_df.index.unique()
        train_dates, test_dates = train_test_split(
            unique_dates, test_size=test_size, random_state=random_state
        )
        train_df = merged_df[merged_df.index.isin(train_dates)]
        test_df = merged_df[merged_df.index.isin(test_dates)]
    else:
        raise ValueError("Unsupported split approach. Use 'random'.")
    # Features and targets
    features = [
        "wakeup_intensity",
        "avg_fasting_libre",
        "avg_fasting_dexcom",
        "avg_fasting_HR",
        "top10_avg_iauc",
    ]
    labels = ["total_calories", "total_carbs", "total_ratio"]
    return train_df[features], test_df[features], train_df[labels], test_df[labels]


def train_and_predict(
    X_train, y_train, X_test, random_state=2025, model_name="xgboost"
):
    model_name = model_name.lower()
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    if model_name == "xgboost":
        # XGBoost regressor for multitask (fit 4 outputs)
        base_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=2,
            learning_rate=0.01,
            random_state=random_state,
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)
    elif model_name == "linear":
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)
        model = MultiOutputRegressor(LinearRegression())
        model.fit(X_train, y_train)
    elif model_name == "mlp":
        X_train.fillna(0, inplace=True)
        X_test.fillna(0, inplace=True)
        model = MultiOutputRegressor(
            MLPRegressor(
                hidden_layer_sizes=(512, 256, 128, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate="adaptive",
                max_iter=600,
                early_stopping=True,
                random_state=random_state,
            )
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model: {model_name}.")
    return y_scaler.inverse_transform(model.predict(X_test))


def main(interval="15T", top_n=5):
    ts_df, meal_df = generate_24H_CGMacros_dataset()
    ts_df.index = pd.to_datetime(ts_df.index)
    meal_df.index = pd.to_datetime(meal_df.index)
    merged_df = pd.DataFrame()
    dining_windows_align_df = pd.DataFrame()
    for pid in ts_df["PID"].unique():
        pid_fasting_df = estimate_daily_fasting_biomarkers(ts_df[ts_df["PID"] == pid])
        pid_daily_intakes_df = calc_daily_intakes(meal_df[meal_df["PID"] == pid])
        pid_dining_windows_df = estimate_dining_periods_accuracy(
            ts_df[ts_df["PID"] == pid],
            meal_df[meal_df["PID"] == pid],
            interval=interval,
            top_n=top_n,
            meal_window_hours=3,
        )
        dining_windows_align_df = pd.concat(
            [dining_windows_align_df, pid_dining_windows_df], axis=0
        )
        pid_dining_df = estimate_dining_periods(
            ts_df[ts_df["PID"] == pid], meal_df[meal_df["PID"] == pid]
        )
        pid_merged_df = pid_fasting_df.join(
            [pid_daily_intakes_df, pid_dining_df], how="inner"
        )
        pid_merged_df["top10_avg_iauc"] = (
            pid_merged_df["top10_avg_auc"] - pid_merged_df["avg_fasting_libre"]
        )
        merged_df = pd.concat([merged_df, pid_merged_df], axis=0)
    # Drop rows with missing targets
    merged_df = merged_df.dropna(
        subset=["total_calories", "total_protein", "total_fat", "total_carbs"]
    )

    # Train/test split
    split_approach = "per date"  # ["random", "per person", "per date"]
    X_train, X_test, y_train, y_test = data_split(merged_df, approach=split_approach)

    # NOTE: Try with Linear Regression
    model_name = "linear"
    y_pred = train_and_predict(X_train, y_train, X_test, model_name=model_name)
    nrmse = calc_nrmse(y_test, y_pred)
    print(
        f"Data Split: {split_approach};",
        f"Model: {model_name};",
        f"NRMSE {list(y_test.columns)}:",
        [round(v, 3) for v in nrmse],
    )

    # NOTE: Try with XGBoost
    model_name = "xgboost"
    y_pred = train_and_predict(X_train, y_train, X_test, model_name=model_name)
    nrmse = calc_nrmse(y_test, y_pred)
    print(
        f"Data Split: {split_approach};",
        f"Model: {model_name};",
        f"NRMSE {list(y_test.columns)}:",
        [round(v, 3) for v in nrmse],
    )

    # NOTE: Try with MLP
    model_name = "mlp"
    y_pred = train_and_predict(X_train, y_train, X_test, model_name=model_name)
    nrmse = calc_nrmse(y_test, y_pred)
    print(
        f"Data Split: {split_approach};",
        f"Model: {model_name};",
        f"NRMSE {list(y_test.columns)}:",
        [round(v, 3) for v in nrmse],
    )

    # NOTE: Measuring how much agreement
    # Calculate agreement metrics for dining window alignmen
    align_arr = dining_windows_align_df["align_with_dining_time"].to_numpy()
    # Here, we compare to a baseline where we always predict "aligned" (1)
    accuracy = align_arr.sum() / len(align_arr) * 100
    print(f"Dining window alignment: accuracy={accuracy:.2f}%")
    pdb.set_trace()


if __name__ == "__main__":
    main()
    # for top_n in [15, 20, 60]:
    #     for interval in ["15T", "30T", "45T", "1H"]:
    #         main(interval=interval, top_n=top_n)
    #         print(f"Above it interval={interval}, top_n={top_n}")
