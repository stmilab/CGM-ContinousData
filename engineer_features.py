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


def get_wakeup_idx(intensity_series):
    # Filter intensity index to between 6AM and 10AM
    mask = (intensity_series.index.time >= pd.to_datetime("06:00").time()) & (
        intensity_series.index.time <= pd.to_datetime("10:00").time()
    )
    morning_intensity = intensity_series[mask]
    candidate_idx = morning_intensity[morning_intensity == morning_intensity.max()]
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

        intensity = group["Intensity"].dropna()
        if intensity.empty:
            tqdm.write(f"No intensity data available for PID:{pid} {date}, skipping...")
            continue  # Skip if no intensity data available
        wakeup_idx = get_wakeup_idx(intensity)
        if wakeup_idx is None:
            tqdm.write(f"No fasting intensity found for PID:{pid} {date}, skipping...")
            continue
        avg_libre = get_avg_biomarker_until_wakeup(group, wakeup_idx, "Libre GL")
        avg_dexcom = get_avg_biomarker_until_wakeup(group, wakeup_idx, "Dexcom GL")
        avg_HR = get_avg_biomarker_until_wakeup(group, wakeup_idx, "HR")
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


def estimate_dining_periods(ts_df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime index
    ts_df = ts_df.copy()
    if not isinstance(ts_df.index, pd.DatetimeIndex):
        ts_df.index = pd.to_datetime(ts_df.index)

    results = []
    for date, group in ts_df.groupby(ts_df.index.date):
        # Calculate AUC for each hour in this day
        auc_per_hour = []
        for hour_start, hour_group in group.groupby(group.index.floor("H")):
            values = hour_group["Libre GL"].dropna()
            if len(values) > 1:
                auc = np.trapz(values, dx=1)
            else:
                auc = 0
            auc_per_hour.append((hour_start, auc))
        # Get top 10 hours by AUC
        top_10 = sorted(auc_per_hour, key=lambda x: x[1], reverse=True)[:10]
        daily_top10_auc_sum = sum(auc for _, auc in top_10)
        results.append(
            {
                "date": pd.to_datetime(date),
                "top10_auc_sum": daily_top10_auc_sum,
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
        "top10_auc_sum",
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


def main():
    ts_df, meal_df = generate_24H_CGMacros_dataset()
    merged_df = pd.DataFrame()
    for pid in ts_df["PID"].unique():
        pid_fasting_df = estimate_daily_fasting_biomarkers(ts_df[ts_df["PID"] == pid])
        pid_daily_intakes_df = calc_daily_intakes(meal_df[meal_df["PID"] == pid])
        pid_dining_df = estimate_dining_periods(ts_df[ts_df["PID"] == pid])
        pid_merged_df = pid_fasting_df.join(
            [pid_daily_intakes_df, pid_dining_df], how="inner"
        )
        merged_df = pd.concat([merged_df, pid_merged_df], axis=0)
    # Drop rows with missing targets
    merged_df = merged_df.dropna(
        subset=["total_calories", "total_protein", "total_fat", "total_carbs"]
    )

    # Train/test split
    split_approach = "random"  # ["random", "per person", "per date"]
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


if __name__ == "__main__":
    main()
