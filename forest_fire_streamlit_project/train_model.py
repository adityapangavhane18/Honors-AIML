
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_model():
    df = pd.read_csv("data/forestfires.csv")

    month_encoder = LabelEncoder()
    day_encoder = LabelEncoder()

    df["month"] = month_encoder.fit_transform(df["month"])
    df["day"] = day_encoder.fit_transform(df["day"])

    df["log_area"] = np.log1p(df["area"])

    X = df.drop(["area", "log_area"], axis=1)
    y = df["log_area"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model Performance")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    joblib.dump(model, "forest_fire_model.pkl")

    return model, month_encoder, day_encoder

if __name__ == "__main__":
    train_and_save_model()
