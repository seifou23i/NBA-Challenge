import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.nba_longevity import NBALongevity

parser = argparse.ArgumentParser(description="Train NBALongevity model")
parser.add_argument("--data_path", type=str, required=True, help="Path to the NBA data CSV file")
parser.add_argument("--test_size", type=float, default=0.1, help="Proportion of data for testing (default: 0.2)")
parser.add_argument("--model_path", type=str, default="./data/outputs/models/log_regression.h5",
                    help="Path to save the trained LogisticRegression model (default: ../data/outputs/models/log_regression.h5)")
parser.add_argument("--scaler_path", type=str, default="./data/outputs/models/std_scaler.h5",
                    help="Path to save the trained StandardScaler (default: ./data/outputs/models/std_scaler.h5)")
parser.add_argument("--metrics", nargs="+", default=["precision", "recall"],
                    help="List of metrics to calculate during evaluation (default: precision, recall)")
args = parser.parse_args()


def main():

    # Load data
    data = pd.read_csv(args.data_path)

    # Data Preprocessing
    data.loc[data.isna().any(axis=1), "3P%"] = 0  # fill NaN values with 0
    data = data.drop_duplicates()  # drop duplicate rows based on all columns.
    # drop duplicate player names in the data where their target values differ
    data = data.drop_duplicates(subset=data.columns.difference(["TARGET_5Yrs"]), keep=False)
    # retaining the most recent data for each player
    data = data.sort_values(["Name", "GP"], ascending=False).drop_duplicates(subset=["Name"], keep='first')

    # Split data into features (X) and target variable (y)
    X = data[['GP', 'PTS', 'OREB']]
    y = data["TARGET_5Yrs"]

    # Split data into training and testing sets (for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # Create an NBALongevity instance
    nba_longevity = NBALongevity()

    # Train the model
    nba_longevity.fit(X_train, y_train)

    # Evaluate model performance
    print(nba_longevity.evaluate(X_test, y_test, metrics=args.metrics))

    # Save the model and scaler
    nba_longevity.save_models(args.model_path, args.scaler_path)

    print(f"Model and scaler saved successfully to: \n - Model: {args.model_path} \n - Scaler: {args.scaler_path}")
    print(pd.DataFrame(nba_longevity.model.coef_, columns=X.columns, index=["Logistic Regression coef"]).T)


if __name__ == "__main__":
    main()
