import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def train_ml_model():
    data = {
        "unique_values_ratio": [0.9, 0.01, 0.3, 0.02, 0.6, 0.1, 0.02, 0.05],
        "variance": [0.8, 0.1, 0.5, 0.02, 0.2, 0.05, 0.01, 0.03],
        "missing_values_ratio": [0.1, 0.05, 0.01, 0.2, 0.3, 0.02, 0.1, 0.15],
        "skewness": [0.5, 0.02, 0.3, 0.01, 1.2, 0.8, 0.01, 0.02],
        "type": ["numerical", "categorical", "numerical", "categorical", "datetime", "numerical", "categorical",
                 "categorical"]
    }

    df_train = pd.DataFrame(data)

    #Encoding labels
    le = LabelEncoder()
    df_train["type_encoded"] = le.fit_transform(df_train["type"])

    X = df_train.drop(columns=["type", "type_encoded"])
    y = df_train["type_encoded"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le


ml_model, label_encoder = train_ml_model()

def extract_feat(df):
    feats = []

    for col in df.columns:
        unique_values_ratio = df[col].nunique() / len(df)
        variance = df[col].var() if df[col].dtype in ["int64", "float64"] else 0
        missing_values_ratio = df[col].isnull().sum() / len(df)
        skewness = df[col].skew() if df[col].dtype in ["int64", "float64"] else 0

        feats.append([unique_values_ratio, variance, missing_values_ratio, skewness])

    return pd.DataFrame(feats, columns=["unique_values_ratio", "variance", "missing_values_ratio", "skewness"])

def predict_columns(df):
    feature_df = extract_feat(df)
    predicts = ml_model.predict(feature_df)
    column_types = label_encoder.inverse_transform(predicts)

    return dict(zip(df.columns, column_types))

def suggestion_of_charts(predictions):
    chart_suggestions = {}

    for col, col_type in predictions.items():
        if col_type == "numerical":
            chart_suggestions[col] = "Histogram or Correlation Matrix"
        elif col_type == "categorical":
            chart_suggestions[col] = "Bar Chart or Pie Chart"
        elif col_type == "datetime":
            chart_suggestions[col] = "Time-Series Line Plot"

    return chart_suggestions

def generate_best_chart(df, predictions):

    plt.figure(figsize=(10, 6))

    for col, col_type in predictions.items():
        if col_type == "numerical":
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f"Histogram of {col}")
        elif col_type == "categorical":
            sns.countplot(x=df[col])
            plt.title(f"Bar Chart of {col}")
            plt.xticks(rotation=45)
        elif col_type == "datetime":
            num_cols = df.select_dtypes(include=["number"]).columns
            if num_cols.any():
                plt.plot(df[col], df[num_cols[0]])
                plt.title(f"Time-Series Plot ({col} vs {num_cols[0]})")
                plt.xlabel(col)
                plt.ylabel(num_cols[0])
        plt.show()

def main(file_path):
    df = pd.read_csv(file_path, parse_dates=True)

    if df.empty:
        print("Error, file is empty")
        return

    df.info()
    predictions = predict_columns(df)
    print("\nPredicted Column Types:", predictions)

    suggested_charts = suggestion_of_charts(predictions)
    print("\nSuggested Chart Types:", suggested_charts)

    generate_best_chart(df, predictions)












