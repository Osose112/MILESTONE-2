import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
import joblib


def preprocess_data(df):
    """
    Comprehensive data preprocessing function
    """
    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Categorical encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Detect skewness and transform if needed
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    skewed_columns = df[numeric_columns].apply(lambda x: np.abs(x.skew()) > 1)

    for col in skewed_columns[skewed_columns].index:
        df[col] = np.log1p(df[col])

    # Outlier handling using RobustScaler
    scaler = RobustScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df, label_encoders


def convert_to_discrete_classes(series, n_bins=5):
    """
    Convert continuous target to discrete classes using quantile binning
    """
    # Reshape to 2D array
    series_2d = series.values.reshape(-1, 1)

    # Use quantile-based binning
    bins = np.quantile(series, np.linspace(0, 1, n_bins + 1))
    discrete_classes = np.digitize(series, bins[1:-1]) - 1

    return discrete_classes


def main():
    # Load new test data
    new_test_df = pd.read_csv('data/processed_amazon_purchase_data.csv')

    # Preprocess the new test data
    new_test_df, _ = preprocess_data(new_test_df)

    # Convert Purchase to discrete classes
    y_new_test = convert_to_discrete_classes(new_test_df['Purchase'])
    X_new_test = new_test_df.drop('Purchase', axis=1)

    # Load PCA transformer
    pca = joblib.load('model/pca_transformer.joblib')

    # Apply PCA transformation
    X_new_test_reduced = pca.transform(X_new_test)

    # Load the trained model
    final_model = joblib.load('model/sgd_classifier_model.joblib')

    # Predict on new test data
    y_new_pred = final_model.predict(X_new_test_reduced)

    # Evaluate the model on new test data
    print("\nNew Test Data Performance Metrics:")
    print("Classification Report:\n", classification_report(y_new_test, y_new_pred))

    # Confusion Matrix Visualization for new test data
    plt.figure(figsize=(10, 8))
    cm_new = confusion_matrix(y_new_test, y_new_pred)
    sns.heatmap(cm_new, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - New Test Data')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix_new_test.png')
    plt.close()

    # Additional insights
    print("\nConfusion Matrix Interpretation:")
    print("Diagonal elements represent correct predictions.")
    print("Off-diagonal elements show misclassifications.")


if __name__ == "__main__":
    main()