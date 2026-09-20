# for data manipulation
import pandas as pd
import os

# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# for model serialization and experiment tracking
import joblib
import mlflow
from huggingface_hub import HfApi, hf_hub_download

# Set MLflow tracking URI and experiment name
# If running in GitHub Actions, MLflow UI will be started locally
# If running locally, ensure public_url from ngrok is set
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if mlflow_tracking_uri:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

mlflow.set_experiment("Wellness_Tourism_Package_Prediction")     # same as the dev experimentation cell

# Get Hugging Face credentials from environment variables
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")

if not HF_TOKEN:
    print("Warning: HF_TOKEN not found in environment variables. Model/Dataset will not be pushed/pulled to/from Hugging Face.")
if not HF_MODEL_REPO:
    print("Warning: HF_MODEL_REPO not found in environment variables. Model will not be pushed to Hugging Face Model Hub.")
if not HF_DATASET_REPO:
    print("Warning: HF_DATASET_REPO not found in environment variables. Data splits will be loaded from local files instead of Hugging Face.")

# Load Xtrain/Xtest/ytrain/ytest from Hugging Face or local files
if HF_DATASET_REPO and HF_TOKEN:
    print(f"Attempting to download data splits from Hugging Face Dataset Hub: {HF_DATASET_REPO}")
    try:
        Xtrain_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="data_splits/Xtrain.csv", repo_type="dataset", token=HF_TOKEN)
        Xtest_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="data_splits/Xtest.csv", repo_type="dataset", token=HF_TOKEN)
        ytrain_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="data_splits/ytrain.csv", repo_type="dataset", token=HF_TOKEN)
        ytest_path = hf_hub_download(repo_id=HF_DATASET_REPO, filename="data_splits/ytest.csv", repo_type="dataset", token=HF_TOKEN)

        Xtrain = pd.read_csv(Xtrain_path)
        Xtest = pd.read_csv(Xtest_path)
        ytrain = pd.read_csv(ytrain_path).squeeze()
        ytest = pd.read_csv(ytest_path).squeeze()
        print("Data splits loaded successfully from Hugging Face.")
    except Exception as e:
        print(f"Error loading data splits from Hugging Face: {e}. Falling back to local files (if available).")
        # Fallback to local files if HF download fails
        Xtrain = pd.read_csv("Xtrain.csv")
        Xtest = pd.read_csv("Xtest.csv")
        ytrain = pd.read_csv("ytrain.csv").squeeze()
        ytest = pd.read_csv("ytest.csv").squeeze()
        print("Data splits loaded successfully from local files.")
else:
    print("Hugging Face dataset repo ID or token not provided. Loading data splits from local files.")
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
    print("Data splits loaded successfully from local files.")


numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]   # list all numerical feature names (same as in prep.py)

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation"
]   # list all categorical feature names (same as in prep.py)

# Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# Define hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__colsample_bytree': [0.6, 0.8],
    'xgbclassifier__colsample_bylevel': [0.6, 0.8],
    'xgbclassifier__learning_rate': [0.01, 0.1],
    'xgbclassifier__reg_lambda': [1, 2]
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)   # build the model pipeline by chaining preprocessor and xgb_model

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run,
    # so all experiments can be compared side by side in the MLflow UI
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    classification_threshold = 0.5    # Choose a classification threshold between 0 and 1.

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    # Save the model locally for MLflow artifact logging
    model_path = "tourism_project/deployment/best_tourism_package_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")

    # Push model to Hugging Face Model Hub if credentials are available
    if HF_TOKEN and HF_MODEL_REPO:
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_tourism_package_model.joblib",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
            commit_message="Add best_tourism_package_model.joblib"
        )
        print(f"Model successfully pushed to Hugging Face Model Hub: https://huggingface.co/models/{HF_MODEL_REPO}")
    else:
        print("Skipping Hugging Face Model Hub push due to missing credentials.")
