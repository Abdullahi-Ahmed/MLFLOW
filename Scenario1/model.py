import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
from xgboost import XGBRegressor
from sklearn import set_config
from sklearn.pipeline import Pipeline
import sys

if __name__ == '__main__':
    transformers = []
    df_loaded = pd.read_csv("Walmart_Store_sales.csv")
    numerical_pipeline = Pipeline(steps=[
        ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
        ("imputer", SimpleImputer(strategy="mean"))
    ])
    transformers.append(("numerical", numerical_pipeline, ["CPI", "Fuel_Price", "Unemployment", "Store", "Temperature"]))
    one_hot_encoder = OneHotEncoder(handle_unknown="ignore")
    transformers.append(("onehot", one_hot_encoder, ["Date", "Holiday_Flag"]))
    preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)
    standardizer = StandardScaler()

    split_X = df_loaded.drop(['Weekly_Sales'], axis=1)
    split_y = df_loaded['Weekly_Sales']

    # Split out train data
    X_train, split_X_rem, y_train, split_y_rem = train_test_split(
        split_X, 
        split_y, 
        train_size=0.6, 
        random_state=979224757)

    # Split remaining data equally for validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        split_X_rem, 
        split_y_rem, 
        test_size=0.5, 
        random_state=979224757)

    set_config(display='diagram')

    xgb_regressor = XGBRegressor(
    colsample_bytree=0.6669908680393172,
    learning_rate=0.22560423988961822,
    max_depth=3,
    min_child_weight=7,
    n_estimators=328,
    n_jobs=100,
    subsample=0.22601343271363417,
    verbosity=0,
    random_state=979224757,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("standardizer", standardizer),
        ("regressor", xgb_regressor),
    ])

    # Create a separate pipeline to transform the validation dataset. This is used for early stopping.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("standardizer", standardizer),
    ])

    mlflow.sklearn.autolog(disable=True)
    X_val_processed = pipeline.fit_transform(X_val, y_val)

    model

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(log_input_examples=True, silent=True)
    
    learning_rate = float(sys.argv[1]) if len(sys.argv) > 0.0001  else 0.5
    with mlflow.start_run(run_name="xgboost") as mlflow_run:
        model.fit(X_train, y_train, regressor__early_stopping_rounds=5, regressor__eval_set=[(X_val_processed,y_val)], regressor__verbose=False)
        
        # Training metrics are logged by MLflow autologging
        # Log metrics for the validation set
        xgb_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

        # Log metrics for the test set
        xgb_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

        # Display the logged metrics
        xgb_val_metrics = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
        xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}
