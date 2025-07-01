import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded. Shape: {self.df.shape}")

    def split_data(self, target_column="AnyFraud", test_size=0.2, random_state=42):
        X = self.df.drop(columns=[target_column, "CustomerId"], errors="ignore")
        y = self.df[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print("Data split completed.")

    def train_and_log_model(self, model_name="logistic_regression"):
        if model_name == "logistic_regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
            params = {"model_type": "LogisticRegression", "max_iter": 1000}
        elif model_name == "random_forest":
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            params = {"model_type": "RandomForest", "n_estimators": 100}
        else:
            raise ValueError("Unsupported model name")

        with mlflow.start_run(run_name=model_name):
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            # Metrics
            acc = accuracy_score(self.y_test, predictions)
            prec = precision_score(self.y_test, predictions, zero_division=0)
            rec = recall_score(self.y_test, predictions, zero_division=0)
            f1 = f1_score(self.y_test, predictions, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, predictions)

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc
            })

            # Log the model
            mlflow.sklearn.log_model(model, "model")

            print(f"Model {model_name} trained and logged with MLflow.")

    def tune_model(self, model, param_grid, scoring="f1", cv=5):
        """Tune hyperparameters using GridSearchCV."""
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
        grid_search.fit(self.X_train, self.y_train)
        print("Best params:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)
        return grid_search.best_estimator_
    
    def log_model(self, model, model_name):
        mlflow.sklearn.log_model(model, model_name)
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None
        self.evaluate_model(y_pred, y_prob)
    
    def evaluate_model(self, y_pred, y_prob=None):
        """Evaluate model performance and print metrics."""
        print("Evaluation Metrics:")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Precision:", precision_score(self.y_test, y_pred, zero_division=0))
        print("Recall:", recall_score(self.y_test, y_pred, zero_division=0))
        print("F1 Score:", f1_score(self.y_test, y_pred, zero_division=0))

        if y_prob is not None and len(set(self.y_test)) > 1:
            try:
                roc_auc = roc_auc_score(self.y_test, y_prob)
                print("ROC AUC:", roc_auc)
                mlflow.log_metric("roc_auc", roc_auc)
            except ValueError:
                print("ROC AUC: Could not be calculated due to single class in y_test.")
        else:
            print("ROC AUC: Not available (predict_proba missing or only one class).")
   


