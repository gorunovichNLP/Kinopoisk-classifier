import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

client.create_model_version(
    name="rubert-sentiment",
    source="runs:/c99b205238c24a378ad8270c54409487/model",
    run_id="c99b205238c24a378ad8270c54409487",
)
print("Model version created")
