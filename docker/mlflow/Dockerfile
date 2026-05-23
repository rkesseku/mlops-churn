# Custom MLflow image with the database and object-store drivers MLflow needs
# to talk to Postgres + MinIO. The official ghcr.io/mlflow/mlflow image
# ships MLflow only; the connectors are intentionally left to the user.

FROM ghcr.io/mlflow/mlflow:v2.17.2

# psycopg2-binary: Postgres driver (the SQLAlchemy backend store needs it)
# boto3:          AWS SDK; MLflow uses it to upload artifacts to MinIO/S3
RUN pip install --no-cache-dir \
    psycopg2-binary==2.9.10 \
    boto3==1.35.60
