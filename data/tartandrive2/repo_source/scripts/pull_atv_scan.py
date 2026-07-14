from minio import Minio
from minio.error import S3Error

import os


# Minio client configuration
access_key = "nFNreHkFY2ca56vIHVaU"
secret_key = "IHnkXfe30TjJxkVpF8LuP8wQ7kWoMRrb5QpwcK7Z"
endpoint_url = "airlab-share-02.andrew.cmu.edu:9000"

minio_client = Minio(endpoint_url, access_key=access_key, secret_key=secret_key,secure=True, cert_check=False)

# Bucket name
bucket_name = 'tartandrive2'

minio_client.fget_object(bucket_name, 'vicky1_clean.pts', './assets/vicky1_clean.pts')
