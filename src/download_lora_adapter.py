import os
from pathlib import Path
from urllib.parse import urlparse

import boto3

# Use the following code to connect using Wasabi profile from .aws/credentials file
session = boto3.Session(profile_name="default")
credentials = session.get_credentials()
#
aws_access_key_id = credentials.access_key
aws_secret_access_key = credentials.secret_key

# aws_access_key_id = os.getenv("WASABI_ACCESS_KEY")
# aws_secret_access_key = os.getenv("WASABI_SECRET_ACCESS_KEY")
adapter_path = os.getenv("WASABI_LORA_ADAPTER_PATH")

print("WASABI_ACCESS_KEY: ", aws_access_key_id)
print("WASABI_SECRET_ACCESS_KEY: ", aws_secret_access_key)
print("WASABI_LORA_ADAPTER_PATH: ", adapter_path)

# Endpoint is determined when bucket is created
ENDPOINT_URL = 'https://s3.eu-west-1.wasabisys.com'

s3 = boto3.client('s3',
                  endpoint_url=ENDPOINT_URL,  # s3.wasabisys.com ?
                  aws_access_key_id=aws_access_key_id,
                  aws_secret_access_key=aws_secret_access_key)


def download_s3_folder(s3_uri, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        s3_uri: the s3 uri to the top level of the files you wish to download
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.resource("s3",
                  endpoint_url=ENDPOINT_URL,
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)
    bucket = s3.Bucket(urlparse(s3_uri).hostname)
    s3_path = urlparse(s3_uri).path.lstrip('/')
    if local_dir is not None:
        local_dir = Path(local_dir)
    for obj in bucket.objects.filter(Prefix=s3_path):
        target = Path(obj.key) if local_dir is None else local_dir / Path(obj.key).relative_to(s3_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, str(target))

if __name__ == "__main__":
    download_s3_folder(adapter_path)