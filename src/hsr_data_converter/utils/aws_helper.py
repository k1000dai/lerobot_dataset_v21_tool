"""
AWS S3 helper class for HSR data converter
"""

import json
import os
from pathlib import Path

import boto3
import botocore.exceptions
from botocore.exceptions import ClientError


class AWSHelper:
    """Helper class for AWS operations including S3 and Secrets Manager."""

    def __init__(
        self,
        secret_name: str,
        region_name: str = "ap-northeast-1",
        endpoint_url: str = "https://s3.ap-northeast-1.wasabisys.com",
    ):
        """
        Initialize AWSHelper with AWS Secrets Manager credentials.

        Parameters
        ----------
        secret_name : str
            Name of secret in AWS Secrets Manager
        region_name : str, optional
            AWS region name, by default "ap-northeast-1"
        endpoint_url : str, optional
            Wasabi endpoint URL, by default "https://s3.ap-northeast-1.wasabisys.com"
        """
        self.secret_name = secret_name
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self._credentials: None | dict[str, str] = None
        self._s3_client = None

    def _get_credentials(self) -> dict[str, str]:
        """
        Get Wasabi credentials from AWS Secrets Manager.

        Returns
        -------
        Dict[str, str]
            Dictionary containing access key and secret key

        Raises
        ------
        Exception
            If failed to retrieve credentials
        """
        if self._credentials is None:
            try:
                client = boto3.client("secretsmanager", region_name=self.region_name)
                response = client.get_secret_value(SecretId=self.secret_name)
                secret = json.loads(response["SecretString"])
                self._credentials = {
                    "access_key": secret["WASABI_ACCESS_KEY_ID"],
                    "secret_key": secret["WASABI_SECRET_ACCESS_KEY"],
                }
            except botocore.exceptions.BotoCoreError as e:
                print(f"Error retrieving secret: {e}")
                raise
        if self._credentials is None:
            raise RuntimeError("Failed to retrieve credentials")
        return self._credentials

    def _get_s3_client(self):
        """
        Create and return Wasabi S3 client.

        Returns
        -------
        boto3.client
            S3 client configured for Wasabi
        """
        if self._s3_client is None:
            credentials = self._get_credentials()
            self._s3_client = boto3.client(
                "s3",
                aws_access_key_id=credentials["access_key"],
                aws_secret_access_key=credentials["secret_key"],
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
            )
        return self._s3_client

    def check_meta_file_exists(self, bucket_name: str, s3_key: str) -> bool:
        """
        Check if meta.json file exists in the S3 directory.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket
        s3_key : str
            S3 key prefix (directory path)

        Returns
        -------
        bool
            True if meta.json exists, False otherwise
        """
        s3_client = self._get_s3_client()
        meta_file_key = f"{s3_key.rstrip('/')}/meta.json"

        try:
            s3_client.head_object(Bucket=bucket_name, Key=meta_file_key)
            print(f"[INFO] Found {meta_file_key} in {bucket_name}")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print(f"[INFO] NOT Found {meta_file_key} in {bucket_name}")
                return False
            else:
                raise

    def download_s3_directory(
        self, bucket_name: str, s3_prefix: str, local_dir: Path
    ) -> None:
        """
        Download all files from the specified directory (prefix) in S3 while preserving folder structure.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket
        s3_prefix : str
            Directory path in S3 (e.g., "tmc/rosbags/lego/")
        local_dir : Path
            Local directory to save the downloaded files
        """
        s3_client = self._get_s3_client()
        local_dir.mkdir(parents=True, exist_ok=True)

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]

                    # Calculate relative path from the prefix
                    relative_path = s3_key[len(s3_prefix) :].lstrip("/")
                    if relative_path:  # Skip empty paths (directory itself)
                        local_file_path = local_dir / relative_path
                        local_file_path.parent.mkdir(parents=True, exist_ok=True)

                        print(f"Downloading {s3_key} to {local_file_path} ...")
                        s3_client.download_file(
                            bucket_name, s3_key, str(local_file_path)
                        )

    def upload_directory_to_s3(
        self, local_dir: Path, bucket_name: str, s3_prefix: str
    ) -> None:
        """
        Upload files from a local directory to an S3 bucket, preserving folder structure.

        Parameters
        ----------
        local_dir : Path
            Path to the local directory to upload
        bucket_name : str
            Name of the destination S3 bucket
        s3_prefix : str
            Directory path within S3 (e.g., "tmc/rosbags/lego/")
        """
        s3_client = self._get_s3_client()

        for root, _, files in os.walk(local_dir):
            print(f"root: {root}")
            print(f"files: {files}")
            for file in files:
                local_file_path = Path(root) / file
                relative_path = local_file_path.relative_to(local_dir)
                s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}".replace("\\", "/")

                print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key} ...")
                s3_client.upload_file(str(local_file_path), bucket_name, s3_key)

        print("Upload completed.")

    def upload_processing_complete(self, bucket_name: str, s3_prefix: str) -> None:
        """
        Upload a processing complete marker to S3.

        Parameters
        ----------
        bucket_name : str
            Name of the destination S3 bucket
        s3_prefix : str
            Directory path within S3 (e.g., "tmc/rosbags/lego/")
        """
        s3_client = self._get_s3_client()

        # Create empty marker file
        complete_key = f"{s3_prefix.rstrip('/')}/.processing_complete"

        try:
            s3_client.put_object(Bucket=bucket_name, Key=complete_key, Body=b"")
            print(f"Uploaded empty marker to s3://{bucket_name}/{complete_key}")
        except ClientError as e:
            print(f"Error uploading to S3: {e}")
            raise

    def list_s3_directory(self, bucket_name: str, prefix: str) -> list[str]:
        """
        List all directories in the specified prefix in S3.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket
        prefix : str
            Directory path in S3 (e.g., "tmc/rosbags/lego/")

        Returns
        -------
        List[str]
            List of directory prefixes
        """
        s3_client = self._get_s3_client()

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

        directories = []
        for page in pages:
            if "CommonPrefixes" in page:
                for obj in page["CommonPrefixes"]:
                    directories.append(obj["Prefix"])

        return directories

    def list_template_directories(
        self, bucket_name: str, batch_prefix: str
    ) -> list[str]:
        """
        List all template-* directories within a batch directory.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket
        batch_prefix : str
            Batch directory path in S3

        Returns
        -------
        List[str]
            List of template directory paths
        """
        s3_client = self._get_s3_client()

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=bucket_name, Prefix=batch_prefix, Delimiter="/"
        )

        template_directories = []
        for page in pages:
            if "CommonPrefixes" in page:
                for obj in page["CommonPrefixes"]:
                    directory_name = obj["Prefix"].rstrip("/").split("/")[-1]
                    # Filter for template-* directories
                    if directory_name.startswith("template-"):
                        template_directories.append(obj["Prefix"])

        return template_directories

    def find_rosbag_directories(
        self, bucket_name: str, root_prefix: str
    ) -> list[dict[str, str | list[str]]]:
        """
        Find all directories containing meta.json and *.bag files recursively.

        Parameters
        ----------
        bucket_name : str
            Name of the S3 bucket
        root_prefix : str
            Root directory path in S3 to search from

        Returns
        -------
        List[Dict[str, str]]
            List of dictionaries containing directory info:
            [{"directory": "path/to/dir/", "meta_file": "meta.json", "bag_files": ["*.bag"]}]
        """
        s3_client = self._get_s3_client()

        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket_name, Prefix=root_prefix)

        # Dictionary to group files by directory
        dir_files: dict[str, dict[str, str | list[str] | None]] = {}

        for page in pages:
            if "Contents" in page:
                for obj in page["Contents"]:
                    s3_key = obj["Key"]
                    file_name = s3_key.split("/")[-1]
                    dir_path = "/".join(s3_key.split("/")[:-1]) + "/"

                    if dir_path not in dir_files:
                        dir_files[dir_path] = {"meta": None, "bags": []}

                    # Check for meta.json
                    if file_name == "meta.json":
                        dir_files[dir_path]["meta"] = file_name

                    # Check for .bag files
                    elif file_name.endswith(".bag"):
                        bags = dir_files[dir_path]["bags"]
                        if isinstance(bags, list):
                            bags.append(file_name)

        # Filter directories that have both meta.json and at least one .bag file
        rosbag_directories = []
        for dir_path, files in dir_files.items():
            if files["meta"] and files["bags"]:
                rosbag_directories.append(
                    {
                        "directory": dir_path,
                        "meta_file": files["meta"],
                        "bag_files": files["bags"],
                    }
                )
                print(
                    f"[INFO] Found rosbag directory: {dir_path} with {len(files['bags'])} bag files"
                )

        return rosbag_directories
