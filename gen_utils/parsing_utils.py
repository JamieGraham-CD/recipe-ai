# utils/parsing_utils.py

import json 
import os
from logging import Logger
import logging
import pandas as pd # type: ignore
from datetime import datetime
from typing import Dict, Any, Union, Tuple
from dotenv import load_dotenv
from google.cloud import secretmanager, storage
from google.api_core.exceptions import NotFound, PermissionDenied
# from yolo.yolo_utils import yolo_inference_filter
import subprocess 

def configure_logging(id:str) -> Logger:
    """
    Configure logging for parser run.
    
    Args:
        id: Run job id for parser.
    """
    # Get current timestamp
    timestamp = datetime.now().isoformat()

    # Define log directory and file path
    log_dir = "./logging"
    os.makedirs(log_dir, exist_ok=True)  # Ensure the logging directory exists

    # Configure Logging
    LOG_FILE = f'''./logging/{id + "_" + timestamp + ".log"}''' # Change this to your preferred log file path
    logging.basicConfig(
        level=logging.DEBUG,  # Log all levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),  # Append mode
        ]
    )

    # get logger
    logger = logging.getLogger(__name__)

    return logger

def read_text_file(file_path:str) -> str:
    """
    Reads a text file at a given filepath.

    Input:
        file_path (str): Filepath where the txt file is stored
    Output:
        output_string (str): String to be read.
    """

    with open(file_path,"r", encoding="utf-8") as f:
        output_string = f.read()

    return output_string


def retrieve_secret(secret_name: str, project_id: str) -> dict:
    """
    Retrieve a secret from GCP Secret Manager and parse it as a dictionary, loading it in as environment variables.

    Args:
        secret_name (str): The name of the secret.
        project_id (str): The GCP project ID.

    Returns:
        dict: A dictionary containing the secret's key-value pairs.
    """
    # Get logger
    logger = logging.getLogger(__name__)

    # Create a Secret Manager client
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"

    # Fetch the secret
    response = client.access_secret_version(request={"name": secret_path})
    secret_json = response.payload.data.decode("UTF-8")  # Decode secret value

    # Parse the JSON secret
    secret_dict = json.loads(secret_json)

    # Set each secret as an environment variable
    for key, value in secret_dict.items():
        os.environ[key] = value  # Store in environment

    return secret_dict
    


def load_config(file_path:str) -> dict:
    """
    Safely load a JSON config. 

    Args:
        file_path (str): A file path containing the config json file. 
        logger (Logger): Python logger to handle errors.
    Return
        config (dict): Dictionary containing the configuration information.
    """
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        raise RuntimeError("Config file not found: './configs/config.json'")
    except json.JSONDecodeError:
        raise RuntimeError("Invalid JSON format in './configs/config.json'")
    return config


def validate_config(config:Dict[str,Any],logger:Logger):
    """
    Validates that all required properties in the config dictionary are filled out and have the correct data types.

    Args:
        config (Dict[str, Any]): The configuration dictionary.
        logger (logging.Logger): Logger instance for logging errors.

    Raises:
        ValueError: If any required field is missing, empty, or has an incorrect type.
    """

    # Define required keys with expected types
    required_keys_with_types = {
        "project_id": str,
        "secret_name": str,
        "max_chunk_size": int,
        "output_filepath": str,
        "input_type": str,
        "user_input_path": str,
        "dynamic_schema_prompt": str,
        "dynamic_schema_instructions": str,
        "system_prompt_path": str,
        "merge_system_prompt": str,
        "run_id": str,
        "max_llm_retries": int,
        "base_temperature": float,
        "llm-model-deployment-name": str
    }

    # Check for missing or empty values
    missing_keys = [key for key in required_keys_with_types if key not in config or config[key] in [None, "", []]]
    if missing_keys:
        logger.error(f"Config validation failed. Missing or empty fields: {missing_keys}")
        raise ValueError(f"Invalid config: Missing or empty fields: {missing_keys}")

    # Check for incorrect types
    type_mismatches = {
        key: expected_type for key, expected_type in required_keys_with_types.items()
        if key in config and not isinstance(config[key], expected_type)
    }
    if type_mismatches:
        logger.error(f"Config validation failed. Incorrect data types: {type_mismatches}")
        raise ValueError(f"Invalid config: Incorrect data types: {type_mismatches}")



def save_output_dataframe(output:dict, input_dict:dict) -> pd.DataFrame:
    """
    Save generated output dataframe from parsing output.

    Args: 
        output (dict): Output dictionary from parser.
        input_dict (dict): Input dictionary to parser
    Returns:
        df (pd.DataFrame): Dataframe of parser output.
    
    """
    # Check for output variable data type
    assert isinstance(output, dict), f"Expected a dictionary, but got {type(output).__name__}"

    # Define log directory and file path
    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the logging directory exists

    # Get current timestamp
    timestamp = datetime.now().isoformat()

    # Conditionally handle chunking
    if "chunked_responses" not in output:
        # If output is a single dict, we can immediately cast to DataFrame
        df = pd.DataFrame([output])
        output_filepath = f'''{input_dict['output_filepath'].replace(".csv","_" + timestamp + ".csv")}'''
        df.to_csv(input_dict['output_filepath'], index=False)
    else:
        # This means we have multiple partial responses, flatten them
        partial_list = output["chunked_responses"]
        df = pd.DataFrame(partial_list)
        output_filepath = f'''{input_dict['output_filepath'].replace(".csv","_chunked_" + timestamp + ".csv")}'''

    if input_dict.get("timestamp_output_mode", True):
        # Export DF to CSV using output_filepath
        df.to_csv(output_filepath, index=False)

    # Export JSON to output filepath.
    with open(input_dict['output_filepath'].replace(".csv",".json"), 'w') as file:
        json.dump(output, file, indent=4)

    return df

def initialize_environment_variables(secret_name: str, project_id: str, logger:Logger):
    """
    Retrieve GCP secret using google cloud secret manager, fallback to loading environment variables locally

    Args:
        secret_name (str): The name of the secret.
        project_id (str): The GCP project ID.
        logger (Logger): Logger instance from python's logging module

    Returns:
        dict: A dictionary containing the secret's key-value pairs.
    """
    try:
        # Attempt to retrieve secrets from GCP
        retrieve_secret(secret_name, project_id)
        logger.info(f"Successfully retrieved secrets from GCP Secret Manager: {secret_name}")

    except (secretmanager.exceptions.NotFound, secretmanager.exceptions.PermissionDenied) as gcp_error:
        logger.warning(f"Failed to retrieve secrets from GCP Secret Manager ({secret_name}): {gcp_error}")
        logger.info("Falling back to local .env file...")

        # Load from .env file as a fallback
        load_dotenv()

    except Exception as e:
        logger.error(f"Unexpected error while retrieving secrets: {e}", exc_info=True)
        raise RuntimeError("Failed to initialize environment variables.")


def safe_blob_download(
        blob: storage.Blob, 
        download_file_prefix: str,
        source_blob_name: str
    ) -> str:
    """
    Safely download a blob file from Google Cloud Storage.

    Args:
        blob: The blob object to download.
        download_file_prefix: The local directory to save the downloaded file.
        source_blob_name: The name of the source blob.  
    Returns:
        local_filename: The file path where the blob was downloaded.
    """
    try:
        # Download the file
        local_filename = os.path.join(download_file_prefix, os.path.basename(blob.name))
        blob.download_to_filename(local_filename)   

        # Handle and parse folder structure
        file_name = blob.name.split("/")[-1]
        file_suffix = file_name.split(".")[-1]
        metadata_path = download_file_prefix + "/" + file_name.replace(file_suffix,"json")

        # Download metadata of the file
        with open(metadata_path, 'w') as file:
            json.dump(blob.metadata, file, indent=4)

    except NotFound:
        logging.error(f"File not found in GCS: {source_blob_name}")
    except PermissionDenied:
        logging.error(f"Permission denied for file: {source_blob_name}")
    except Exception as e:
        logging.error(f"Unexpected error downloading {source_blob_name}: {e}")
    
    return local_filename
