import pandas as pd # type: ignore
from google.cloud import storage # type: ignore
from io import StringIO
from datetime import datetime
import csv
import time
import random
import json 
from io import BytesIO
import io

def create_folder_if_not_exists(folders, max_retries=5):

    for attempt in range(max_retries):
        try:
            for folder in folders:
                current_date = datetime.now()
                bucket_name = "data-extraction-services"
                destination_blob_path = f'wesel-enterprise/{folder}'
                folder_name = f"{current_date.strftime('%B-%Y')}-results".lower()
                # folder_name = f"2024-11-11-results".lower()
                folder_path = f"{destination_blob_path}/{folder_name}/"
                
                client = storage.Client()
                bucket = client.bucket(bucket_name)

                # Check if the folder exists
                blobs = list(bucket.list_blobs(prefix=folder_path, delimiter='/'))
                if not blobs:
                    # Create an empty object to represent the folder
                    blob = bucket.blob(folder_path)
                    blob.upload_from_string('')
                    print(f"Folder '{folder_path}' created.")
                else:
                    print(f"Folder '{folder_path}' already exists.")
                
                return None
        except Exception as e:
            # print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 3))    # Exponential backoff
            else:
                print("Max retries reached for GCP create folders if not exists.")
                raise
        

def upload_csv_to_bucket(data, folder, subfolder, max_retries=5, csv_name=''):
    
    for attempt in range(max_retries):
        try:
            current_date = datetime.now()
            bucket_name = "data-extraction-services"
            destination_blob_path = f'{folder}'
            folder_path = f"{destination_blob_path}/{subfolder}/"
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Create the folder if it doesn't exist
            blob = bucket.blob(folder_path)
            # blob.upload_from_string('')  # Create an empty object to represent the folder

            csv_buffer = StringIO()
            for key in data:
                if isinstance(data[key], list):
                    data[key] = ', '.join(data[key])
            
            df = pd.DataFrame([data])
            df.to_csv(csv_buffer, index=False)
            if csv_name == '':
                csv_name = f"{data['SKU']}-{data['Manufacturer']}-{data['Product Name']}-{current_date.strftime('%B-%Y')}-results.csv".lower()
            csv = csv_name.replace(" ", "-").replace("(","").replace(")","").replace("/","")
            # Save the CSV file to GCP with the specified name
            blob = bucket.blob(f"{folder_path}{csv}")  # Specify the full path including the file name
            blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
            
            print(f"uploaded result to {csv_name}")

            return None
        
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 3)) # Exponential backoff
            else:
                print("Max retries reached for GCP results Upload.")
                raise
        

def read_csv_from_gcs(folder,max_retries=5):

    for attempt in range(max_retries):
        try:
            bucket_name = "data-extraction-services"
            destination_blob_path = f'wesel-enterprise/{folder}'
            if attempt >= 1:
                destination_blob_path = f'{folder}'
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(destination_blob_path)

            # Download the file as bytes and decode it into a string
            content = blob.download_as_text()

            # Use io.StringIO to read the CSV data
            csv_reader = csv.DictReader(StringIO(content))
            data = [row for row in csv_reader]
            return data
        except Exception as e:
            # print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 3))    # Exponential backoff
            else:
                print("Max retries reached for GCP read csv.")
                raise



def read_xlsx_from_gcs(folder, max_retries=5):
    for attempt in range(max_retries):
        try:
            bucket_name = "data-extraction-services"
            destination_blob_path = f'wesel-enterprise/{folder}'
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_path)

            # Download the file as bytes
            content = blob.download_as_bytes()

            # Read the Excel file into a Pandas ExcelFile object
            excel_data = pd.ExcelFile(BytesIO(content))

            # Convert each sheet into a DataFrame and store in a list
            dataframes = [excel_data.parse(sheet_name) for sheet_name in excel_data.sheet_names]

            return dataframes
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 3))    # Exponential backoff
            else:
                print("Max retries reached for GCP read xlsx.")
                raise




def download_json_from_gcs(source_blob_name):
    """
    Downloads a JSON file from a GCP bucket and returns its content as a dictionary.

    Args:
        bucket_name (str): Name of the GCP bucket.
        source_blob_name (str): Path to the JSON file in the bucket.

    Returns:
        dict: The JSON content as a dictionary.
    """
    bucket_name = "data-extraction-services"

    # Initialize a GCS client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob (file object) from the bucket
    blob = bucket.blob(source_blob_name)

    # Download the blob content as text
    json_content = blob.download_as_text()

    try:
        # Parse the JSON content and return it as a dictionary
        return json.loads(json_content)
    except Exception as e:
        print("Could not download JSON")
        raise



def upload_df_to_gcs(df: pd.DataFrame, bucket_name: str, destination_blob_path: str):
    """
    Uploads a Pandas DataFrame as a CSV file to a Google Cloud Storage (GCS) bucket.

    Args:
        df (pd.DataFrame): The Pandas DataFrame to upload.
        bucket_name (str): Name of the GCS bucket.
        destination_blob_path (str): Path inside the bucket where the file will be stored (e.g., "folder/data.csv").
        project_id (str, optional): Google Cloud Project ID (if not set in environment).

    Returns:
        str: Public URL of the uploaded file (if bucket is public).
    """

    # Convert DataFrame to CSV in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)  # Write DataFrame to the buffer
    csv_buffer.seek(0)  # Reset buffer position

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)

    # Upload CSV file
    blob.upload_from_string(csv_buffer.getvalue(), content_type="text/csv")

    # print(f"File uploaded to gs://{bucket_name}/{destination_blob_path}")

    return f"https://storage.googleapis.com/{bucket_name}/{destination_blob_path}"  # Public URL if bucket is public



def load_csvs_from_gcs(bucket_name: str, folder_prefix: str = "") -> pd.DataFrame:
    """
    Fetches all CSV files from a Google Cloud Storage bucket, concatenates them vertically, and returns a Pandas DataFrame.

    Args:
        bucket_name (str): Name of the GCS bucket.
        folder_prefix (str, optional): Folder path prefix in the bucket (e.g., "data/"). Defaults to "" (root).

    Returns:
        pd.DataFrame: A single concatenated DataFrame containing all CSV data.
    """

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # List all CSV files in the bucket (with optional folder filtering)
    blobs = list(bucket.list_blobs(prefix=folder_prefix))
    csv_blobs = [blob for blob in blobs if blob.name.endswith(".csv")]

    if not csv_blobs:
        print("No CSV files found in the specified bucket/folder.")
        return pd.DataFrame()  # Return an empty DataFrame if no files found

    # List to store DataFrames
    df_list = []

    for blob in csv_blobs:
        # print(f"Downloading {blob.name}...")
        try:
            # Download CSV file into memory
            csv_data = blob.download_as_text()

            # Read CSV into a Pandas DataFrame
            df = pd.read_csv(io.StringIO(csv_data))
        except:
            continue

        df_list.append(df)

    # Concatenate all DataFrames vertically
    final_df = pd.concat(df_list, ignore_index=True)

    print(f"Successfully concatenated {len(csv_blobs)} CSV files.")
    return final_df
