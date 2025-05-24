# utils/pdf_ocr_utils.py

import os
import time
import shutil
import tempfile
from typing import List, Tuple
from pdf2image import convert_from_path
import pytesseract 

def pdf_to_images(pdf_path: str, output_id: str = "default_id", dpi: int = 300) -> Tuple[List[str], str]:
    """
    Converts each page of a PDF to images using pdf2image.
    Returns a list of image file paths created in a time-stamped directory.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Create a time-stamped subfolder in ./intermediate
    output_folder = f"./intermediate/{output_id}"
    os.makedirs(output_folder, exist_ok=True)

    # Convert pages to images in a temp directory first
    temp_dir = tempfile.mkdtemp(prefix="pdf_images_")
    pages = convert_from_path(pdf_path, dpi=dpi, fmt="png", output_folder=temp_dir)
    
    image_paths = []
    for i, page in enumerate(pages):
        image_filename = f"page_{i+1}.png"
        final_path = os.path.join(output_folder, image_filename)

        # Save the image into the final folder
        page.save(final_path, "PNG")
        image_paths.append(final_path)

    # Cleanup the temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    return image_paths, output_folder


def ocr_images(image_paths: List[str], output_id: str) -> str:
    """
    Applies OCR to each image using pytesseract and returns concatenated text.
    Saves the combined OCR text to 'ocr_combined.txt' in the same folder.
    Args:
        image_paths (List[str]): A list of file paths pointing to the images.
        output_id (str): A unique identifier (e.g., timestamp, job ID) used to create an output directory for storing the OCR results.
    Returns:
        str: The combined OCR-extracted text from all images.
    Saves the combined OCR text to 'ocr_combined.txt' in the same folder.
    """
    all_text = []
    for path in image_paths:
        text = pytesseract.image_to_string(path, lang="eng")
        all_text.append(text)

    combined_text = "\n".join(all_text)

    # Create a time-stamped subfolder in ./intermediate
    output_folder = f"./intermediate/{output_id}"
    os.makedirs(output_folder, exist_ok=True)
    
    ocr_file_path = os.path.join(output_folder, "ocr_combined.txt")
    with open(ocr_file_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    return combined_text


def pdf_to_text(pdf_path: str, output_id: str = "default_id") -> Tuple[str, str]:
    """
    End-to-end function:
      - Convert PDF to images
      - OCR each page
      - Return combined text
      - Also returns the folder path with images + OCR text
    """
    image_paths, output_folder = pdf_to_images(pdf_path, output_id=output_id)
    combined_text = ocr_images(image_paths, output_folder)
    return combined_text, output_folder

def explore_image_files(image_folder_path: str) -> List[str]:
    """
  Explore all image files in a given directory and return their paths.
  Args:
      image_folder_path (str): The path to the folder containing image files.
  Returns:
      image_files (List[str]): A list of file paths for all image files found in the directory.

  Raises:
      FileNotFoundError: If the path does not exist.
      NotADirectoryError: If the path is not a directory.
      OSError: If an error occurs reading the directory contents.
    """
    if not os.path.exists(image_folder_path):
        raise FileNotFoundError(f"The path '{image_folder_path}' does not exist.")

    if not os.path.isdir(image_folder_path):
        raise NotADirectoryError(f"The path '{image_folder_path}' is not a directory.")

    try:
        all_files = os.listdir(image_folder_path)
    except OSError as e:
        raise OSError(f"Error reading directory '{image_folder_path}': {e}") from e

    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    image_files = [
        f for f in sorted(all_files)
        if f.lower().endswith(image_extensions)
    ]

    return image_files