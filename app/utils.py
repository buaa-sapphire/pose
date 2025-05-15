import os
import uuid

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_uploaded_file(file, directory=UPLOAD_DIR):
    # Ensure filename is safe and unique
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(directory, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path, filename