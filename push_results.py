"""
HuggingFace upload wrapper for Rough-EVT results.
Uses HfApi.upload_file (same as Hilbert-PLV engine).
"""

import os
from huggingface_hub import HfApi
from config import RESULTS_REPO


def upload_results(file1_path: str, file2_path: str, hf_token: str = None):
    """
    Upload the two JSON result files to HuggingFace.
    """
    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is required")
    
    api = HfApi()
    
    # Upload both files
    for file_path in [file1_path, file2_path]:
        if not os.path.exists(file_path):
            print(f"⚠️  Warning: {file_path} not found, skipping")
            continue
        
        # Extract filename from path
        filename = os.path.basename(file_path)
        
        print(f"   Uploading {filename}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=RESULTS_REPO,
            token=token,
            repo_type="dataset"
        )
    
    print("✅ Upload complete!")
