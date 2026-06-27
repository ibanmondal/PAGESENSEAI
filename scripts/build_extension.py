import os
import zipfile
import shutil
from pathlib import Path

def create_zip():
    project_root = Path(__file__).parent.parent
    extension_dir = project_root / 'extension'
    website_downloads_dir = project_root / 'website' / 'downloads'
    
    # Ensure the downloads directory exists
    website_downloads_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = website_downloads_dir / 'pagesense-extension.zip'
    
    print(f"Packaging {extension_dir} into {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(extension_dir):
            # Exclude unwanted directories
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
            
            for file in files:
                if file.endswith('.DS_Store'):
                    continue
                    
                file_path = Path(root) / file
                # The arcname is the path inside the zip file
                arcname = file_path.relative_to(extension_dir)
                zipf.write(file_path, arcname)
                
    print(f"Successfully created {zip_path}")

if __name__ == '__main__':
    create_zip()
