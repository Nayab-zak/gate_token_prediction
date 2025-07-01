import os
import zipfile
from datetime import datetime

# Define what to include
INCLUDE_PATHS = [
    'agents',
    'config',
    'scripts',
    'utils',
    'models',
    'outputs/baseline_metrics.json',
    'data/preprocessed/preprocessed_features.csv',
    'logs',
    'reports',
    'README.md',
    'requirements.txt',
    'Makefile',
    'docs',
]

OUTPUT_DIR = 'dist'
OUTPUT_ZIP = os.path.join(OUTPUT_DIR, 'token_count_project.zip')


def zipdir(path, ziph, arc_prefix=""):
    for root, dirs, files in os.walk(path):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, start=os.getcwd())
            arcname = os.path.join(arc_prefix, os.path.relpath(abs_path, path))
            ziph.write(abs_path, arcname)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in INCLUDE_PATHS:
            if os.path.isdir(item):
                zipdir(item, zipf, arc_prefix=item)
            elif os.path.isfile(item):
                zipf.write(item, item)
    print(f"Packaged artifacts to {OUTPUT_ZIP}")

if __name__ == '__main__':
    main()
