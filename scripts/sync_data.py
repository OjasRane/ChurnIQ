"""Copy root data into the self-contained Render service. Run after dataset changes."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
(root/'backend/data').mkdir(exist_ok=True)
for file in (root/'data').glob('*.csv'):
    shutil.copy2(file,root/'backend/data'/file.name)
    print(file.name)
