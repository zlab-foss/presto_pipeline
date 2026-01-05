from pathlib import Path
import shutil

numbers = [
    19, 22, 24, 25, 38, 39, 43, 54, 57, 73, 74, 82,
    101, 102, 103, 106, 120, 123, 124, 145, 159,
    162, 173, 181, 200, 201, 218
]

src_dir = Path("/home/ubuntu/clony/presto_pipeline/data/dataset-lulc/rgbnir/landsat")
dst_dir = Path("/home/ubuntu/clony/presto_pipeline/data/dataset-lulc/rgbnir/val")

dst_dir.mkdir(parents=True, exist_ok=True)

# convert numbers to strings once
num_strs = [str(n) for n in numbers]

moved = 0
for file in src_dir.iterdir():
    if not file.is_file():
        continue

    name = file.name
    if any(n in name for n in num_strs):
        shutil.move(str(file), dst_dir / file.name)
        moved += 1

print(f"Moved {moved} files.")
