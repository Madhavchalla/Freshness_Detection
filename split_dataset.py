import os
import shutil
import random
SOURCE_DIR = 'dataset'
TARGET_DIR = 'final_dataset'


SPLIT_RATIOS = (0.7, 0.15, 0.15)  # 70% train, 15% validation, 15% test


categories = ['Fresh', 'Rotten']

for category in categories:
    category_path = os.path.join(SOURCE_DIR, category)
    files = os.listdir(category_path)
    random.shuffle(files)

    total = len(files)
    train_end = int(SPLIT_RATIOS[0] * total)
    val_end = train_end + int(SPLIT_RATIOS[1] * total)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for split_name, split_files in zip(['train', 'validation', 'test'], [train_files, val_files, test_files]):
        dest_dir = os.path.join(TARGET_DIR, split_name, category)
        os.makedirs(dest_dir, exist_ok=True)

        for file in split_files:
            src_file = os.path.join(category_path, file)
            dst_file = os.path.join(dest_dir, file)
            shutil.copy2(src_file, dst_file)

print("✅ Dataset split completed!")
