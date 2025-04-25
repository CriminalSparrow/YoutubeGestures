import os
import shutil
import random
from pathlib import Path

# Параметры
source_dir = Path('data/photos')
target_dir = Path('data/split')
split_ratio = (0.7, 0.15, 0.15)  # train, val, test

# Фиксация random
random.seed(42)

# Создание целевых директорий
for split in ['train', 'val', 'test']:
    for class_dir in source_dir.iterdir():
        (target_dir / split / class_dir.name).mkdir(parents=True, exist_ok=True)

# Копирование файлов с разбиением
for class_dir in source_dir.iterdir():
    images = list(class_dir.glob('*'))
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    split_files = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, files in split_files.items():
        print(f"Started working with{split}")
        for img in files:
            shutil.copy(img, target_dir / split / class_dir.name / img.name)
        print(f"Finished working with{split}")

print("Готово! Данные разбиты и скопированы в 'data/split'.")