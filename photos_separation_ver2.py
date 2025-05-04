import os
import shutil
import random
from tqdm import tqdm

# Путь до исходного датасета
DATASET_PATH = 'C:/Users/socol/HaGRIDv2_dataset_512'

# Папка для нового датасета
OUTPUT_PATH = 'data/split'

# Процент файлов для валидации и теста
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Список всех жестов (читаем из папок)
all_gestures = [g for g in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, g))]

# Создание структуры папок
for split in ['train', 'val', 'test']:
    for gesture in all_gestures:
        os.makedirs(os.path.join(OUTPUT_PATH, split, gesture), exist_ok=True)

# Функция для разбиения на train/val/test
def split_and_copy(file_list, target_dir, gesture):
    random.shuffle(file_list)
    n_total = len(file_list)
    n_val = int(n_total * VAL_RATIO)
    n_test = int(n_total * TEST_RATIO)
    
    val_files = file_list[:n_val]
    test_files = file_list[n_val:n_val+n_test]
    train_files = file_list[n_val+n_test:]

    for f in train_files:
        shutil.copy(f, os.path.join(target_dir, 'train', gesture))
    for f in val_files:
        shutil.copy(f, os.path.join(target_dir, 'val', gesture))
    for f in test_files:
        shutil.copy(f, os.path.join(target_dir, 'test', gesture))

# Проход по всем жестам
for gesture in tqdm(all_gestures, desc="Processing gestures"):
    gesture_path = os.path.join(DATASET_PATH, gesture)
    images = [os.path.join(gesture_path, img) for img in os.listdir(gesture_path)
              if img.lower().endswith(('.jpg', '.png'))]

    split_and_copy(images, OUTPUT_PATH, gesture)

print("✅ Разделение завершено!")