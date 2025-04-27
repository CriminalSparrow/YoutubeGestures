import os
import shutil
import random
from tqdm import tqdm

# Путь до исходного датасета
DATASET_PATH = os.path.join('C:/Users/socol/HaGRIDv2_dataset_512')

# Жесты, которые ты хочешь оставить
TARGET_GESTURES = ['like', 'dislike', 'palm', 'thumb_index', 'gun', 'timeout', 'stop']

# Папка для нового датасета
OUTPUT_PATH = 'data/split'

# Процент файлов для валидации и теста
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Создание структуры папок
for split in ['train', 'val', 'test']:
    for gesture in TARGET_GESTURES + ['no gesture']:
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

# Проход по папкам
all_gestures = os.listdir(DATASET_PATH)

for gesture in tqdm(all_gestures, desc="Processing gestures"):
    gesture_path = os.path.join(DATASET_PATH, gesture)
    if not os.path.isdir(gesture_path):
        continue

    images = [os.path.join(gesture_path, img) for img in os.listdir(gesture_path) if img.endswith('.jpg') or img.endswith('.png')]

    if gesture in TARGET_GESTURES:
        # Оставляем все изображения
        split_and_copy(images, OUTPUT_PATH, gesture)
    else:
        # Берем 6% случайных изображений и кидаем в no gesture
        n_select = max(1, int(len(images) * 0.06))
        selected_images = random.sample(images, n_select)
        split_and_copy(selected_images, OUTPUT_PATH, 'no gesture')

print("✅ Разделение завершено!")