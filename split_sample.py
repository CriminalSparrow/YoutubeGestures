import os, random, shutil
from pathlib import Path

# Путь к исходным данным
src_root = Path('data/split/train')
# Путь, куда сложим уменьшенный датасет
dst_root = Path('data/sample_train')
dst_root.mkdir(exist_ok=True)

# Сколько изображений оставить на каждый класс
samples_per_class = 5000

random.seed(42)
for cls_dir in src_root.iterdir():
    if not cls_dir.is_dir(): continue
    images = list(cls_dir.glob('*'))
    # Если в классе меньше, чем нужно, берем все
    chosen = images if len(images) <= samples_per_class else random.sample(images, samples_per_class)
    dst_cls = dst_root/cls_dir.name
    dst_cls.mkdir(exist_ok=True, parents=True)
    for img_path in chosen:
        shutil.copy(img_path, dst_cls/img_path.name)

print("Готово! В папке data/sample_train по", samples_per_class, "файлов на класс.")