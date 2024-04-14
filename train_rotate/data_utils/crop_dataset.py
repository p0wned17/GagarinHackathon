import cv2
import pandas as pd
import os


def crop_and_save_images(csv_path, output_base_folder, crop_size=180):
    df = pd.read_csv(csv_path)
    grouped = df.groupby("image_path")

    for image_path, group in grouped:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            continue

        # Проверка наличия других классов кроме 0
        contains_other_classes = any(group["class_id"] != 0)

        for index, row in group.iterrows():
            x, y, class_id = int(row["x"]), int(row["y"]), int(row["class_id"])

            # Пропуск класса 0, если есть другие классы
            if class_id == 0 and contains_other_classes:
                continue

            # Определение координат кропа
            x1 = max(x - crop_size // 2, 0)
            y1 = max(y - crop_size // 2, 0)
            x2 = min(x + crop_size // 2, image.shape[1])
            y2 = min(y + crop_size // 2, image.shape[0])

            crop = image[y1:y2, x1:x2]

            # Создание папки для класса, если она не существует
            class_folder = os.path.join(output_base_folder, str(class_id))
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Сохранение кропа
            crop_filename = f"{os.path.basename(image_path).split('.')[0]}_{index}.png"
            try:
                cv2.imwrite(os.path.join(class_folder, crop_filename), crop)
            except Exception:
                print(os.path.join(class_folder, crop_filename))


# Пример использования
crop_and_save_images("../output.csv", "croped_dataset")
