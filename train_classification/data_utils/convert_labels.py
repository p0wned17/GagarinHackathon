import csv
from pathlib import Path

DATASET_PATH = Path("/home/cv_user/projects/hackathon/pipe_dataset/FRAMES")


def convert_to_csv(input_path, output_csv_path):
    with open(input_path, "r", encoding="windows-1251") as file, open(
        output_csv_path, "w", newline=""
    ) as out_file:
        csv_writer = csv.writer(out_file)
        csv_writer.writerow(["image_path", "x", "y", "class_id"])  # Заголовки столбцов

        for line in file:
            line = line.strip()
            if line.endswith(".frame"):  # Проверка на строку пути
                # Преобразование пути и проверка его существования
                image_path = line.replace("\\", "/").replace(".frame", ".bmp")

                image_path = DATASET_PATH / image_path
                if not image_path.exists():
                    print(f"Путь не существует: {image_path}")
                    continue
            else:
                if image_path.exists():
                    print(image_path, image_path.exists())
                    # Извлечение координат и класса
                    x, y, class_id = line.split(", ")
                    csv_writer.writerow([image_path, x, y, class_id])


# Пример использования
convert_to_csv(
    "/home/cv_user/projects/hackathon/pipe_dataset/metadata/set.cfg", "output.csv"
)
