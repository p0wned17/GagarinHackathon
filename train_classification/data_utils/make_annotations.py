import os
import csv


def create_csv_for_dataset(base_folder, csv_filename):
    with open(csv_filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "class_id"])  # Заголовки столбцов

        for class_id in os.listdir(base_folder):
            class_folder = os.path.join(base_folder, class_id)
            for image_filename in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_filename)
                writer.writerow([image_path, class_id])


# Пример использования
create_csv_for_dataset("croped_dataset", "cropped_dataset.csv")
