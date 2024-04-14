import csv
from pathlib import Path
from typing import Union

import typer
from tqdm import tqdm


class CSVAnnotations:
    def __init__(self, file_path: str = "annotations.csv"):
        self.fieldnames = ["image_path", "class_id"]
        self.file_path = Path(file_path)
        self.file_path.touch(exist_ok=True)
        with open(self.file_path, "w") as csv_file:
            self.writer = csv.DictWriter(
                csv_file, fieldnames=self.fieldnames, delimiter=";")
            self.writer.writeheader()

    def write(self, image_path: str, class_id: Union[int, str]):
        with open(self.file_path, "a") as csv_file:
            self.writer = csv.DictWriter(
                csv_file, fieldnames=self.fieldnames, delimiter=";")
            self.writer.writerow({"image_path": image_path, "class_id": class_id})


def generate_annotations(input_path: str, output_path: str):
    annotator = CSVAnnotations(file_path=output_path)
    class_counter = 0  # Начальный идентификатор класса
    
    for class_dir in tqdm(list(Path(input_path).iterdir())):
        if not class_dir.is_dir():
            continue  # Пропускаем, если это не директория
        
        image_paths = list(class_dir.rglob("*"))
        if 'PTS' in str(class_dir):
            # Для PTS присваиваем всем изображениям один и тот же класс
            class_id_pts = class_counter
            for image_path in image_paths:
                # relation_path = str(image_path.relative_to(input_path))  # Относительный путь файла
                annotator.write(image_path=image_path, class_id=class_id_pts)
            class_counter += 1  # Увеличиваем счётчик классов только на один для PTS
        else:
            # Для других папок используем два класса
            class_id_1 = class_counter
            class_id_2 = class_counter + 1
            for image_path in image_paths:
                file_name = image_path.name
                if file_name.endswith("-1.png"):
                    specific_class_id = class_id_1
                elif file_name.endswith("-2.png"):
                    specific_class_id = class_id_2
                else:
                    specific_class_id = class_id_1  # По умолчанию первая страница
                
                # relation_path = str(image_path.relative_to(input_path))
                annotator.write(image_path=image_path, class_id=specific_class_id)
            class_counter += 2  # Для остальных папок увеличиваем на два

    typer.secho(f"Всего классов: {class_counter}", fg=typer.colors.BRIGHT_YELLOW)
    typer.secho(f"Сохранено в {output_path}")





def main(
    input_path: Path = typer.Argument(
        "./dataset/", help="Путь до папки с датасетом"
    ),
    output_path: Path = typer.Argument(
        "./dataset/annotations.csv",
        help="Путь до выходной папки с датасетом",
    ),
):
    generate_annotations(input_path, output_path)


if __name__ == "__main__":
    try:
        typer.run(main)
    except SystemExit:
        typer.secho("Программа завершена", fg=typer.colors.BRIGHT_GREEN)
