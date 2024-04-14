import flet as ft
import ast

import requests


url = "http://46.191.235.91:3333/predict/"


def get_result(url, file_path):
    print(file_path)
    with open(file_path, "rb") as file:
        files = {"file": (file_path, file, "image/jpeg")}
        response = requests.post(url, files=files)

    return response.text


def main(page: ft.Page):
    page.bgcolor = ft.colors.WHITE
    page.update()
    img = ft.Image()

    def display_im(path):
        img = ft.Image(
            src=path,
            width=400,
            height=400,
            fit=ft.ImageFit.CONTAIN,
        )
        images = ft.Row(expand=1, wrap=False, scroll="always")
        page.controls.append(img)
        page.update()

    def get_model_result(path):
        res = get_result(url, path)
        print("res")
        print(res)
        res = ast.literal_eval(res)
        print(res)
        return {'Класс изображения': res['type'], 'Страница': res['page_number'], 'Confidence': res['confidence'], 'Серия документа': res['series'], 'Номер': res['number']}

    def display_results(res):
        for key, value in res.items():
            text = key + ": " + str(value)
            page.controls.append(ft.Text(text, color=ft.colors.BLACK))
        page.update()


    def pick_files_result(e: ft.FilePickerResultEvent):
        print(selected_file_text.value)
        print(len(page.controls))
        if len(page.controls) > 1:
            for i in range (len(page.controls) - 1):
                page.controls.pop()
            page.update()
        selected_file_text.value = (
             ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
         )
        selected_file_text.update()
        selected_file_im = e.files[0].path
        print(selected_file_im)
        print(selected_file_im)
        display_im(selected_file_im)
        results = get_model_result(selected_file_im)
        display_results(results)



    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    selected_file_text = ft.Text()
    selected_file_im = ft.Image()

    page.overlay.append(pick_files_dialog)

    page.add(
        ft.Row(
            [
                ft.ElevatedButton(
                    "Загрузить изображение",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False,
                        allowed_extensions=['png', 'jpg', 'jpeg'],
                    ),
                ),
                selected_file_text,
            ]
        )
    )


ft.app(target=main)
