import torch
import cv2
import numpy as np


OVERLAP = 0.2
INPUT_SIZE = 160


IMAGE_PATH = "/home/cv_user/projects/hackathon/pipe_detect/0_183.bmp"


def hex_to_bgr(hex_color):
    """Преобразует шестнадцатеричный цвет в формат BGR."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 8:  # Если присутствует альфа-канал
        hex_color = hex_color[:-2]  # Удаляем альфа-канал
    b, g, r = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
    return (b, g, r)


classes = {
    0: ("66FF66AA", "не дефект"),
    1: ("8833FFFF", "потертость"),
    2: ("0000FFFF", "черная точка"),
    3: ("FF8800FF", "плена"),
    4: ("FF0000FF", "маркер"),
    5: ("AE7C10FF", "грязь"),
    6: ("FFFFFFFF", "накол"),
    7: ("FFDDAAFF", "н.д. накол"),
    8: ("FF00FFFF", "микровыступ"),
    9: ("880088FF", "н.д. микровыступ"),
    10: ("FFAA88FF", "вмятина"),
    11: ("3366FFFF", "мех.повреждение"),
    12: ("009900FF", "риска"),
    13: ("CC9900FF", "царапина с волчком"),
}
for class_id, (hex_color, name) in classes.items():
    classes[class_id] = (hex_to_bgr(hex_color), name)

model = torch.jit.load("bestv2.torchscript", map_location="cuda:0")
model = model.half().eval()
model.to(memory_format=torch.channels_last)


def read_image(image_path: str):
    image = cv2.imread(image_path)

    return image


def preprocess(image: np.ndarray):
    # Перевод изображения в формат RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Преобразование формата и типа данных
    image = np.ascontiguousarray(image).transpose((2, 0, 1))
    image = torch.from_numpy(image).float()

    # Масштабирование значений пикселей
    image = image / 255.0

    # Нормализация с использованием значений ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized_image = (image - mean) / std

    return normalized_image.unsqueeze(0)


def sliding_inference(image):
    # image = image.unsqueeze(0)
    image = image.half().cuda()
    stride = int(INPUT_SIZE * (1 - OVERLAP))

    overlap = INPUT_SIZE - stride

    image_height, image_width = image.shape[-2:]
    original_image_height, original_image_width = image.shape[-2:]

    num_horizontal_windows = (
        (image_width) // INPUT_SIZE + 1 if image_width != INPUT_SIZE else 1
    )
    num_vertical_windows = (
        (image_height) // INPUT_SIZE + 1 if image_height != INPUT_SIZE else 1
    )

    w_free_len = image_width - (num_horizontal_windows - 1) * INPUT_SIZE
    h_free_len = image_height - (num_vertical_windows - 1) * INPUT_SIZE

    if num_vertical_windows == 1:
        optim_overlap_height = 0
    else:
        optim_overlap_height = (INPUT_SIZE - h_free_len) / (num_vertical_windows - 1)

    if num_horizontal_windows == 1:
        optim_overlap_width = 0
    else:
        optim_overlap_width = (INPUT_SIZE - w_free_len) / (num_horizontal_windows - 1)

    stride_h = int(INPUT_SIZE - optim_overlap_height)
    stride_w = int(INPUT_SIZE - optim_overlap_width)

    new_height = num_vertical_windows * stride + overlap
    new_width = num_horizontal_windows * stride + overlap

    pad_height = new_height - image_height
    pad_width = new_width - image_width

    pad_height = new_height - image_height if image_height < INPUT_SIZE else 0
    pad_width = new_width - image_width if image_width < INPUT_SIZE else 0

    image = torch.nn.functional.pad(
        image, (0, pad_width, 0, pad_height), mode="constant", value=114 / 255
    )
    image_height, image_width = image.shape[-2:]
    patches = []

    windows_coords = []
    # num_windows = num_horizontal_windows*num_vertical_windows

    for y in range(0, image.shape[2] - INPUT_SIZE + 1, stride_h):
        for x in range(0, image.shape[3] - INPUT_SIZE + 1, stride_w):
            patch = image[:, :, y : y + INPUT_SIZE, x : x + INPUT_SIZE]
            windows_coords.append((x, y))
            patches.append(patch)

    patches = torch.cat(patches, dim=0)

    results = []
    with torch.inference_mode():
        outputs = model(patches.to(memory_format=torch.channels_last))

        scores = torch.softmax(outputs, dim=1).cpu()
        predicts = torch.argmax(scores, dim=1).cpu()

        for i, class_id in enumerate(predicts):
            confidence = scores[i][class_id]
            x, y = windows_coords[i]
            x2, y2 = x + INPUT_SIZE, y + INPUT_SIZE
            results.append(
                {
                    "class_id": class_id.item(),
                    "confidence": confidence.item(),
                    "x1": x,
                    "y1": y,
                    "x2": x2,
                    "y2": y2,
                }
            )

    return results


def scaling_predictions(predictions, scale):
    for predict in predictions:
        predict["x1"] = int(predict["x1"] * scale)
        predict["y1"] = int(predict["y1"] * scale)
        predict["x2"] = int(predict["x2"] * scale)
        predict["y2"] = int(predict["y2"] * scale)

    return predictions


def main():
    original_image = read_image(IMAGE_PATH)
    image = preprocess(original_image)

    import time

    print(type(image))
    t1 = time.time()
    predictions = sliding_inference(image)
    print(time.time() - t1)

    for prediction in predictions:
        x1 = prediction["x1"]
        y1 = prediction["y1"]
        x2 = prediction["x2"]
        y2 = prediction["y2"]
        class_id = prediction["class_id"]
        if class_id == 0:
            continue
        confidence = prediction["confidence"]
        color, class_name = classes.get(
            class_id, ((255, 255, 255), "Неизвестный класс")
        )

        center_x = int(x1 + (x2 - x1) / 2)
        center_y = int(y1 + (y2 - y1) / 2)
        cv2.putText(
            original_image,
            f"{class_name}, {confidence:.2f}",
            (center_x, center_y - 10),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            color,
            1,
        )
        cv2.circle(
            original_image,
            (int(center_x), int(center_y)),
            radius=7,
            color=color,
            thickness=-1,
        )
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 3)

    cv2.imwrite("visualize.jpg", original_image)


if __name__ == "__main__":
    main()
