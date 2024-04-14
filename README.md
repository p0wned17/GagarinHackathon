# GagarinHack - Решение команды GigaFlex

## Обучение модели

### Обучение для обоих классификаторов схожи, поэтому выбираем нужный

Запуск обучения:

```bash
python main.py
```

## Конвертация модели в torchscript

```bash
python train/convert_to_torchscript.py 
```

## Инференс
Пример инференса лежит в ./inference/inference.py, inference_on_dir.py


Для запуска модели в продакшн в виде приложения обратитесь к инструкции [запуск модели в продакшн](flet_app/README.md)
