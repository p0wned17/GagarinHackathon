import telebot
import requests

TOKEN = "6851734849:AAEEdAIzXhtCmcdt148TqSgQpSdDlsBtdJs"
url = "http://46.191.235.91:3333/predict/"

bot = telebot.TeleBot(TOKEN)

id_to_classes_map = {
    "personal_passport": "Паспорт",
    "vehicle_passport": "ПТС",
    "driver_license": "Водительское удостоверение",
    "vehicle_certificate": "СТС, 1 страница",
}

@bot.message_handler(commands=["start"])
def send_welcome(message):
    bot.reply_to(message, "Привет! Отправь мне изображение, и я его обработаю.")

@bot.message_handler(content_types=["photo"])
def handle_image(message):
    
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    files = {"file": ("photo.jpg", downloaded_file, "image/jpeg")}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        api_response = response.json()

        bot.reply_to(
            message,
            f"<b>Тип документа:</b>{id_to_classes_map[api_response['type']]}\n<b>Номер страницы:</b>{api_response['page_number']}\n<b>Уверенность в тип документа:</b> {api_response['confidence']}\n<b>Серия: {api_response['series']}</b>\n<b>Номер: {api_response['number']}</b>",
            parse_mode="HTML",
        )
    else:
        bot.reply_to(message, "Произошла ошибка при обработке изображения. Попробуйте еще раз позже.")

bot.polling()