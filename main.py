"""Интерактивный помощник по фильмам и сериалам на базе DeepSeek API."""

import os
import time
from dataclasses import dataclass
from threading import Thread
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError


MODEL_NAME = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"
BASE_SYSTEM_PROMPT = "Ты - эксперт по фильмам и сериалам. Отвечай на русском языке."
DEFAULT_FINISH_INSTRUCTION = "Заверши свой ответ, когда достигнешь основного вывода"

# Простая проверка: достаточно одного слова из списка в тексте вопроса.
TOPIC_KEYWORDS = {
    "фильм", "фильмы", "кино", "кинематограф", "сериал", "сериалы",
    "сезон", "серия", "эпизод", "актер", "актёр", "актриса", "режиссер",
    "режиссёр", "сценарий", "сценарист", "жанр", "премьера", "трейлер",
    "каст", "персонаж", "саундтрек", "мультфильм", "аниме", "оскара",
    "оскар", "netflix", "hbo", "imdb",
}


@dataclass
class Limits:
    """Настройки трёх ограничений запроса."""

    json_enabled: bool = False
    max_tokens_enabled: bool = False
    max_tokens: int = 500
    finish_enabled: bool = False
    finish_instruction: str = DEFAULT_FINISH_INSTRUCTION


def enabled_label(value: bool) -> str:
    """Возвращает русскую метку состояния переключателя."""
    return "ВКЛ" if value else "ВЫКЛ"


def clear_console() -> None:
    """Очищает окно консоли в Windows, macOS и Linux."""
    os.system("cls" if os.name == "nt" else "clear")


def is_movie_or_series_question(question: str) -> bool:
    """Проверяет наличие тематических ключевых слов в вопросе."""
    lowered_question = question.lower()
    return any(keyword in lowered_question for keyword in TOPIC_KEYWORDS)


def is_list_request(question: str) -> bool:
    """Определяет, просит ли пользователь подборку из нескольких произведений."""
    lowered_question = question.lower()
    list_markers = (
        "список", "подборк", "топ", "посовет", "порекомендуй",
        "какие фильмы", "какие сериалы", "назови", "перечисли",
        "несколько фильмов", "несколько сериалов",
    )
    return any(marker in lowered_question for marker in list_markers)


def read_positive_integer(prompt: str, default: int) -> int:
    """Запрашивает положительное целое; пустой ввод сохраняет значение."""
    while True:
        raw_value = input(prompt).strip()
        if not raw_value:
            return default
        try:
            value = int(raw_value)
            if value > 0:
                return value
        except ValueError:
            pass
        print("Ошибка: введите положительное целое число.")


def manage_limits(limits: Limits) -> None:
    """Показывает подменю и изменяет настройки ограничений."""
    while True:
        clear_console()
        print("\n=== Управление ограничениями ===")
        print(f"1. Формат ответа (JSON) - [{enabled_label(limits.json_enabled)}]")
        print(
            "2. Максимальная длина - "
            f"[{enabled_label(limits.max_tokens_enabled)}] "
            f"(текущее значение: {limits.max_tokens} токенов)"
        )
        print(f"3. Условие завершения - [{enabled_label(limits.finish_enabled)}]")
        print("4. Назад в главное меню")
        choice = input("Выберите действие: ").strip()

        if choice == "1":
            limits.json_enabled = not limits.json_enabled
            print(f"Формат JSON: {enabled_label(limits.json_enabled)}.")
        elif choice == "2":
            limits.max_tokens = read_positive_integer(
                f"Максимальная длина в токенах (Enter — оставить {limits.max_tokens}): ",
                limits.max_tokens,
            )
            limits.max_tokens_enabled = not limits.max_tokens_enabled
            print(f"Максимальная длина: {enabled_label(limits.max_tokens_enabled)}.")
        elif choice == "3":
            new_instruction = input(
                "Условие завершения (Enter — оставить текущее): "
            ).strip()
            if new_instruction:
                limits.finish_instruction = new_instruction
            limits.finish_enabled = not limits.finish_enabled
            print(f"Условие завершения: {enabled_label(limits.finish_enabled)}.")
        elif choice == "4":
            return
        else:
            print("Ошибка: выберите пункт от 1 до 4.")


def build_request(question: str, limits: Limits | None = None) -> dict[str, Any]:
    """Собирает запрос с ограничениями или без них, если limits равен None."""
    system_prompt = BASE_SYSTEM_PROMPT
    if limits and limits.finish_enabled:
        system_prompt += f"\n{limits.finish_instruction}"

    # DeepSeek требует слова «json» в сообщениях при использовании JSON Output.
    # Эта инструкция избавляет пользователя от необходимости добавлять его в вопрос.
    if limits and limits.json_enabled:
        if is_list_request(question):
            system_prompt += (
                "\nВерни ответ в формате json: только валидный JSON-объект без "
                "Markdown и пояснений. Пользователь просит список, поэтому верни "
                "структуру: {\"items\": [{\"название\": \"...\", \"тип\": "
                "\"фильм или сериал\", \"год\": 2024, \"рейтинг\": \"...\", "
                "\"жанр\": \"...\", \"описание\": \"...\"}]}. Каждый элемент "
                "списка должен быть отдельным объектом. Если данные о годе или "
                "рейтинге неизвестны, укажи null."
            )
        else:
            system_prompt += (
                "\nВерни ответ в формате json: только валидный JSON-объект без "
                "Markdown и пояснений. Используй структуру: "
                '{"answer": "текст ответа на русском языке"}.'
            )

    request: dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        # Режим рассуждений включён у модели по умолчанию. Отключаем его, чтобы
        # в поле content всегда возвращался итоговый ответ, а не только рассуждения.
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    if limits and limits.json_enabled:
        request["response_format"] = {"type": "json_object"}
    if limits and limits.max_tokens_enabled:
        request["max_tokens"] = limits.max_tokens
    return request


def get_response_text(response: Any) -> str | None:
    """Безопасно получает итоговый текст из ответа Chat Completions."""
    if not response.choices:
        return None

    content = response.choices[0].message.content
    if isinstance(content, str) and content.strip():
        return content
    return None


def send_request(client: OpenAI, question: str, limits: Limits | None) -> str:
    """Отправляет один запрос и возвращает текст ответа или русское сообщение об ошибке."""
    try:
        response = client.chat.completions.create(**build_request(question, limits))
        answer = get_response_text(response)
        return answer or "DeepSeek не сформировал итоговый текст ответа."
    except APIConnectionError:
        return "Ошибка сети: не удалось подключиться к DeepSeek API."
    except APIStatusError as error:
        return f"Ошибка API DeepSeek (код {error.status_code}): {error.message}"
    except OpenAIError as error:
        return f"Ошибка при обращении к DeepSeek API: {error}"


def send_request_with_loading(
    client: OpenAI,
    question: str,
    limits: Limits | None,
    loading_message: str,
) -> str:
    """Показывает анимацию загрузки, пока запрос DeepSeek выполняется в потоке."""
    result: list[str] = []

    def request_worker() -> None:
        try:
            result.append(send_request(client, question, limits))
        except Exception as error:  # Защита от непредвиденной ошибки внутри потока.
            result.append(f"Непредвиденная ошибка при обработке запроса: {error}")

    worker = Thread(target=request_worker)
    worker.start()

    frames = ("|", "/", "-", "\\")
    frame_index = 0
    while worker.is_alive():
        print(
            f"\r{loading_message} {frames[frame_index % len(frames)]}",
            end="",
            flush=True,
        )
        frame_index += 1
        time.sleep(0.15)

    worker.join()
    # Стираем строку анимации перед печатью ответа.
    print("\r" + " " * 80 + "\r", end="", flush=True)
    return result[0]


def ask_question(client: OpenAI, limits: Limits) -> None:
    """Принимает вопрос, проверяет тематику и выводит ответ API."""
    question = input("Введите вопрос о фильмах или сериалах: ").strip()
    if not question:
        print("Ошибка: вопрос не должен быть пустым.")
        return
    if not is_movie_or_series_question(question):
        print(
            "Ошибка: Я специализируюсь только на вопросах о фильмах и сериалах. "
            "Пожалуйста, задайте вопрос по теме."
        )
        return

    # Первый запрос учитывает текущее состояние всех трёх ограничений.
    answer_with_limits = send_request_with_loading(
        client,
        question,
        limits,
        "Получаю ответ с выбранными ограничениями...",
    )

    # Второй запрос получает тот же вопрос, но только с базовым системным промптом.
    answer_without_limits = send_request_with_loading(
        client,
        question,
        None,
        "Получаю ответ без ограничений...",
    )

    print("\n=== Ответ с выбранными ограничениями ===")
    print(answer_with_limits)
    print("\n=== Ответ без ограничений ===")
    print(answer_without_limits)
    input("\nНажмите Enter, чтобы вернуться в главное меню...")


def main() -> None:
    """Запускает главное меню приложения."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Ошибка: не найдена переменная окружения DEEPSEEK_API_KEY.")
        print("Укажите API-ключ DeepSeek в этой переменной и запустите программу снова.")
        return

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    limits = Limits()

    while True:
        clear_console()
        print('Добро пожаловать в "КиноЭксперт"! Я помогаю с вопросами о фильмах и сериалах.')
        print("\n=== Главное меню ===")
        print("1. Задать вопрос")
        print("2. Управление ограничениями")
        print("3. Выход")
        choice = input("Выберите действие: ").strip()

        if choice == "1":
            ask_question(client, limits)
        elif choice == "2":
            manage_limits(limits)
        elif choice == "3":
            print("Спасибо за использование КиноЭксперта. До свидания!")
            return
        else:
            print("Ошибка: выберите пункт от 1 до 3.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nРабота программы завершена пользователем.")
