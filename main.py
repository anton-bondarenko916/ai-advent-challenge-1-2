"""Интерактивное сравнение ответов DeepSeek при разных температурах."""

import os
import time
from dataclasses import dataclass

from colorama import Fore, init
from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from tabulate import tabulate


MODEL_NAME = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
SYSTEM_PROMPT = "Ты - полезный помощник. Отвечай на русском языке."


@dataclass(frozen=True)
class TemperatureSetting:
    """Настройки одного запуска запроса."""

    value: float
    color_name: str
    color: str


@dataclass
class QueryResult:
    """Ответ API и метрики его получения."""

    temperature: float
    answer: str
    elapsed_seconds: float
    total_tokens: int


SETTINGS = (
    TemperatureSetting(0.0, "Синий", Fore.BLUE),
    TemperatureSetting(0.7, "Зеленый", Fore.GREEN),
    TemperatureSetting(1.2, "Красный", Fore.RED),
)


def create_client() -> OpenAI | None:
    """Создаёт клиент, если ключ DeepSeek указан в окружении."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("Ошибка: переменная окружения DEEPSEEK_API_KEY не найдена.")
        print("Укажите API-ключ DeepSeek и запустите программу снова.")
        return None
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def get_answer(client: OpenAI, user_query: str, temperature: float) -> QueryResult:
    """Отправляет один запрос и возвращает текст, время и число токенов."""
    started_at = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=temperature,
            max_tokens=1000,
        )
    except APIConnectionError as error:
        raise RuntimeError("Ошибка сети: не удалось подключиться к DeepSeek API.") from error
    except APIStatusError as error:
        detail = getattr(error, "message", str(error))
        raise RuntimeError(f"Ошибка DeepSeek API (код {error.status_code}): {detail}") from error
    except OpenAIError as error:
        raise RuntimeError(f"Ошибка при обращении к DeepSeek API: {error}") from error

    elapsed = time.perf_counter() - started_at
    choices = getattr(response, "choices", [])
    content = choices[0].message.content if choices else None
    answer = content.strip() if isinstance(content, str) and content.strip() else "DeepSeek не вернул текст ответа."
    usage = getattr(response, "usage", None)
    tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return QueryResult(temperature, answer, elapsed, tokens)


def print_answer(result: QueryResult, setting: TemperatureSetting) -> None:
    """Печатает ответ; цвет применяется только к его содержимому."""
    print(f"\n--- Ответ при temperature={setting.value:.1f} ({setting.color_name}) ---")
    print(f"{setting.color}{result.answer}{Fore.RESET}")
    print("--- Конец ответа ---")


def print_comparison(results: list[QueryResult]) -> None:
    """Выводит сводную таблицу метрик с выравниванием колонок."""
    rows = [[f"{item.temperature:.1f}", f"{item.elapsed_seconds:.2f}", item.total_tokens] for item in results]
    print("\n=== СРАВНЕНИЕ ОТВЕТОВ ===")
    try:
        print(tabulate(rows, headers=["Температура", "Время (сек)", "Токены"], tablefmt="grid"))
    except (TypeError, ValueError) as error:
        print(f"Не удалось отформатировать таблицу: {error}")


def compare_temperatures(client: OpenAI, user_query: str) -> None:
    """Последовательно получает и отображает три ответа DeepSeek."""
    print("\nВыполняю запросы с разной температурой...")
    results: list[QueryResult] = []
    for setting in SETTINGS:
        try:
            result = get_answer(client, user_query, setting.value)
        except RuntimeError as error:
            print(f"\nНе удалось получить ответ при temperature={setting.value:.1f}: {error}")
            continue
        results.append(result)
        print_answer(result, setting)

    if results:
        print_comparison(results)
    else:
        print("\nНе удалось получить ни одного ответа. Проверьте ключ, сеть и лимиты API.")


def run_menu(client: OpenAI) -> None:
    """Запускает главное меню до тех пор, пока пользователь не выберет выход."""
    print("=== Сравнение температур DeepSeek ===")
    print("Добро пожаловать! Я сравню ответы нейросети при разных температурах.")
    while True:
        print("\nГлавное меню:")
        print("1. Ввести запрос")
        print("2. Выход")
        choice = input("Выберите действие: ").strip()

        if choice == "1":
            user_query = input("Введите ваш запрос к нейросети: ").strip()
            if not user_query:
                print("Ошибка: запрос не должен быть пустым.")
                continue
            compare_temperatures(client, user_query)
        elif choice == "2":
            print("До свидания!")
            return
        else:
            print("Ошибка: выберите пункт 1 или 2.")


def main() -> None:
    """Проверяет окружение и запускает интерактивное приложение."""
    init(autoreset=True)
    client = create_client()
    if client is not None:
        run_menu(client)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nРабота программы завершена пользователем.")
