"""Консольный решатель логических, алгоритмических и аналитических задач."""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIStatusError, OpenAI, OpenAIError
from tabulate import tabulate

MODEL_NAME = "deepseek-v4-pro"
BASE_URL = "https://api.deepseek.com"
METHODS = {
    "способ1": "1. Прямой ответ",
    "способ2": "2. Пошаговое решение",
    "способ3": "3. Создание и использование промпта",
    "способ4": "4. Группа экспертов",
}


@dataclass
class Result:
    """Ответ одного способа и его измеримые показатели."""

    answer: str
    tokens: int
    elapsed: float
    score: float = 0.0
    comment: str = "Оценка ещё не выполнена."


class TaskSolver:
    """Организует запросы к DeepSeek и хранит состояние приложения."""

    def __init__(self, task_path: str | None = None) -> None:
        self.task_text = ""
        self.task_path: Path | None = None
        self.selected = set(METHODS)
        self.results: dict[str, Result] = {}
        self.client: OpenAI | None = None
        if task_path:
            self.load_task(task_path)

    def load_task(self, file_name: str) -> bool:
        """Загружает задачу из UTF-8 файла после проверки пути."""
        path = Path(file_name).expanduser()
        if not path.is_file():
            print("Ошибка: файл не существует или не является обычным файлом.")
            return False
        try:
            text = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError:
            print("Ошибка: файл должен быть сохранён в кодировке UTF-8.")
            return False
        except OSError as error:
            print(f"Ошибка чтения файла: {error}")
            return False
        if not text:
            print("Ошибка: файл с задачей пуст.")
            return False
        self.task_text, self.task_path = text, path
        self.results.clear()
        print(f"Задача загружена: {path}")
        return True

    def get_client(self) -> OpenAI | None:
        """Создаёт API-клиент только перед фактическим обращением к сети."""
        if self.client:
            return self.client
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("Ошибка: не найдена переменная окружения DEEPSEEK_API_KEY.")
            print("Укажите API-ключ DeepSeek и повторите запуск решения.")
            return None
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)
        return self.client

    @staticmethod
    def response_text(response: Any) -> str:
        """Безопасно извлекает итоговый текст Chat Completions."""
        if not response.choices:
            return "DeepSeek не вернул вариантов ответа."
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) and content.strip() else "DeepSeek не вернул текст ответа."

    def request(self, system: str, user: str) -> tuple[str, int, float]:
        """Выполняет запрос и возвращает текст, токены и затраченное время."""
        client = self.get_client()
        if client is None:
            raise RuntimeError("API-ключ DeepSeek не настроен.")
        started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                extra_body={"thinking": {"type": "disabled"}},
            )
        except APIConnectionError as error:
            raise RuntimeError("Не удалось подключиться к DeepSeek API.") from error
        except APIStatusError as error:
            raise RuntimeError(f"Ошибка DeepSeek API (код {error.status_code}): {error.message}") from error
        except OpenAIError as error:
            raise RuntimeError(f"Ошибка при обращении к DeepSeek API: {error}") from error
        usage = getattr(response, "usage", None)
        return self.response_text(response), int(getattr(usage, "total_tokens", 0) or 0), time.perf_counter() - started

    def solve_method(self, method: str) -> Result:
        """Запускает последовательность запросов, предусмотренную способом."""
        if method == "способ1":
            answer, tokens, elapsed = self.request(
                "Ты - помощник, решающий логические, алгоритмические и аналитические задачи. Ответь на русском языке.", self.task_text)
        elif method == "способ2":
            answer, tokens, elapsed = self.request(
                "Ты - помощник, решающий задачи. Решай задачу пошагово, объясняя каждый шаг. Ответь на русском языке. Решай пошагово.", self.task_text)
        elif method == "способ3":
            prompt, first_tokens, first_elapsed = self.request(
                "Ты - эксперт по созданию эффективных промптов. Создай детальный промпт для решения следующей задачи. Промпт должен быть четким и структурированным. Ответь на русском языке.", self.task_text)
            # Промпт нужен только для второго запроса и не показывается пользователю.
            answer, second_tokens, second_elapsed = self.request(prompt, self.task_text)
            tokens, elapsed = first_tokens + second_tokens, first_elapsed + second_elapsed
        elif method == "способ4":
            answer, tokens, elapsed = self.request(
                "Ты - группа из 4 экспертов: Аналитик, Инженер, Инженер-2 и Скептик. Каждый эксперт должен дать свое решение задачи. Эксперты не должны видеть ответы друг друга. Ответ представь в формате: [Аналитик]: ... [Инженер]: ... [Инженер-2]: ... [Скептик]: ... Ответь на русском языке.", self.task_text)
        else:
            raise ValueError(f"Неизвестный способ: {method}")
        return Result(answer, tokens, elapsed)

    def run_solutions(self) -> None:
        """Решает задачу выбранными способами, затем запускает оценку."""
        if not self.task_text:
            print("Сначала загрузите файл с задачей.")
            return
        if not self.selected:
            print("Предупреждение: не выбран ни один способ решения.")
            return
        if not self.get_client():
            return
        self.results.clear()
        for method in METHODS:
            if method not in self.selected:
                continue
            print(f"\nВыполняется: {METHODS[method]}...")
            try:
                result = self.solve_method(method)
            except RuntimeError as error:
                print(f"Не удалось выполнить {METHODS[method]}: {error}")
                continue
            self.results[method] = result
            print(f"=== {METHODS[method]} ===\n{result.answer}")
        if self.results:
            self.evaluate_results()

    @staticmethod
    def parse_json(text: str) -> dict[str, Any] | None:
        """Читает JSON даже если модель поместила его в Markdown-блок."""
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        candidates = [cleaned]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            candidates.append(cleaned[start:end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def evaluate_results(self) -> None:
        """Просит нейросеть оценить все полученные ответы по шкале от 1 до 10."""
        answers = "\n\n".join(f"{key} ({METHODS[key]}):\n{item.answer}" for key, item in self.results.items())
        shape = ",\n  ".join(f'"{key}": {{"оценка": 8, "комментарий": "..."}}' for key in self.results)
        system = (
            "Ты - объективный оценщик решений. Оцени каждый представленный ответ по 10-балльной шкале "
            "по правильности, полноте, логичности и обоснованности. Верни только валидный JSON без Markdown:\n{\n  "
            + shape + "\n}"
        )
        try:
            answer, _, _ = self.request(system, f"Задача:\n{self.task_text}\n\nОтветы:\n{answers}")
        except RuntimeError as error:
            print(f"Не удалось оценить точность: {error}")
            return
        scores = self.parse_json(answer)
        if scores is None:
            print("Предупреждение: оценка вернула некорректный JSON; использованы нули.")
            return
        for key, result in self.results.items():
            data = scores.get(key, {})
            if not isinstance(data, dict):
                continue
            try:
                result.score = max(0.0, min(10.0, float(data.get("оценка", 0))))
            except (TypeError, ValueError):
                result.score = 0.0
            result.comment = str(data.get("комментарий", "Нет комментария."))

    def compare_results(self) -> None:
        """Печатает таблицу метрик и лучший способ по оценке нейросети."""
        if not self.results:
            print("Нет результатов для сравнения. Сначала запустите решение.")
            return
        rows = [[METHODS[key], item.tokens, f"{item.elapsed:.2f}с", f"{item.score:.1f}"] for key, item in self.results.items()]
        print("\n=== СРАВНЕНИЕ СПОСОБОВ РЕШЕНИЯ ===")
        print(tabulate(rows, headers=["Способ", "Токены", "Время", "Оценка точности"], tablefmt="grid"))
        best_key, best = max(self.results.items(), key=lambda pair: pair[1].score)
        print(f"Лучший способ: {METHODS[best_key]} (оценка: {best.score:.1f})")
        for key, item in self.results.items():
            print(f"{METHODS[key]} — комментарий оценщика: {item.comment}")

    def choose_methods(self) -> None:
        """Включает или отключает способы, пока пользователь не подтвердит выбор."""
        while True:
            print("\nВыберите способы решения (введите номер для переключения):")
            for number, key in enumerate(METHODS, 1):
                print(f"[{'✓' if key in self.selected else ' '}] {number}. {METHODS[key][3:]}")
            print("0. Готово")
            choice = input("Ваш выбор: ").strip()
            if choice == "0":
                return
            if choice in {"1", "2", "3", "4"}:
                key = list(METHODS)[int(choice) - 1]
                self.selected.symmetric_difference_update({key})
            else:
                print("Ошибка: введите номер от 0 до 4.")

    def menu(self) -> None:
        """Запускает главный цикл, который завершается только пунктом «Выход»."""
        print("=== Решатель задач с ИИ ===")
        print("Привет! Я помогу решить логические, алгоритмические и аналитические задачи.")
        while True:
            current = str(self.task_path) if self.task_path else "не загружена"
            status = "все активны" if len(self.selected) == 4 else f"активно: {len(self.selected)} из 4"
            print(f"\nТекущая задача: {current}\n\nГлавное меню:")
            print("1. Загрузить задачу из файла")
            print(f"2. Выбрать способы решения ({status})")
            print("3. Запустить решение\n4. Сравнить результаты\n5. Выход")
            choice = input("Выберите действие: ").strip()
            if choice == "1":
                self.load_task(input("Путь к UTF-8 файлу с задачей: ").strip().strip('"'))
            elif choice == "2":
                self.choose_methods()
            elif choice == "3":
                self.run_solutions()
            elif choice == "4":
                self.compare_results()
            elif choice == "5":
                print("Спасибо за использование решателя. До свидания!")
                return
            else:
                print("Ошибка: выберите пункт от 1 до 5.")


def main() -> None:
    """Принимает необязательный путь к файлу в командной строке и запускает меню."""
    parser = argparse.ArgumentParser(description="Решатель задач с помощью DeepSeek API")
    parser.add_argument("task_file", nargs="?", help="путь к UTF-8 файлу с задачей")
    args = parser.parse_args()
    TaskSolver(args.task_file).menu()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nРабота программы завершена пользователем.")
