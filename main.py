#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Приложение для сравнения трех моделей DeepSeek API.
Сравнивает скорость, стоимость и качество ответов.
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tabulate import tabulate
from colorama import init, Fore, Style

# Инициализация colorama для Windows
init(autoreset=True)

# Конфигурация моделей
MODELS = {
    "flash": {
        "name": "deepseek-v4-flash",
        "display": "Flash (слабая)",
        "price_per_1k": 0.01,  # $0.01 за 1K токенов
        "type": "слабая"
    },
    "vision": {
        "name": "deepseek-v4-flash-vision-exp",
        "display": "Vision (средняя)",
        "price_per_1k": 0.03,  # $0.03 за 1K токенов
        "type": "средняя"
    },
    "pro": {
        "name": "deepseek-v4-pro",
        "display": "Pro (сильная)",
        "price_per_1k": 0.05,  # $0.05 за 1K токенов
        "type": "сильная"
    }
}

# Системный промпт для оценки качества
EVALUATION_SYSTEM_PROMPT = """Ты - объективный оценщик качества ответов. Оцени каждый из 3 представленных ответов по 10-балльной шкале (от 1 до 10) по следующим критериям:
- Правильность и точность
- Полнота и глубина
- Логичность и структурированность

Также укажи, какой ответ наиболее качественный, а какой наименее.

Предоставь оценку в формате JSON:
{
  "flash": {"оценка": 7, "комментарий": "..."},
  "vision": {"оценка": 8, "комментарий": "..."},
  "pro": {"оценка": 9, "комментарий": "..."},
  "лучший": "pro",
  "худший": "flash"
}

Важно: Ответ должен быть только в формате JSON, без дополнительного текста."""


class DeepSeekComparator:
    """Класс для сравнения моделей DeepSeek API."""

    def __init__(self):
        """Инициализация клиента API."""
        self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            print(Fore.RED + "❌ Ошибка: Переменная окружения DEEPSEEK_API_KEY не найдена.")
            print("Пожалуйста, установите API ключ: export DEEPSEEK_API_KEY='ваш_ключ'")
            sys.exit(1)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.results = {}

    def check_api_key(self) -> bool:
        """Проверка наличия API ключа."""
        return bool(self.api_key)

    def send_request(self, model_key: str, user_query: str) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Отправка запроса к API DeepSeek.

        Args:
            model_key: Ключ модели (flash, vision, pro)
            user_query: Текст запроса пользователя

        Returns:
            Tuple[Optional[str], Optional[Dict]]: (текст ответа, метрики) или (None, None) при ошибке
        """
        model_config = MODELS[model_key]
        model_name = model_config["name"]

        try:
            start_time = time.time()

            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Ты - полезный помощник. Отвечай на русском языке."},
                    {"role": "user", "content": user_query}
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
            )

            elapsed_time = time.time() - start_time

            # Извлечение данных из ответа
            answer_text = response.choices[0].message.content
            total_tokens = response.usage.total_tokens

            # Расчет стоимости
            price_per_1k = model_config["price_per_1k"]
            cost = (total_tokens / 1000) * price_per_1k

            metrics = {
                "time": round(elapsed_time, 2),
                "tokens": total_tokens,
                "cost": cost,
                "cost_formatted": f"${cost:.4f}"
            }

            return answer_text, metrics

        except Exception as e:
            print(Fore.RED + f"❌ Ошибка при запросе к модели {model_config['display']}: {str(e)}")
            return None, None

    def evaluate_answers(self, answers: Dict[str, str]) -> Optional[Dict]:
        """
        Оценка качества ответов с помощью сильной модели (Pro).

        Args:
            answers: Словарь с ответами моделей {key: answer_text}

        Returns:
            Optional[Dict]: Результаты оценки или None при ошибке
        """
        print(Fore.YELLOW + "\n🔄 Оценка качества ответов...")

        # Формируем запрос для оценки
        evaluation_prompt = f"""
Оцени следующие ответы на запрос пользователя:

Ответ Flash (слабая модель):
{answers.get('flash', 'Нет ответа')}

Ответ Vision (средняя модель):
{answers.get('vision', 'Нет ответа')}

Ответ Pro (сильная модель):
{answers.get('pro', 'Нет ответа')}

Оцени каждый ответ по 10-балльной шкале и укажи лучший и худший.
"""

        try:
            response = self.client.chat.completions.create(
                model=MODELS["pro"]["name"],
                messages=[
                    {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                    {"role": "user", "content": evaluation_prompt}
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}}
            )

            evaluation_text = response.choices[0].message.content

            # Парсим JSON из ответа
            # Ищем JSON в ответе (на случай, если есть лишний текст)
            json_start = evaluation_text.find('{')
            json_end = evaluation_text.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                print(Fore.RED + "❌ Не удалось найти JSON в ответе оценщика")
                return None

            json_str = evaluation_text[json_start:json_end]

            try:
                evaluation_result = json.loads(json_str)
                return evaluation_result
            except json.JSONDecodeError as e:
                print(Fore.RED + f"❌ Ошибка парсинга JSON: {e}")
                print(Fore.YELLOW + f"Полученный JSON: {json_str}")
                return None

        except Exception as e:
            print(Fore.RED + f"❌ Ошибка при оценке ответов: {str(e)}")
            return None

    def process_query(self, user_query: str) -> bool:
        """
        Обработка запроса пользователя.

        Args:
            user_query: Текст запроса

        Returns:
            bool: True если успешно, False при ошибке
        """
        if not user_query or not user_query.strip():
            print(Fore.RED + "❌ Запрос не может быть пустым.")
            return False

        print(Fore.CYAN + "\n⏳ Выполняю запросы на всех моделях...")

        # Словари для хранения результатов
        answers = {}
        metrics = {}
        successful_models = 0

        # Отправляем запросы на все модели
        for model_key in ["flash", "vision", "pro"]:
            model_config = MODELS[model_key]
            print(Fore.CYAN + f"🔄 Запрос к {model_config['display']}...")

            answer, metric = self.send_request(model_key, user_query)

            if answer is not None and metric is not None:
                answers[model_key] = answer
                metrics[model_key] = metric
                successful_models += 1
            else:
                answers[model_key] = f"[Ошибка: не удалось получить ответ от {model_config['display']}]"
                metrics[model_key] = {"time": 0, "tokens": 0, "cost": 0, "cost_formatted": "$0.0000"}

        if successful_models == 0:
            print(Fore.RED + "❌ Не удалось получить ответ ни от одной модели.")
            return False

        # Выводим ответы моделей
        self.display_answers(answers, metrics)

        # Оцениваем качество ответов
        evaluation = self.evaluate_answers(answers)

        # Выводим сравнительную таблицу
        self.display_comparison_table(metrics, evaluation)

        # Выводим итоговые выводы
        if evaluation:
            self.display_final_conclusions(metrics, evaluation)
        else:
            print(Fore.YELLOW + "\n⚠️ Оценка качества не была выполнена. Показаны только метрики.")

        return True

    def display_answers(self, answers: Dict[str, str], metrics: Dict[str, Dict]):
        """Отображение ответов моделей."""
        print(Fore.CYAN + "\n" + "=" * 60)
        print(Fore.GREEN + Style.BRIGHT + "=== ОТВЕТЫ МОДЕЛЕЙ ===")
        print(Fore.CYAN + "=" * 60)

        # Определяем порядок отображения
        order = [
            ("flash", "Слабая модель"),
            ("vision", "Средняя модель"),
            ("pro", "Сильная модель")
        ]

        for model_key, model_type in order:
            model_config = MODELS[model_key]
            print(Fore.YELLOW + f"\n--- {model_type} ({model_config['name']}) ---")
            print(Fore.WHITE + answers.get(model_key, "Нет ответа"))

            if model_key in metrics:
                m = metrics[model_key]
                print(Fore.CYAN + f"Время: {m['time']} сек | Токены: {m['tokens']} | Стоимость: {m['cost_formatted']}")

    def display_comparison_table(self, metrics: Dict[str, Dict], evaluation: Optional[Dict]):
        """Отображение сравнительной таблицы."""
        print(Fore.CYAN + "\n" + "=" * 60)
        print(Fore.GREEN + Style.BRIGHT + "=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
        print(Fore.CYAN + "=" * 60)

        # Подготовка данных для таблицы
        table_data = []

        # Порядок моделей в таблице
        order = ["flash", "vision", "pro"]
        display_names = {
            "flash": "Flash (слабая)",
            "vision": "Vision (средняя)",
            "pro": "Pro (сильная)"
        }

        # Определяем лучшие показатели для подсветки
        best_time = min([metrics.get(k, {}).get("time", float('inf')) for k in order])
        best_cost = min([metrics.get(k, {}).get("cost", float('inf')) for k in order])

        # Находим лучшую оценку если есть
        best_score = -1
        best_score_model = None
        if evaluation:
            for key in ["flash", "vision", "pro"]:
                if key in evaluation and isinstance(evaluation[key], dict):
                    score = evaluation[key].get("оценка", 0)
                    if score > best_score:
                        best_score = score
                        best_score_model = key

        for model_key in order:
            model_config = MODELS[model_key]
            m = metrics.get(model_key, {})

            # Форматируем строки с возможной подсветкой
            time_str = f"{m.get('time', 0):.2f}"
            tokens_str = str(m.get('tokens', 0))
            cost_str = m.get('cost_formatted', '$0.0000')

            # Подсветка лучших показателей
            if m.get('time', float('inf')) == best_time:
                time_str = Fore.GREEN + time_str + Style.RESET_ALL
            if m.get('cost', float('inf')) == best_cost and m.get('cost', float('inf')) > 0:
                cost_str = Fore.GREEN + cost_str + Style.RESET_ALL

            # Добавляем оценку качества если есть
            score_str = ""
            if evaluation and model_key in evaluation:
                if isinstance(evaluation[model_key], dict):
                    score = evaluation[model_key].get("оценка", 0)
                    if score > 0:
                        if model_key == best_score_model:
                            score_str = Fore.GREEN + f"{score}/10 ★" + Style.RESET_ALL
                        else:
                            score_str = f"{score}/10"

            table_data.append([
                display_names[model_key],
                time_str,
                tokens_str,
                cost_str,
                score_str if score_str else ""
            ])

        # Создаем таблицу с использованием tabulate
        headers = ["Модель", "Время (сек)", "Токены", "Стоимость", "Оценка"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")

        print(Fore.WHITE + table)

    def display_final_conclusions(self, metrics: Dict[str, Dict], evaluation: Dict):
        """Отображение итоговых выводов."""
        print(Fore.CYAN + "\n" + "=" * 60)
        print(Fore.GREEN + Style.BRIGHT + "=== ИТОГОВОЕ СРАВНЕНИЕ ===")
        print(Fore.CYAN + "=" * 60)

        # Скорость (сортировка по времени)
        time_order = sorted(
            [(key, metrics[key]["time"]) for key in metrics if metrics[key]["time"] > 0],
            key=lambda x: x[1]
        )

        if time_order:
            speed_str = " > ".join([f"{MODELS[k]['display']} ({v:.2f}с)" for k, v in time_order])
            print(Fore.CYAN + f"⏱️ Скорость: {speed_str}")

        # Стоимость (сортировка по стоимости)
        cost_order = sorted(
            [(key, metrics[key]["cost"]) for key in metrics if metrics[key]["cost"] > 0],
            key=lambda x: x[1]
        )

        if cost_order:
            cost_str = " < ".join([f"{MODELS[k]['display']} ({metrics[k]['cost_formatted']})" for k, _ in cost_order])
            print(Fore.CYAN + f"💰 Стоимость: {cost_str}")

        # Качество (из оценки)
        if evaluation:
            quality_items = []
            for key in ["flash", "vision", "pro"]:
                if key in evaluation and isinstance(evaluation[key], dict):
                    score = evaluation[key].get("оценка", 0)
                    quality_items.append((key, score))

            quality_items.sort(key=lambda x: x[1], reverse=True)

            if quality_items:
                quality_str = " > ".join([f"{MODELS[k]['display']} ({v}/10)" for k, v in quality_items])
                print(Fore.CYAN + f"📊 Качество: {quality_str}")

                # Рекомендация
                print(Fore.CYAN + "\n💡 Рекомендация:")
                best_model = quality_items[0][0] if quality_items else None
                if best_model:
                    if best_model == "flash":
                        print(Fore.GREEN + "✅ Для задач, где важна скорость - используйте Flash.")
                    elif best_model == "pro":
                        print(Fore.GREEN + "✅ Для задач, где важно качество - используйте Pro.")
                    elif best_model == "vision":
                        print(Fore.GREEN + "✅ Vision - хороший компромисс между скоростью и качеством.")

                # Комментарии оценщика
                print(Fore.YELLOW + "\n📝 Комментарии оценщика:")
                for key in ["flash", "vision", "pro"]:
                    if key in evaluation and isinstance(evaluation[key], dict):
                        comment = evaluation[key].get("комментарий", "")
                        if comment:
                            print(Fore.WHITE + f"  • {MODELS[key]['display']}: {comment}")

                # Лучший и худший
                if "лучший" in evaluation and evaluation["лучший"] in MODELS:
                    print(Fore.GREEN + f"🏆 Лучший ответ: {MODELS[evaluation['лучший']]['display']}")

                if "худший" in evaluation and evaluation["худший"] in MODELS:
                    print(Fore.RED + f"📉 Худший ответ: {MODELS[evaluation['худший']]['display']}")

    def run(self):
        """Запуск главного меню приложения."""
        print(Fore.CYAN + "=" * 60)
        print(Fore.GREEN + Style.BRIGHT + "=== Сравнение моделей DeepSeek ===")
        print(Fore.CYAN + "=" * 60)
        print(Fore.WHITE + "Добро пожаловать! Я сравню работу трех моделей DeepSeek.")

        while True:
            print(Fore.YELLOW + "\nГлавное меню:")
            print(Fore.WHITE + "1. Ввести запрос")
            print(Fore.WHITE + "2. Выход")

            choice = input(Fore.CYAN + "Выберите действие: ").strip()

            if choice == "1":
                print(Fore.YELLOW + "\nВведите ваш запрос к нейросети:")
                user_query = input(Fore.WHITE + "> ").strip()

                if not user_query:
                    print(Fore.RED + "❌ Запрос не может быть пустым. Попробуйте снова.")
                    continue

                # Обработка запроса
                self.process_query(user_query)

                print(Fore.YELLOW + "\nНажмите Enter для продолжения...")
                input()

            elif choice == "2":
                print(Fore.GREEN + "\nДо свидания!")
                break

            else:
                print(Fore.RED + "❌ Неверный выбор. Пожалуйста, выберите 1 или 2.")


def main():
    """Точка входа в приложение."""
    try:
        app = DeepSeekComparator()
        app.run()
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\nПрограмма прервана пользователем.")
        sys.exit(0)
    except Exception as e:
        print(Fore.RED + f"\n❌ Непредвиденная ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()