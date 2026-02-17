# CP — Agent-Based FX Market Model

Проект моделирует FX-рынок с несколькими площадками исполнения (CLOB + AMM: CPMM/HFMM), стресс-сценарием и метриками исполнения для анализа гипотез H1–H4.

## Что внутри

- `main_fx.py` — основной entrypoint для запуска симуляции и построения графиков
- `main.py` — перебор конфигураций агентов (исследовательский скрипт)
- `AgentBasedModel/` — ядро модели: агенты, площадки, симулятор, метрики, визуализация
- `test_full.py`, `test_theta.py` — smoke/validation-скрипты

## Быстрый старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main_fx.py
```

## Минимальный сценарий для разработки

```bash
# запуск основной модели
python main_fx.py

# проверки/валидация
python test_theta.py
python test_full.py
```

## Рекомендуемый workflow в команде

1. Создавайте ветку от `main`: `feature/<topic>` или `fix/<topic>`
2. Делайте маленькие атомарные коммиты
3. Перед PR прогоняйте локально `python test_theta.py` (и по возможности `test_full.py`)
4. Открывайте Pull Request в `main` с описанием:
   - что изменено,
   - как проверить,
   - какие метрики/графики затронуты.

## Структура пакета

```text
AgentBasedModel/
  agents/         # трейдеры и поведенческие стратегии
  environment/    # процесс среды/шоки
  events/         # рыночные события
  metrics/        # логирование и агрегации метрик
  simulator/      # симуляторы (общий + FX)
  states/         # состояние рынка
  utils/          # математика, ордера
  venues/         # CLOB и AMM-механизмы
  visualization/  # графики и аналитика
```

## Публикация на GitHub

```bash
git init
git add .
git commit -m "Initial commit: FX ABM model"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

После публикации добавьте коллег в `Settings → Collaborators` и включите branch protection для `main` (merge только через PR).
