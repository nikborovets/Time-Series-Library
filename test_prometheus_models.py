#!/usr/bin/env python3
"""
Скрипт для тестирования различных моделей временных рядов на данных из Prometheus.
Автоматически запускает обучение и тестирование нескольких популярных моделей.
"""

import subprocess
import sys
import os
import time
import json
import logging
import argparse
from datetime import datetime

# Список моделей для тестирования (отсортированы по сложности - от простых к сложным)
MODELS_TO_TEST = [
    'DLinear',     # Простая, но эффективная
    'NLinear',     # Нелинейная версия Linear
    'PatchTST',    # Transformer на патчах - быстрый
    'iTransformer', # Инвертированный Transformer
    'Autoformer',  # Автокорреляционный Transformer
    'FEDformer',   # Fourier Enhanced Decomposed Transformer
    'Informer',    # Классический Informer
    'TimesNet',    # Многопериодный анализ - медленный
    'Crossformer', # Cross-dimension Transformer
    'Transformer', # Ванильный Transformer - самый медленный
]

# Базовые параметры для экспериментов (оптимизированы для качественного ночного обучения + доп. фичи)
BASE_PARAMS = {
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'data': 'prometheus',
    'root_path': './',
    'data_path': '',  # Не используется для prometheus
    'features': 'S',  # Univariate forecasting
    'target': 'common_delayp90',
    'freq': '15s',
    'checkpoints': './checkpoints/',
    'seq_len': 192,     # Увеличил историю (48 минут при 15s интервале)
    'label_len': 96,    # Увеличил стартовые токены (24 минуты)
    'pred_len': 96,     # Предсказание (24 минуты)
    'enc_in': 1,        # Размер входа энкодера (1 для univariate)
    'dec_in': 1,        # Размер входа декодера
    'c_out': 1,         # Размер выхода (1 для univariate)
    'd_model': 768,     # Увеличил размерность модели для лучшего качества
    'n_heads': 8,
    'e_layers': 3,      # Увеличил количество слоев энкодера
    'd_layers': 2,      # Увеличил количество слоев декодера
    'd_ff': 3072,       # Увеличил размерность feed-forward (4 * d_model)
    'dropout': 0.05,    # Уменьшил dropout для лучшего обучения
    'activation': 'gelu',
    'embed': 'timeF',
    'batch_size': 16,        # Увеличил batch size для стабильности
    'learning_rate': 0.00005, # Уменьшил LR для стабильного обучения
    'train_epochs': 80,      # Увеличил для качественного ночного обучения
    'patience': 15,          # Увеличил patience для долгого обучения
    'lradj': 'cosine',       # Косинусный планировщик LR
    'gpu': 0,
    'itr': 1,
    'des': 'prometheus_night_training_enhanced',
    
    # 🔧 Дополнительные фичи для улучшения качества:
    # Декомпозиция временных рядов
    'moving_avg': 25,        # Окно скользящего среднего (должно быть меньше seq_len)
    'decomp_method': 'moving_avg',  # Метод декомпозиции тренд+сезонность
    'use_norm': 1,           # Нормализация данных (int: 1=True, 0=False)
    
    # Оптимизация внимания
    'factor': 3,             # Коэффициент прореживания для Informer (уменьшает сложность)
    
    # Многомасштабный анализ (ИСПРАВЛЕНО: убрал down_sampling - может вызывать проблемы)
    # 'down_sampling_layers': 1,     # Убрал - может конфликтовать с некоторыми моделями
    # 'down_sampling_method': 'avg', # Убрал
    # 'down_sampling_window': 2,     # Убрал
    
    # Аугментация данных для робастности
    'augmentation_ratio': 1,       # 1x аугментация данных
    'jitter': True,               # Добавление легкого шума (store_true флаг)
    'scaling': True,              # Масштабирование временных рядов (store_true флаг)
    'seed': 2021,                 # Фиксированный seed для воспроизводимости
    
    # GPU параметры
    'use_gpu': True,             # Использование GPU (bool значение)
    
    # ИСПРАВЛЕНО: Добавил недостающие параметры для совместимости
    'expand': 2,                 # Expansion factor для некоторых моделей
    'd_conv': 4,                 # Conv kernel size для некоторых моделей
}

def run_experiment(model_name, additional_params=None, timeout=7200):
    """Запуск эксперимента для конкретной модели"""
    logging.info(f"\n{'='*60}")
    logging.info(f"Тестирование модели: {model_name}")
    logging.info(f"{'='*60}")
    
    # Подготовка параметров
    params = BASE_PARAMS.copy()
    params['model'] = model_name
    params['model_id'] = f'{model_name}_prometheus'
    
    # Добавление специфичных для модели параметров
    if additional_params:
        params.update(additional_params)
    
    # Формирование команды
    cmd = ['python', 'run.py']
    
    # Список флагов store_true из run.py
    store_true_flags = {
        'inverse', 'use_amp', 'use_multi_gpu', 'jitter', 'scaling', 
        'permutation', 'randompermutation', 'magwarp', 'timewarp', 
        'windowslice', 'windowwarp', 'rotation', 'spawner', 'dtwwarp', 
        'shapedtwwarp', 'wdba', 'discdtw', 'discsdtw'
    }
    
    for key, value in params.items():
        if key in store_true_flags:
            if value:  # Добавляем флаг только если True
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    
    logging.info(f"Команда: {' '.join(cmd)}")
    logging.info(f"Начало обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Запуск обучения
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        end_time = time.time()
        
        duration = end_time - start_time
        logging.info(f"Время выполнения: {duration:.2f} секунд")
        
        if result.returncode == 0:
            logging.info(f"✅ Модель {model_name} успешно обучена!")
            logging.info("STDOUT (последние 1000 символов):")
            logging.info(result.stdout[-1000:])
        else:
            logging.error(f"❌ Ошибка при обучении модели {model_name}")
            logging.error("STDERR:")
            logging.error(result.stderr[-1000:])
            logging.error("STDOUT:")
            logging.error(result.stdout[-1000:])
            
        return result.returncode == 0, duration
        
    except subprocess.TimeoutExpired:
        logging.error(f"⏰ Таймаут при обучении модели {model_name}")
        return False, timeout
    except Exception as e:
        logging.error(f"💥 Исключение при обучении модели {model_name}: {e}")
        return False, 0

def setup_logging():
    """Настройка логирования"""
    log_filename = f"prometheus_models_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_filename

def save_results(results, log_filename):
    """Сохранение результатов в JSON"""
    results_filename = log_filename.replace('.log', '_results.json')
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results_filename

def main():
    """Основная функция для запуска всех экспериментов"""
    parser = argparse.ArgumentParser(description='Тестирование моделей временных рядов на данных Prometheus')
    parser.add_argument('--models', nargs='+', default=MODELS_TO_TEST,
                       help='Список моделей для тестирования')
    parser.add_argument('--epochs', type=int, default=80,
                       help='Количество эпох обучения')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Размер батча')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Использовать GPU')
    parser.add_argument('--timeout', type=int, default=14400,
                       help='Таймаут в секундах для каждой модели (4 часа по умолчанию)')
    
    args = parser.parse_args()
    
    log_filename = setup_logging()
    
    logging.info("🚀 Начинаем тестирование моделей на данных из Prometheus")
    logging.info(f"Модели для тестирования: {args.models}")
    logging.info(f"Эпохи: {args.epochs}, Batch size: {args.batch_size}")
    logging.info(f"GPU: {'Да' if args.use_gpu else 'Нет'}, Таймаут: {args.timeout}s")
    logging.info(f"Данные: 90600 точек с 2025-04-27 18:00:00 по 2025-05-13 11:41:00")
    logging.info(f"Лог файл: {log_filename}")
    
    # Обновляем базовые параметры
    BASE_PARAMS['train_epochs'] = args.epochs
    BASE_PARAMS['batch_size'] = args.batch_size
    BASE_PARAMS['use_gpu'] = args.use_gpu
    
    results = {}
    total_start_time = time.time()
    
    # Специфичные параметры для некоторых моделей (оптимизированы)
    model_specific_params = {
        'PatchTST': {
            'patch_len': 16,              # Патчи по 16 точек (4 минуты)
        },
        'TimesNet': {
            'top_k': 5,                   # Топ-5 периодичностей
            'num_kernels': 6,             # 6 сверточных ядер
        },
        'Crossformer': {
            'seg_len': 48,                # Увеличил сегменты до 12 минут
        },
        'FEDformer': {
            'version': 'Fourier',         # Фурье декомпозиция
            'mode_select': 'random',      # Случайный выбор мод
            'modes': 64,                  # Увеличил количество мод
        },
        'Informer': {
            'factor': 5,                  # Больше прореживания для Informer
        },
        'Autoformer': {
            'moving_avg': 25,             # Специальное окно для Autoformer (безопасное значение)
        },
        'iTransformer': {
            'channel_independence': 1,     # Независимость каналов
        }
    }
    
    for i, model in enumerate(args.models, 1):
        logging.info(f"\n📊 Прогресс: {i}/{len(args.models)}")
        
        additional_params = model_specific_params.get(model, {})
        success, duration = run_experiment(model, additional_params, args.timeout)
        
        results[model] = {
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        # Сохраняем промежуточные результаты
        results_file = save_results(results, log_filename)
        logging.info(f"💾 Промежуточные результаты сохранены в {results_file}")
        
        # Небольшая пауза между экспериментами
        if i < len(args.models):
            logging.info("⏳ Пауза 30 секунд перед следующим экспериментом...")
            time.sleep(30)
    
    # Итоговый отчет
    total_duration = time.time() - total_start_time
    logging.info(f"\n{'='*80}")
    logging.info("📈 ИТОГОВЫЙ ОТЧЕТ")
    logging.info(f"{'='*80}")
    logging.info(f"Общее время выполнения: {total_duration:.2f} секунд ({total_duration/60:.1f} минут)")
    
    # Финальное сохранение результатов
    final_results_file = save_results(results, log_filename)
    logging.info(f"🎯 Финальные результаты сохранены в {final_results_file}")
    
    successful_models = []
    failed_models = []
    
    for model, result in results.items():
        status = "✅ Успешно" if result['success'] else "❌ Ошибка"
        duration_str = f"{result['duration']:.2f}s"
        logging.info(f"{model:15} | {status:12} | {duration_str:>10}")
        
        if result['success']:
            successful_models.append(model)
        else:
            failed_models.append(model)
    
    logging.info(f"\n📊 Статистика:")
    logging.info(f"Успешно обучено: {len(successful_models)}/{len(MODELS_TO_TEST)} моделей")
    
    if successful_models:
        logging.info(f"\n🎉 Успешные модели: {', '.join(successful_models)}")
        
        # Сортировка по времени обучения
        successful_with_time = [(model, results[model]['duration']) for model in successful_models]
        successful_with_time.sort(key=lambda x: x[1])
        
        logging.info(f"\n⚡ Самые быстрые модели:")
        for model, duration in successful_with_time[:3]:
            logging.info(f"  {model}: {duration:.2f}s")
    
    if failed_models:
        logging.info(f"\n💔 Неудачные модели: {', '.join(failed_models)}")
    
    logging.info(f"\n📁 Результаты сохранены в директории: ./checkpoints/")
    logging.info(f"🔍 Для анализа результатов используйте файлы в checkpoints/")

if __name__ == "__main__":
    main() 