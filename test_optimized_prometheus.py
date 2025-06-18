#!/usr/bin/env python3
"""
Оптимизированное тестирование моделей на данных Prometheus
с улучшенными гиперпараметрами для высокочастотных данных
"""

import subprocess
import time
import json
import logging
from datetime import datetime
import os

# Настройка логирования
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"prometheus_optimized_test_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Оптимизированные параметры для высокочастотных данных
OPTIMIZED_CONFIG = {
    # Основные параметры модели
    'task_name': 'long_term_forecast',
    'is_training': 1,
    'data': 'prometheus',
    'root_path': './',
    'data_path': '',
    'features': 'S',
    'target': 'common_delayp90',
    'freq': '15s',
    'checkpoints': './checkpoints/',
    
    # 🎯 ОПТИМИЗИРОВАННЫЕ параметры для высокочастотных данных
    'seq_len': 48,           # 12 минут истории (вместо 48 минут)
    'label_len': 24,         # 6 минут перекрытия
    'pred_len': 24,          # 6 минут предсказания (вместо 24 минут)
    
    # Архитектура модели - уменьшенная для быстрых данных
    'enc_in': 1,
    'dec_in': 1,
    'c_out': 1,
    'd_model': 256,          # Уменьшено с 768
    'n_heads': 4,            # Уменьшено с 8
    'e_layers': 2,           # Уменьшено с 3
    'd_layers': 1,           # Уменьшено с 2
    'd_ff': 1024,            # Уменьшено с 3072
    'dropout': 0.1,          # Увеличено для регуляризации
    'activation': 'gelu',
    'embed': 'timeF',
    
    # Обучение - более агрессивное для быстрой сходимости
    'batch_size': 64,        # Увеличено с 32
    'learning_rate': 1e-4,   # Увеличено с 5e-5
    'train_epochs': 50,      # Уменьшено с 80
    'patience': 10,          # Уменьшено с 15
    'lradj': 'cosine',
    
    # GPU
    'gpu': 0,
    'itr': 1,
    'des': 'prometheus_optimized_highfreq',
    
    # Улучшенная предобработка
    'moving_avg': 11,        # Нечетное число для корректной декомпозиции
    'decomp_method': 'moving_avg',
    'use_norm': 1,
    
    # Оптимизация внимания
    'factor': 5,             # Увеличено для большего прореживания
    
    # Аугментация - более консервативная
    'augmentation_ratio': 1,  # Исправлено на int
    'jitter': True,
    'scaling': True,
    'seed': 2021,
    
    # GPU
    'use_gpu': 'True',
    'expand': 2,
    'd_conv': 4
}

# Модели для тестирования (начинаем с самых быстрых)
MODELS_TO_TEST = [
    'DLinear',      # Самая быстрая линейная модель
    'LightTS',      # Легкая модель временных рядов
    'PatchTST',     # Эффективная для TS
    'iTransformer', # Современная
    'TimesNet',     # Мощная модель
]

# Специфичные параметры для каждой модели
MODEL_SPECIFIC_PARAMS = {
    'PatchTST': {'patch_len': 8},  # Уменьшено с 16
    'iTransformer': {'channel_independence': 1},
    'Informer': {'factor': 5},
    'FEDformer': {'version': 'Fourier', 'mode_select': 'random', 'modes': 32},
    'Crossformer': {'seg_len': 6},
}

def run_model_test(model_name, timeout_seconds=3600):
    """Запуск тестирования одной модели с оптимизированными параметрами"""
    
    # Базовые параметры
    cmd_params = OPTIMIZED_CONFIG.copy()
    cmd_params['model'] = model_name
    cmd_params['model_id'] = f"{model_name}_prometheus_optimized"
    
    # Добавляем специфичные параметры модели
    if model_name in MODEL_SPECIFIC_PARAMS:
        cmd_params.update(MODEL_SPECIFIC_PARAMS[model_name])
    
    # Формируем команду
    cmd_parts = ['python', 'run.py']
    for key, value in cmd_params.items():
        if key in ['jitter', 'scaling'] and value is True:
            # Эти параметры - флаги без значений
            cmd_parts.append(f'--{key}')
        elif isinstance(value, bool):
            # Остальные булевы параметры с значениями
            cmd_parts.extend([f'--{key}', str(value)])
        else:
            cmd_parts.extend([f'--{key}', str(value)])
    
    cmd_str = ' '.join(cmd_parts)
    logging.info(f"Команда: {cmd_str}")
    
    start_time = time.time()
    logging.info(f"Начало обучения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=os.getcwd()
        )
        
        duration = time.time() - start_time
        logging.info(f"Время выполнения: {duration:.2f} секунд")
        
        if result.returncode == 0:
            logging.info(f"✅ Модель {model_name} успешно обучена!")
            
            # Показываем последние строки вывода с метриками
            stdout_lines = result.stdout.strip().split('\n')
            last_lines = stdout_lines[-20:] if len(stdout_lines) > 20 else stdout_lines
            logging.info("Результаты обучения:")
            for line in last_lines:
                if any(keyword in line.lower() for keyword in ['mse:', 'mae:', 'test loss:', 'epoch:']):
                    logging.info(f"  {line}")
            
            return {
                'success': True,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'stdout_last_1000': result.stdout[-1000:] if result.stdout else '',
            }
        else:
            logging.error(f"❌ Ошибка при обучении модели {model_name}")
            logging.error(f"STDERR: {result.stderr}")
            logging.error(f"STDOUT: {result.stdout}")
            
            return {
                'success': False,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'error': result.stderr,
                'stdout': result.stdout
            }
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        logging.error(f"⏰ Таймаут при обучении модели {model_name}")
        return {
            'success': False,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'error': 'Timeout'
        }
    except Exception as e:
        duration = time.time() - start_time
        logging.error(f"💥 Исключение при обучении модели {model_name}: {str(e)}")
        return {
            'success': False,
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }

def main():
    logging.info("🚀 Начинаем ОПТИМИЗИРОВАННОЕ тестирование моделей на данных из Prometheus")
    logging.info(f"Модели для тестирования: {MODELS_TO_TEST}")
    logging.info(f"Оптимизированные параметры:")
    logging.info(f"  seq_len: {OPTIMIZED_CONFIG['seq_len']} (12 минут)")
    logging.info(f"  pred_len: {OPTIMIZED_CONFIG['pred_len']} (6 минут)")
    logging.info(f"  batch_size: {OPTIMIZED_CONFIG['batch_size']}")
    logging.info(f"  learning_rate: {OPTIMIZED_CONFIG['learning_rate']}")
    logging.info(f"  epochs: {OPTIMIZED_CONFIG['train_epochs']}")
    logging.info(f"Лог файл: {log_filename}")
    
    results = {}
    results_filename = f"prometheus_optimized_test_{timestamp}_results.json"
    
    for i, model_name in enumerate(MODELS_TO_TEST, 1):
        logging.info(f"\n📊 Прогресс: {i}/{len(MODELS_TO_TEST)}")
        logging.info(f"\n{'='*60}")
        logging.info(f"Тестирование модели: {model_name}")
        logging.info(f"{'='*60}")
        
        # Запускаем тестирование
        result = run_model_test(model_name, timeout_seconds=3600)  # 1 час таймаут
        results[model_name] = result
        
        # Сохраняем промежуточные результаты
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"💾 Промежуточные результаты сохранены в {results_filename}")
        
        # Пауза между экспериментами
        if i < len(MODELS_TO_TEST):
            logging.info("⏳ Пауза 10 секунд перед следующим экспериментом...")
            time.sleep(10)
    
    # Финальный отчет
    logging.info(f"\n{'='*60}")
    logging.info("📋 ФИНАЛЬНЫЙ ОТЧЕТ")
    logging.info(f"{'='*60}")
    
    successful_models = []
    failed_models = []
    
    for model_name, result in results.items():
        if result['success']:
            successful_models.append((model_name, result['duration']))
            logging.info(f"✅ {model_name}: {result['duration']:.1f}s")
        else:
            failed_models.append((model_name, result.get('error', 'Unknown error')))
            logging.info(f"❌ {model_name}: {result.get('error', 'Unknown error')}")
    
    logging.info(f"\n📊 Итого: {len(successful_models)} успешных, {len(failed_models)} неудачных")
    logging.info(f"💾 Полные результаты сохранены в: {results_filename}")
    logging.info(f"📝 Лог сохранен в: {log_filename}")
    
    if successful_models:
        # Сортируем по времени выполнения
        successful_models.sort(key=lambda x: x[1])
        logging.info(f"\n🏆 Самая быстрая модель: {successful_models[0][0]} ({successful_models[0][1]:.1f}s)")

if __name__ == "__main__":
    main() 