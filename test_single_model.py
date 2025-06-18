#!/usr/bin/env python3
"""
Простой скрипт для быстрого тестирования одной модели на данных из Prometheus.
Использование: python test_single_model.py [model_name]
"""

import sys
import subprocess

def test_model(model_name='TimesNet'):
    """Тестирование одной модели"""
    
    cmd = [
        'python', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--model_id', f'{model_name}_prometheus_test',
        '--model', model_name,
        '--data', 'prometheus',
        '--root_path', './',
        '--features', 'S',
        '--target', 'common_delayp90',
        '--freq', '15s',
        '--seq_len', '96',
        '--label_len', '48', 
        '--pred_len', '96',
        '--enc_in', '1',
        '--dec_in', '1',
        '--c_out', '1',
        '--d_model', '512',
        '--n_heads', '8',
        '--e_layers', '2',
        '--d_layers', '1',
        '--d_ff', '2048',
        '--dropout', '0.1',
        '--embed', 'timeF',
        '--batch_size', '8',  # Уменьшил для экономии памяти
        '--learning_rate', '0.0001',
        '--train_epochs', '3',  # Меньше эпох для быстрого тестирования
        '--patience', '3',
        '--des', 'quick_test',
        '--gpu', '0'  # Используем свободный GPU 5
    ]
    
    print(f"🚀 Тестирование модели: {model_name}")
    print(f"Команда: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Модель {model_name} успешно протестирована!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при тестировании модели {model_name}: {e}")
        return False

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else 'TimesNet'
    test_model(model) 