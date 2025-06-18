#!/usr/bin/env python3
"""
Быстрое тестирование одной модели с оптимизированными параметрами
"""

import subprocess
import sys
import time
from datetime import datetime

# Оптимизированные параметры для высокочастотных данных
OPTIMIZED_PARAMS = [
    '--task_name', 'long_term_forecast',
    '--is_training', '1',
    '--data', 'prometheus',
    '--root_path', './',
    '--data_path', '',
    '--features', 'S',
    '--target', 'common_delayp90',
    '--freq', '15s',
    '--checkpoints', './checkpoints/',
    
    # 🎯 ОПТИМИЗИРОВАННЫЕ параметры
    '--seq_len', '48',        # 12 минут истории
    '--label_len', '24',      # 6 минут перекрытия  
    '--pred_len', '24',       # 6 минут предсказания
    
    # Архитектура - уменьшенная
    '--enc_in', '1',
    '--dec_in', '1', 
    '--c_out', '1',
    '--d_model', '256',       # Уменьшено с 768
    '--n_heads', '4',         # Уменьшено с 8
    '--e_layers', '2',        # Уменьшено с 3
    '--d_layers', '1',        # Уменьшено с 2
    '--d_ff', '1024',         # Уменьшено с 3072
    '--dropout', '0.1',
    '--activation', 'gelu',
    '--embed', 'timeF',
    
    # Обучение - более агрессивное
    '--batch_size', '64',     # Увеличено с 32
    '--learning_rate', '1e-4', # Увеличено с 5e-5
    '--train_epochs', '30',   # Уменьшено для быстрого теста
    '--patience', '8',        # Уменьшено
    '--lradj', 'cosine',
    
    # GPU
    '--gpu', '0',
    '--itr', '1',
    '--des', 'prometheus_quick_test',
    
    # Предобработка
    '--moving_avg', '11',
    '--decomp_method', 'moving_avg',
    '--use_norm', '1',
    '--factor', '5',
    
    # Аугментация
    '--augmentation_ratio', '1',
    '--jitter',
    '--scaling', 
    '--seed', '2021',
    '--use_gpu', 'True',
    '--expand', '2',
    '--d_conv', '4'
]

def main():
    # Определяем модель
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'DLinear'
    
    print(f"🚀 Быстрое тестирование модели: {model_name}")
    print(f"⚡ Оптимизированные параметры:")
    print(f"   seq_len=48 (12 мин), pred_len=24 (6 мин)")
    print(f"   batch_size=64, lr=1e-4, epochs=30")
    print(f"   d_model=256, n_heads=4")
    
    # Формируем команду
    cmd = ['python', 'run.py'] + OPTIMIZED_PARAMS + [
        '--model', model_name,
        '--model_id', f'{model_name}_prometheus_quick'
    ]
    
    # Добавляем специфичные параметры
    if model_name == 'PatchTST':
        cmd.extend(['--patch_len', '8'])
    elif model_name == 'iTransformer':
        cmd.extend(['--channel_independence', '1'])
    elif model_name == 'Informer':
        cmd.extend(['--factor', '5'])
    
    print(f"\n🔧 Команда: {' '.join(cmd)}")
    print(f"⏰ Начало: {datetime.now().strftime('%H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Запускаем обучение
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        duration = time.time() - start_time
        print(f"\n✅ Успешно завершено за {duration:.1f} секунд!")
        
        # Показываем последние строки с результатами
        lines = result.stdout.strip().split('\n')
        print(f"\n📊 Результаты:")
        for line in lines[-10:]:
            if any(keyword in line.lower() for keyword in ['mse:', 'mae:', 'test loss:', 'epoch:']):
                print(f"   {line}")
                
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ Ошибка после {duration:.1f} секунд")
        print(f"Код возврата: {e.returncode}")
        if e.stderr:
            print(f"Ошибка: {e.stderr}")
        if e.stdout:
            print(f"Вывод: {e.stdout[-500:]}")  # Последние 500 символов
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Прервано пользователем")
        
if __name__ == "__main__":
    main() 