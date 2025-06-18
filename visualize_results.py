#!/usr/bin/env python3
"""
Скрипт для визуализации результатов предсказаний модели на данных из Prometheus.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from data_provider.prom_data_loader import fetch_frame
from data_provider.data_factory import data_provider
import torch
import os
import importlib
import argparse
import glob

from test_prometheus_models import MODELS_TO_TEST

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

TRANSFORMER_MODELS = [
    'Autoformer', 'TimesNet', 'FEDformer', 'Informer', 'PatchTST', 'Crossformer', 'Transformer', 'iTransformer'
]

def load_model_and_data(model_name='DLinear', model_path=None):
    """Загружает модель и данные для визуализации"""
    
    # Настройки модели (те же что в test_single_model.py)
    class Args:
        def __init__(self):
            self.task_name = 'long_term_forecast'
            self.model = model_name
            self.data = 'prometheus'
            self.root_path = './'
            self.data_path = ''  # Добавляем недостающий атрибут
            self.features = 'S'
            self.target = 'common_delayp90'
            self.freq = '15s'
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 2
            self.d_layers = 1
            self.d_ff = 2048
            self.dropout = 0.1
            self.embed = 'timeF'
            self.batch_size = 8
            self.use_gpu = False
            self.seasonal_patterns = 'Monthly'
            self.inverse = False
            # Добавляем другие недостающие атрибуты
            self.checkpoints = './checkpoints/long_term_forecast_DLinear_prometheus_test_DLinear_prometheus_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_quick_test_0/'
            self.timeenc = 0
            self.scale = True
            self.num_workers = 10
            self.itr = 1
            self.train_epochs = 3
            self.patience = 3
            self.learning_rate = 0.0001
            self.des = 'quick_test'
            self.loss = 'MSE'
            self.lradj = 'type1'
            self.use_amp = 0
            self.gpu = 0
            self.use_multi_gpu = 0
            self.devices = '0,1,2,3'
            self.p_hidden_dims = [128, 128]
            self.p_hidden_layers = 2
            # Параметры для разных моделей
            self.top_k = 5
            self.num_kernels = 6
            self.moving_avg = 25
            self.factor = 1
            self.distil = 1
            self.activation = 'gelu'
            self.expand = 2
            self.d_conv = 4
            self.fc_dropout = 0.1
            # Специфичные параметры для некоторых моделей
            if model_name == 'PatchTST':
                self.patch_len = 16
            if model_name == 'Crossformer':
                self.seg_len = 24
            if model_name == 'FEDformer':
                self.version = 'Fourier'
                self.mode_select = 'random'
                self.modes = 32
    args = Args()
    
    # Загружаем данные
    data_set, data_loader = data_provider(args, 'test')
    
    # Находим путь к сохраненной модели
    if model_path is None:
        model_path = f'./checkpoints/long_term_forecast_{model_name}_prometheus*/*checkpoint.pth'
        found = glob.glob(model_path)
        if not found:
            raise FileNotFoundError(f"Не найдена сохраненная модель по маске: {model_path}")
        model_path = found[0]
    
    # Динамический импорт модели
    try:
        model_module = importlib.import_module(f"models.{model_name}")
        ModelClass = getattr(model_module, "Model")
    except Exception as e:
        print(f"❌ Не удалось импортировать модель {model_name}: {e}")
        return None, None, None, None
    model = ModelClass(args)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, data_loader, data_set, args

def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mae, rmse, mape

def predict_and_visualize(model, data_loader, data_set, args, num_samples=5):
    """Делает предсказания и создает графики"""
    
    predictions = []
    actuals = []
    inputs = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            if i >= num_samples:  # Ограничиваем количество образцов
                break
                
            # Приводим к правильному типу данных
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            batch_y_mark = batch_y_mark.float()
            
            if hasattr(args, 'model') and args.model in TRANSFORMER_MODELS:
                dec_len = args.label_len + args.pred_len
                x_dec = torch.zeros(batch_y.shape[0], dec_len, batch_y.shape[2])
                x_dec[:, :args.label_len, :] = batch_y[:, :args.label_len, :]
                x_mark_dec = batch_y_mark[:, :dec_len, :]
            else:
                x_dec = None
                x_mark_dec = batch_y_mark
            
            # Предсказание
            outputs = model(batch_x, batch_x_mark, x_dec, x_mark_dec)
            
            # Сохраняем результаты (берем только pred_len последних значений)
            predictions.append(outputs[:, -args.pred_len:, :].detach().cpu().numpy())
            actuals.append(batch_y[:, -args.pred_len:, :].detach().cpu().numpy())
            inputs.append(batch_x.detach().cpu().numpy())
    
    # Объединяем батчи
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    inputs = np.concatenate(inputs, axis=0)
    
    # Создаем графики
    create_plots(inputs, predictions, actuals, args, num_samples, args.model)

def create_plots(inputs, predictions, actuals, args, num_samples, model_name):
    """Создает различные графики для анализа результатов"""
    
    # Загружаем исходные данные для денормализации
    df_raw = fetch_frame(
        use_cache=True,
        verbose=False)
    
    # Вычисляем параметры нормализации из тренировочных данных
    train_size = int(len(df_raw) * 0.7)
    train_data = df_raw['common_delayp90'].iloc[:train_size]
    mean_val = train_data.mean()
    std_val = train_data.std()
    
    def denormalize(data):
        """Денормализация данных"""
        return data * std_val + mean_val
    
    # 1. График временных рядов для нескольких образцов
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 3*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # Входная последовательность (денормализованная)
        input_seq = denormalize(inputs[i, :, 0])
        pred_seq = denormalize(predictions[i, :, 0])
        actual_seq = denormalize(actuals[i, :, 0])
        
        # Получаем реальные временные индексы из тестового набора
        import pandas as pd
        
        # Загружаем исходные данные для получения реальных дат
        df_raw = fetch_frame(use_cache=True, verbose=False)
        
        # Вычисляем границы тестового набора (последние 10%)
        total_len = len(df_raw)
        test_start_idx = int(total_len * 0.9)  # 90% для train+val, 10% для test
        
        # Получаем временные индексы тестового набора
        test_timestamps = df_raw.index[test_start_idx:]
        
        # Для каждого образца берем соответствующий кусок временных меток
        sample_start_idx = i * args.batch_size  # Примерное смещение для образца
        if sample_start_idx < len(test_timestamps):
            # Временные индексы для истории
            hist_start = max(0, sample_start_idx)
            hist_end = min(len(test_timestamps), hist_start + len(input_seq))
            if hist_end - hist_start >= len(input_seq):
                input_times = test_timestamps[hist_start:hist_end]
            else:
                # Если не хватает данных, генерируем от последней доступной даты
                last_time = test_timestamps[hist_start] if hist_start < len(test_timestamps) else test_timestamps[-1]
                input_times = pd.date_range(start=last_time, periods=len(input_seq), freq='15S')
            
            # Временные индексы для предсказаний
            pred_times = pd.date_range(start=input_times[-1] + pd.Timedelta(seconds=15), 
                                      periods=len(pred_seq), freq='15S')
        else:
            # Fallback: если образец выходит за границы, используем конец тестового набора
            last_time = test_timestamps[-1]
            input_times = pd.date_range(start=last_time - pd.Timedelta(seconds=15*(len(input_seq)-1)), 
                                       periods=len(input_seq), freq='15S')
            pred_times = pd.date_range(start=input_times[-1] + pd.Timedelta(seconds=15), 
                                      periods=len(pred_seq), freq='15S')
        
        # Рисуем графики
        ax.plot(input_times, input_seq, 'b-', label='История', linewidth=2)
        ax.plot(pred_times, pred_seq, 'r--', label='Предсказание', linewidth=2)
        ax.plot(pred_times, actual_seq, 'g-', label='Реальность', linewidth=2)
        
        # Вычисляем метрики на денормализованных данных
        mae, rmse, mape = calc_metrics(actual_seq, pred_seq)
        ax.set_title(f'Образец {i+1}: Предсказание vs Реальность (денормализовано)\nMAE={mae:.1f} мс, RMSE={rmse:.1f} мс, MAPE={mape:.2f}%')
        ax.set_xlabel('Время')
        ax.set_ylabel('common_delayp90 (мс)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Поворачиваем метки времени
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Создаем папку для графиков если её нет
    plots_dir = os.path.join('plots', model_name)
    os.makedirs(plots_dir, exist_ok=True)
    
    filename = f'{plots_dir}/{model_name}_prometheus_seq{args.seq_len}_pred{args.pred_len}_timeseries.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Scatter plot: предсказания vs реальные значения (денормализованные)
    plt.figure(figsize=(10, 8))
    pred_flat = denormalize(predictions.flatten())
    actual_flat = denormalize(actuals.flatten())
    
    plt.scatter(actual_flat, pred_flat, alpha=0.6, s=20)
    
    # Линия идеального предсказания
    min_val = min(actual_flat.min(), pred_flat.min())
    max_val = max(actual_flat.max(), pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное предсказание')
    
    plt.xlabel('Реальные значения (мс)')
    plt.ylabel('Предсказанные значения (мс)')
    # Вычисляем метрики на денормализованных данных
    mae_denorm, rmse_denorm, mape_denorm = calc_metrics(actual_flat, pred_flat)
    plt.title(f'Предсказания vs Реальные значения (денормализовано)\nMAE={mae_denorm:.1f} мс, RMSE={rmse_denorm:.1f} мс, MAPE={mape_denorm:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Добавляем метрики на график (в реальных единицах)
    mse = np.mean((pred_flat - actual_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - actual_flat))
    correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
    
    plt.text(0.05, 0.95, f'MSE: {mse:.1f} мс²\nMAE: {mae:.1f} мс\nCorrelation: {correlation:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    filename = f'{plots_dir}/{model_name}_prometheus_seq{args.seq_len}_pred{args.pred_len}_scatter.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Распределение ошибок (в реальных единицах)
    errors = pred_flat - actual_flat
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Гистограмма ошибок
    ax1.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Нулевая ошибка')
    ax1.set_xlabel('Ошибка предсказания (мс)')
    ax1.set_ylabel('Частота')
    ax1.set_title(f'Распределение ошибок (денормализовано)\nMAE={mae_denorm:.1f} мс, RMSE={rmse_denorm:.1f} мс, MAPE={mape_denorm:.2f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot ошибок
    ax2.boxplot(errors, vert=True)
    ax2.set_ylabel('Ошибка предсказания (мс)')
    ax2.set_title(f'Box Plot ошибок (денормализовано)\nMAE={mae_denorm:.1f} мс, RMSE={rmse_denorm:.1f} мс, MAPE={mape_denorm:.2f}%')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{plots_dir}/{model_name}_prometheus_seq{args.seq_len}_pred{args.pred_len}_errors.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Временной анализ исходных данных
    plt.figure(figsize=(15, 8))
    
    # Загружаем полные данные для анализа
    df = fetch_frame(use_cache=True, verbose=False)
    
    # Проверяем названия колонок и конвертируем время
    print("Колонки в данных:", df.columns.tolist())
    if 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    elif 'ts' in df.columns:
        df['datetime'] = pd.to_datetime(df.index)  # ts уже индекс
    else:
        # Используем индекс как время
        df['datetime'] = df.index
    df = df.set_index('datetime')
    
    # Ресемплируем для лучшей визуализации (каждые 5 минут)
    df_resampled = df['common_delayp90'].resample('5T').mean()
    
    plt.plot(df_resampled.index, df_resampled.values, linewidth=1)
    plt.title('Временной ряд common_delayp90 (усредненный по 5 минутам)')
    plt.xlabel('Время')
    plt.ylabel('common_delayp90 (мс)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    original_filename = f'{plots_dir}/prometheus_original_timeseries_90600pts.png'
    plt.savefig(original_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Создаем список сохраненных файлов
    timeseries_file = f'{plots_dir}/{model_name}_prometheus_seq{args.seq_len}_pred{args.pred_len}_timeseries.png'
    scatter_file = f'{plots_dir}/{model_name}_prometheus_seq{args.seq_len}_pred{args.pred_len}_scatter.png'
    errors_file = f'{plots_dir}/{model_name}_prometheus_seq{args.seq_len}_pred{args.pred_len}_errors.png'
    
    print("📊 Графики сохранены для {model_name} в {plots_dir}/")
    print(f"  - {timeseries_file} - Временные ряды предсказаний")
    print(f"  - {scatter_file} - Scatter plot предсказаний")
    print(f"  - {errors_file} - Анализ ошибок")
    print(f"  - {original_filename} - Исходный временной ряд")

def visualize_for_model(model_name, samples=5, model_path=None):
    model, data_loader, data_set, args = load_model_and_data(model_name, model_path)
    predict_and_visualize(model, data_loader, data_set, args, samples)

def batch_visualize(models, samples=5):
    for model_name in models:
        print(f"\n=== Визуализация для {model_name} ===")
        # Поиск чекпоинта
        pattern = f'./checkpoints/long_term_forecast_{model_name}_prometheus*/*checkpoint.pth'
        found = glob.glob(pattern)
        if not found:
            print(f"❌ Чекпоинт не найден для {model_name}")
            continue
        model_path = found[0]
        visualize_for_model(model_name, samples, model_path)

def main():
    parser = argparse.ArgumentParser(description='Визуализация результатов модели')
    parser.add_argument('--model', type=str, default=None, 
                       choices=MODELS_TO_TEST,
                       help='Название модели для визуализации')
    parser.add_argument('--samples', type=int, default=5,
                       help='Количество образцов для визуализации')
    parser.add_argument('--all', action='store_true', help='Визуализировать все модели')
    parser.add_argument('--models', nargs='+', default=None, help='Список моделей для визуализации')
    args = parser.parse_args()
    if args.all:
        batch_visualize(MODELS_TO_TEST, args.samples)
    elif args.models:
        batch_visualize(args.models, args.samples)
    elif args.model:
        visualize_for_model(args.model, args.samples)
    else:
        print('Укажи --model или --all или --models ...')

if __name__ == "__main__":
    main() 