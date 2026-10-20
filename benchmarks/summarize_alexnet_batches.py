"""Render the completed AlexNet batch-size sweep as Markdown."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'benchmarks/alexnet_batches/results.json').read_text())
rows = sorted(data['results'], key=lambda r: r['batch_size'])
completed = [r for r in rows if r['status'] == 'complete']
best = min(completed, key=lambda r: r['median_total_seconds']) if completed else None
lines = [
    '# Сравнение размеров батча AlexNet', '',
    'Apple M1 Pro, 16 ГиБ RAM, MPS. Данные: скачанные с Kaggle PNG, '
    '40 000 train / 10 000 val, seed=52. Модель загружается из определения '
    'класса в AlexNet.ipynb; начальные веса воспроизводятся с seed=52. '
    'Adam lr=3e-4, свежий AutoAugment, четыре persistent workers, '
    'стандартизация train на MPS, готовый val на CPU. '
    'Каждый размер проверяется в отдельном новом процессе. '
    'Чтение PNG и подготовка статистик вынесены за измеряемую эпоху. '
    'После полной эпохи прогрева измеряются три полные эпохи train+val. '
    'MPS синхронизируется на границах фаз. Метрики извлекаются через '
    'item() с той же частотой, что в рабочем цикле. '
    'Отрисовка графиков и сохранение checkpoint исключены.', '',
    '## Результаты', '',
    '| Батч | Train, с | Val, с | Эпоха, с | Изображений train/с | Шагов train | MPS драйвер, ГиБ |',
    '|---:|---:|---:|---:|---:|---:|---:|',
]
for row in rows:
    if row['status'] == 'complete':
        lines.append(f"| {row['batch_size']} | {row['median_train_seconds']:.2f} | "
                     f"{row['median_val_seconds']:.2f} | {row['median_total_seconds']:.2f} | "
                     f"{row['median_train_images_per_second']:.0f} | {row['train_steps']} | "
                     f"{row['max_observed_driver_bytes']/2**30:.2f} |")
    else:
        lines.append(f"| {row['batch_size']} | — | — | {row['status']} | — | — | — |")
if best:
    lines += ['', f"Минимальная медиана полного времени эпохи: батч {best['batch_size']}, "
              f"{best['median_total_seconds']:.2f} с. Это выбор по скорости данного прогона; "
              'качество после сходимости не сравнивалось.']
by_batch = {r['batch_size']: r for r in completed}
if 256 in by_batch and 512 in by_batch:
    small, large = by_batch[256], by_batch[512]
    delta = (small['median_total_seconds']/large['median_total_seconds']-1)*100
    saving = (1-small['max_observed_driver_bytes']/large['max_observed_driver_bytes'])*100
    if abs(delta) < 3:
        lines += ['', 'Практическая рекомендация для этого Mac — батч **256**: '
                  f'время эпохи отличается от 512 всего на {delta:.2f}%, '
                  f'а максимальное наблюдаемое выделение драйвера MPS меньше на {saving:.1f}%. '
                  'Данные не показывают значимого выигрыша скорости у 512; '
                  'при этом меньший батч оставляет больше памяти системе.']
lines += ['', '## Разброс и состояние памяти', '',
          '| Батч | Диапазон трёх эпох, с | Min доступной RAM, ГиБ | Swap перед стартом, ГиБ | Max swap, ГиБ |',
          '|---:|---:|---:|---:|---:|']
for row in rows:
    epochs = row.get('epochs', [])
    interval = (f"{min(e['total_seconds'] for e in epochs):.2f}–"
                f"{max(e['total_seconds'] for e in epochs):.2f}") if epochs else '—'
    memory = row['system_memory']
    lines.append(f"| {row['batch_size']} | {interval} | "
                 f"{memory['min_available_bytes']/2**30:.2f} | "
                 f"{memory['swap_start_bytes']/2**30:.2f} | {memory['max_swap_bytes']/2**30:.2f} |")
lines += ['', 'Время — медианы отдельных фаз и полного времени; сумма медиан фаз '
          'может немного отличаться от медианы эпохи. Диапазоны не включают прогрев. '
          'Системная память измеряется раз в секунду и включает другие приложения.', '',
          'Память MPS — максимум наблюдений после forward и шага оптимизатора, '
          'а не гарантированный пик внутри каждого оператора. '
          'driver_allocated_memory включает кеши аллокатора и выделения MPS/MPSGraph; '
          'это не только живые тензоры. '
          '[Документация PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.mps.driver_allocated_memory.html).', '',
          'Батчи запускались последовательно в порядке 256, 128, 512, 1024. '
          'Другие пользовательские приложения и kernel не закрывались, их память '
          'не очищалась. Фоновая активность и изменение состояния памяти могут '
          'влиять на небольшие различия результатов. Один процесс мониторинга '
          'контролировал доступную RAM и swap; при нехватке памяти он мог завершить '
          'только собственный benchmark и его workers.', '']
for row in rows:
    if row['status'] != 'complete':
        lines += [f"Батч {row['batch_size']}: `{row['status']}`. " + row.get('error', ''), '']
lines += ['## Воспроизведение', '', '```bash',
          '.venv/bin/python -u benchmarks/alexnet_batch_size.py',
          '.venv/bin/python benchmarks/summarize_alexnet_batches.py', '```', '',
          'JSON и логи каждого процесса: `benchmarks/alexnet_batches/`. '
          'Запуск не изменяет notebook, файлы весов и пользовательские kernel. '
          'Нельзя одновременно с этим сравнением запускать другое обучение на GPU: '
          'это изменит измеряемую нагрузку.', '']
target = ROOT / 'docs/alexnet_batch_sizes.md'
target.write_text('\n'.join(lines))
print(target)
