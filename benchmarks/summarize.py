"""Build the Russian timing report from the completed benchmark JSON."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
r = json.loads((HERE / 'final_results.json').read_text())
rows = {v['name']: v for v in r['results']}
base = rows['baseline']['median_total_seconds']
names = {
    'baseline': 'Текущая схема: float32, AutoAugment и стандартизация каждой картинки',
    'baseline_val_precomputed': 'Текущая схема + заранее стандартизированный val',
    'uint8_sample_cpu': 'uint8 + AutoAugment и стандартизация каждой картинки на CPU',
    'uint8_batch_cpu': 'uint8 + AutoAugment, стандартизация батча на CPU',
    'uint8_batch_device': 'uint8 + AutoAugment, стандартизация батча на MPS',
    'uint8_workers2': 'То же + 2 persistent workers',
    'uint8_workers4': 'То же + 4 persistent workers',
    'uint8_device_metrics': 'Батч на MPS, 0 workers, метрики на устройстве',
    'uint8_workers2_metrics': 'Батч на MPS, 2 workers, метрики на устройстве',
    'uint8_workers4_metrics': 'Батч на MPS, 4 workers, метрики на устройстве',
    'cache_uint8_k1': 'Кеш: 1 AutoAugment на картинку, uint8, стандартизация на MPS',
    'cache_uint8_k4': 'Кеш: 4 AutoAugment на картинку, uint8, стандартизация на MPS',
    'cache_float32_k1': 'Кеш: 1 AutoAugment на картинку, готовый float32',
    'cache_float32_k4': 'Кеш: 4 AutoAugment на картинку, готовый float32',
    'simple_augmentation': 'RandomCrop + HorizontalFlip, 0 workers, батч на MPS',
    'no_aug_sample_cpu': 'Без аугментаций: стандартизация каждой картинки на CPU',
    'no_aug_prestandardized': 'Без аугментаций: заранее стандартизированные тензоры',
}
out = [
    '# Замеры подготовки данных LeNet', '',
    'Дата: 20 октября 2026. Измерено на ' + r['environment']['hardware'] +
    ', 16 ГиБ RAM, MPS, Python 3.13.14, PyTorch ' + r['environment']['torch'] +
    ', torchvision ' + r['environment']['torchvision'] + '.', '',
    'Данные: официальный CIFAR-10 train (50 000 изображений), случайное разбиение '
    '40 000 / 10 000, seed=52. В проекте не было исходной папки PNG; использована '
    'копия официального архива с [Zenodo](https://zenodo.org/records/10089977), '
    'MD5 c58f30108f718f92721af3b95e74349a, '
    'совпадающий с контрольной суммой torchvision. Порядок и разбиение могут '
    'отличаться от Kaggle PNG. Декодирование и чтение датасета исключены из времени эпохи, '
    'как и в исходном ноутбуке, где изображения сначала загружаются в RAM.', '',
    'Архитектура совпадает с LeNet5 в ноутбуке: 3→6→16→120, Tanh, AvgPool, '
    '120→84→10. Adam lr=5e-4, weight_decay=1e-4, batch_size=2048, '
    '20 обучающих и 5 валидационных батчей на эпоху. Для каждого варианта: '
    'одна полная эпоха прогрева, затем три измеряемые эпохи. '
    'Показана медиана; MPS синхронизируется на границах train и val. '
    'Варианты запускаются последовательно в фиксированном случайном порядке. '
    'Начальные веса одинаковы. Варианты без переноса метрик сохраняют порядок '
    'loss.item(), затем accuracy.item() из исходного цикла. '
    'Все варианты, кроме baseline, используют заранее стандартизированный val. '
    'Построение графиков и запись checkpoint исключены для всех вариантов.', '',
    'Рекомендуемая из проверенных схем: uint8 → свежий AutoAugment в четырёх '
    'persistent workers → сборка батча → MPS → float32 и стандартизация → LeNet. '
    f"Медиана {rows['uint8_workers4']['median_total_seconds']:.3f} с против {base:.3f} с "
    f"у текущей схемы ({base / rows['uint8_workers4']['median_total_seconds']:.2f}×). "
    'Кеширование не дало преимущества перед этой схемой, а разнообразие аугментаций '
    'при нём ограничено. Накопление метрик на MPS поверх четырёх workers '
    'заметно не ускорило эпоху. Отдельный перенос стандартизации батча с CPU '
    'на MPS без workers также не дал ускорения в этих замерах.', '',
    '## Время эпохи', '',
    '| Вариант | Train, с | Val, с | Всего, с | Диапазон трёх эпох, с | Ускорение |',
    '|---|---:|---:|---:|---:|---:|',
]
for key, title in names.items():
    if key not in rows:
        continue
    v = rows[key]
    times = [e['total_seconds'] for e in v['epochs']]
    out.append(f"| {title} | {v['median_train_seconds']:.3f} | {v['median_val_seconds']:.3f} | "
               f"{v['median_total_seconds']:.3f} | {min(times):.3f}–{max(times):.3f} | "
               f"{base / v['median_total_seconds']:.2f}× |")
out += ['', 'Ускорение — относительно baseline. Медианы train и val вычислены отдельно, '
        'поэтому их сумма может немного отличаться от медианы полного времени.', '',
        '## Цена кеширования', '',
        '| Кеш | Подготовка, с | Размер изображений, МБ (10⁶ байт) | Окупаемость против baseline, эпох |',
        '|---|---:|---:|---:|']
for key in ['cache_uint8_k1', 'cache_float32_k1', 'cache_uint8_k4', 'cache_float32_k4']:
    if key not in rows:
        continue
    v = rows[key]
    saving = base - v['median_total_seconds']
    epochs = math.ceil(v['setup_seconds'] / saving) if saving > 0 else None
    out.append(f"| {key} | {v['setup_seconds']:.3f} | {v['cache_bytes']/1e6:.2f} | "
               f"{epochs if epochs else 'нет'} |")
out += ['', 'Размер кеша учитывает изображения train; исходные тензоры, val, модель, '
        'оптимизатор и временные буферы требуют дополнительной памяти. '
        'Создание val заняло ' + f"{r['validation_precompute_seconds']:.3f}" + ' с. '
        'Подготовка кешей выполнялась на CPU, без сохранения на диск. '
        'Разовая подготовка исключена из времени эпохи. '
        'Окупаемость рассчитана как ceil(подготовка / экономия на эпохе) относительно baseline; '
        'она не доказывает преимущество кеша перед workers.', '',
        '## Что означает предварительная стандартизация', '',
        'Mean и std вычисляются один раз по исходному train. При отсутствии аугментаций '
        'весь train и val можно стандартизировать заранее. При свежем AutoAugment на '
        'каждой эпохе сначала выполняется аугментация uint8, затем стандартизация. '
        'Результат стандартизации имеет дробные и отрицательные значения, поэтому '
        'сохранять его в uint8 нельзя. Деление на 255 можно объединить со стандартизацией: '
        '`x.float() * (1 / (255 * std)) - mean / std`. '
        'Обработка батча не пересчитывает mean и std.', '',
        'Для кешей K=4 при каждом обращении выбирается случайный вариант одного '
        'исходного изображения. Эпоха сохраняет 40 000 примеров, а не увеличивается '
        'в четыре раза. Выбор вариантов и shuffle остаются онлайн-операциями. '
        'Кеш K=1 фиксирует единственное искажение каждой картинки. '
        'Кеши, простая аугментация и отсутствие аугментации меняют обучение; '
        'полный цикл сходимости и итоговая accuracy не сравнивались.', '',
        'Проверка значений: исходный float32→uint8→float32 путь и uint8 без обратного '
        'преобразования дали одинаковые результаты на 32 реальных изображениях при '
        'одинаковых seed AutoAugment (максимальная ошибка 0). '
        'Максимальная ошибка объединённой формулы стандартизации: '
        f"{r['normalization_max_abs_error']:.3g}" + '.', '',
        '## Воспроизведение', '',
        '```bash', '.venv/bin/python -u benchmarks/lenet_pipeline.py --download --output benchmarks/final_results.json',
        '.venv/bin/python benchmarks/summarize.py', '```', '',
        'Для своих PNG: добавьте `--images data/train --labels data/trainLabels.csv` '
        'вместо `--download`. Для отдельных вариантов: `--only baseline uint8_workers4`. '
        'На macOS запускать из среды с доступом к MPS. Если MPS недоступен, '
        'скрипт выберет CUDA или CPU и запишет устройство в JSON. '
        'Версии зависимостей — в requirements.txt. '
        'Исходный ноутбук и рабочие utils не изменялись.', '']
(HERE / 'REPORT.md').write_text('\n'.join(out))
print(HERE / 'REPORT.md')
