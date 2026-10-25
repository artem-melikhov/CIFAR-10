#!/usr/bin/env bash
# Run from the project directory: bash download_data.sh
# Requires Python 3.11+ with Kaggle CLI, unzip, and 7z/7zz.
set +x
set -Eeuo pipefail

die() {
    printf 'Ошибка: %s\n' "$*" >&2
    exit 1
}

[[ -n "${KAGGLE_API_TOKEN:-}" ]] || die \
    'Задайте переменную KAGGLE_API_TOKEN с API-токеном Kaggle (см. README.md).'
export KAGGLE_API_TOKEN

command -v unzip >/dev/null 2>&1 || die 'Не найден unzip. Ubuntu/Debian: sudo apt-get install unzip'
if command -v 7zz >/dev/null 2>&1; then
    seven_zip="$(command -v 7zz)"
elif command -v 7z >/dev/null 2>&1; then
    seven_zip="$(command -v 7z)"
else
    die 'Не найден 7z/7zz. Ubuntu/Debian: sudo apt-get install p7zip-full'
fi

if [[ -x "$PWD/.venv-kaggle/bin/kaggle" ]]; then
    kaggle_cli="$PWD/.venv-kaggle/bin/kaggle"
elif [[ -x "$PWD/.venv/bin/kaggle" ]]; then
    kaggle_cli="$PWD/.venv/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
    kaggle_cli="$(command -v kaggle)"
else
    die 'Не найден Kaggle CLI. Установка в отдельное окружение описана в README.md.'
fi

# The destination is relative to where the script is invoked, not its own path.
data_dir="$PWD/data"
archive="$data_dir/cifar-10.zip"
mkdir -p "$data_dir"

image_count() {
    if [[ -d "$1" ]]; then
        find "$1" -maxdepth 1 -type f -name '*.png' -print | wc -l | tr -d '[:space:]'
    else
        printf '0\n'
    fi
}

train_count="$(image_count "$data_dir/train")"
test_count="$(image_count "$data_dir/test")"
if [[ "$train_count" == 50000 && "$test_count" == 300000 && \
      -s "$data_dir/trainLabels.csv" && -s "$data_dir/sampleSubmission.csv" ]]; then
    printf 'Данные уже на месте: %s (train: %s, test: %s).\n' "$data_dir" "$train_count" "$test_count"
    exit 0
fi

if [[ ! -s "$archive" ]] || ! unzip -tq "$archive" >/dev/null 2>&1; then
    printf 'Загружаю соревнование cifar-10 в %s\n' "$data_dir"
    if ! "$kaggle_cli" competitions download cifar-10 --path "$data_dir" --force; then
        die 'Загрузка не удалась. Проверьте KAGGLE_API_TOKEN и принятие правил: https://www.kaggle.com/competitions/cifar-10/rules'
    fi
else
    printf 'Использую ранее скачанный архив: %s\n' "$archive"
fi

# Extract only the four known competition files. Archives are retained for reruns.
unzip -o "$archive" train.7z test.7z trainLabels.csv sampleSubmission.csv -d "$data_dir"

if [[ "$train_count" != 50000 ]]; then
    "$seven_zip" x -y -aoa "-o$data_dir" "$data_dir/train.7z"
fi
if [[ "$test_count" != 300000 ]]; then
    "$seven_zip" x -y -aoa "-o$data_dir" "$data_dir/test.7z"
fi

train_count="$(image_count "$data_dir/train")"
test_count="$(image_count "$data_dir/test")"
[[ "$train_count" == 50000 ]] || die "Ожидалось 50000 train PNG, найдено $train_count."
[[ "$test_count" == 300000 ]] || die "Ожидалось 300000 test PNG, найдено $test_count."
[[ -s "$data_dir/trainLabels.csv" && -s "$data_dir/sampleSubmission.csv" ]] || die 'Отсутствуют CSV-файлы.'

printf '\nГотово: %s\nTrain: %s PNG\nTest: %s PNG\n' "$data_dir" "$train_count" "$test_count"
