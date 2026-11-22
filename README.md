# Распознавание овощей по фото (Трек A: нейросеть, transfer learning)

Тестовое задание «Junior Computer Vision / Machine Learning»: по фото определить, какой овощ на нём изображён: Capsicum (перец),
Carrot (морковь), Cucumber (огурец), Potato (картофель), Raddish (редис), Squash (кабачок). Датасет: Kaggle `preethi866/vegetable-dataset`.
Главный акцент сделан на честной оценке: в официальном разбиении данных обнаружилась серьёзная утечка, поэтому все метрики
посчитаны на собственном разбиении по группам.

* Отчёт (2 страницы): [`reports/report.md`](reports/report.md)
* Рабочий журнал со всеми цифрами и решениями: [`PROGRESS.md`](PROGRESS.md) (на английском)

## Главные результаты (модель `models/best.pt`, собственный тест по группам: 966 изображений, 42 группы)

| | собственное разбиение по группам | официальное разбиение Kaggle (переобучено с той же схемой) |
|---|---|---|
| accuracy | **93,2 %** (95 % ДИ 78–100 %, bootstrap по группам) | 100 % (тестовых файлов, идентичных train: 717 из 799) |
| macro-F1 | 0,926 | 1,000 |

Неверно классифицировано 66 из 966 изображений, 63 из них — в одном тестовом видео с перцем. На реальных фото из магазина
ожидается заметное падение точности (см. отчёт).

## Установка

Разработано на Python 3.12.5 под Windows (Python 3.10–3.12 должны работать; для torch 2.5.1 нет сборок под Python 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt   # одна команда, CPU-версия torch (дополнительный индекс прописан в файле)
```

Зафиксированные версии: torch 2.5.1 (CPU), torchvision 0.20.1, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, pillow 10.4.0,
imagehash 4.3.1, matplotlib 3.9.2, seaborn 0.13.2, tqdm 4.66.5, onnx 1.17.0, onnxruntime 1.20.1, grad-cam 1.5.4.
Для самого `predict.py` нужны только torch, torchvision, pillow (и onnxruntime, если используется `--onnx`).

## Предсказание (то, что запускается на вашем закрытом наборе)

```bash
python predict.py --input путь/к/папке --output preds.csv
```

* Формат результата: `filename,label,confidence`; label — один из 6 классов или `unknown`; confidence с 2 знаками после запятой.
* Только CPU, без интернета. Веса: `models/final.pt` (обучен на всех 3879 изображениях), а если его нет — `models/best.pt`.
* `unknown`: файл не читается (confidence `0.00`) либо изображение не похоже ни на один из 6 овощей
  (энергетическая оценка ниже порога из `models/unknown_config.json`). Отключается флагом `--no-unknown`.
* Остальные параметры: `--weights`, `--onnx models/final.onnx` (onnxruntime, быстрее примерно в 1,5–1,8 раза), `--batch-size`,
  `--threads`, `--workers`, `--recursive`.
* Скорость на машине разработки (16 логических ядер, ResNet18, 224 px): полный путь «файл → предсказание» около 18–28 мс на
  изображение с PyTorch и около 11 мс с ONNX Runtime (16 потоков); 55–118 мс / около 37 мс на 1 потоке. Время на этом
  общем ПК колебалось до 2 раз между запусками (см. `outputs/latency.csv`, `outputs/latency_onnx.csv`; сравнение
  ONNX и PyTorch измерено в одном запуске).

## Как воспроизвести всё с нуля

1. Скачайте датасет Kaggle (https://www.kaggle.com/datasets/preethi866/vegetable-dataset или командой
   `kaggle datasets download -d preethi866/vegetable-dataset` через Kaggle CLI) и распакуйте так, чтобы были папки
   `Dataset/Dataset/<класс>/` и `final_dataset/final_dataset/{train,val,test}/<класс>/`. Укажите коду, где лежат данные:
   `set MAGNUM_DATA_ROOT=D:\путь\к\archive` (PowerShell: `$env:MAGNUM_DATA_ROOT="D:\путь\к\archive"`).
   По умолчанию в `src/config.py` прописан мой локальный путь. В Colab передайте `--images-root /content/archive/Dataset/Dataset`
   в `04_train.py` и `05_eval.py`.
2. Запустите скрипты в таком порядке (время указано для 16-ядерного CPU, GPU не нужен):

```bash
python src/01_audit.py                         # аудит данных, разбор имён файлов               ~15 с
python src/02_duplicates.py                    # MD5 / pHash / эмбеддинги, утечка               ~2 мин (один раз скачает веса ResNet18 ImageNet)
python src/03_split.py                         # собственное разбиение по группам + проверки    ~1 мин
python src/04_train.py                         # ResNet18, 2 этапа -> models/best.pt            ~9 мин
python src/05_eval.py                          # метрики на тесте, errors.csv, матрица ошибок
python src/05_eval.py --make-official-split    # сравнение с официальным разбиением
python src/04_train.py --split-file splits/official_split.csv --out models/official.pt --tag _official   # ~33 мин
python src/05_eval.py --model models/official.pt --split-file splits/official_split.csv --tag _official
python src/06_gradcam.py                       # Grad-CAM + проверки зависимости от фона
python src/tools_error_sheet.py                # контактный лист из 66 ошибок
python src/07_predict_checks.py                # проверка predict.py: формат / совпадение / задержка на CPU
python src/08_get_ood.py                       # изображения для unknown (только для разработки, нужен интернет)
python src/08_unknown.py                       # порог для "unknown" -> models/unknown_config.json
python src/08_onnx.py                          # экспорт в ONNX, проверка совпадения, скорость
python src/04_train.py --final --out models/final.pt --tag _final     # все данные, итоговая модель  ~13 мин
python src/08_onnx.py --weights models/final.pt --out models/final.onnx
```

Зерно генератора фиксировано (42), но порядок потоков на CPU не детерминирован до бита, поэтому последние знаки метрик могут слегка
отличаться. Цифры в отчёте получены в одном запуске; исходные результаты лежат в `outputs/` и `splits/`.

## Структура репозитория

```
predict.py                  скрипт предсказания (в корне, как требовалось)
src/config.py, common.py    пути/константы, общий код модели и предобработки
src/01_audit.py ... 08_onnx.py   по одному скрипту на шаг (см. выше), tools_error_sheet.py
models/                     best.pt (отложенная оценка), final.pt (итоговая), *.onnx, unknown_config.json
splits/                     split.csv (своё разбиение), official_split.csv, folds.csv, units.csv
outputs/                    метрики, графики, списки ошибок, таблицы задержки, картинки Grad-CAM
reports/                    report.md
PROGRESS.md                 рабочий журнал (решения, находки, цифры)
```

`models/best.pt` — модель, на которой получены все приведённые метрики. `models/final.pt` обучена на всех изображениях
(train+val+test) по той же схеме, поэтому видела больше сессий съёмки; именно её `predict.py` использует по умолчанию. Её собственные
цифры на этих изображениях поэтому не являются отложенной оценкой. Порог для «unknown» подобран на `best.pt` и перенесён на `final.pt`.

## Известные ограничения (подробности в отчёте)

* Во всём датасете только 19 видео и 185 фотографий (204 независимые группы); на класс приходится 2–6 видео; один телефон,
  четыре дня съёмки; Raddish снят в одной сессии.
* В тесте всего 6 отложенных видео, поэтому у accuracy широкий доверительный интервал.
* Модель частично опирается на фон и освещение; в данных есть кадры без видимого овоща (шум в разметке).
* Детектор «unknown» лишь умеренный (AUROC 0,87 против других овощей; с `best.pt` помечается 55 % фото других овощей, с итоговой
  `final.pt` — 37 %); реальных фото из магазина у меня не было.
