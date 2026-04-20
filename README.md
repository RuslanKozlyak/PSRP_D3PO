# PSRP MTPPO

Чистая реализация статьи "Enhanced multi-task deep reinforcement learning for the integrated inventory-routing problem under VMI mode".

Что включено:

- среда IRP-VMI c одним поставщиком, одним транспортом и многократными рейсами;
- архитектура `MTPPO` с `GIN`, multi-head attention, двумя актерами и общим критиком;
- обучение и оценка для joint-режима;
- эвристические baseline-политики для сравнения;
- `ipynb`-ноутбук для запуска экспериментов;
- smoke-тесты среды, моделей и обучения.

Ключевые параметры по статье:

- `gamma = 0.9`
- `learning_rate = 1e-3`
- `batch_size = 256`
- `GIN layers = 3`
- `GIN dims = (64, 128, 128)`
- `MLP dims = (128, 128)`
- `value clip = 0.1`
- `initial KL coefficient = 0.2`
- `attention heads = 8` по лучшему результату sensitivity analysis

Пара деталей в статье описаны неполно, поэтому в коде зафиксированы воспроизводимые инженерные допущения:

- history window для состояния пополнения = 7 дней;
- routing attention реализован как masked transformer decoder/scorer поверх GIN-эмбеддингов;
- для baseline сравнения добавлены воспроизводимые эвристики, потому что полный код GA/DDPG/A3C из статьи не опубликован.
