# Эксперимент Base → SR без Flow Matching

Страница: http://127.0.0.1:8790/ . Рабочий каталог: `training_runs/sr_base_only_20261116`.

Цепочка: обученный компактный Base31709 → RGB32 → RRDB ×4 → материалы/UV/текстуры стен → отдельный патрульный спрайт врага → пушка и руки последним слоем → RGB128. RF-сэмплирование и VQGAN не выполняются. При строгой загрузке прежнего renderer bundle извлекаются Base и его частная geometry-opacity сеть6241 параметра; RF, RF-эксперты и кодек удаляются из зарегистрированных модулей. Конфигурация исходного обучения сохраняется как provenance, выполнение выбирается флагом `base_only_enabled`. Runtime-метаданные явно сообщают `flow_matching=false`, `codec=false`, `solver_steps=0`.

Апскейлер начинает с полного SR7000, включая RRDB, материалы стен, оригинальные texture memories, atlas спрайтов и weapon memories. Это независимая адаптация с новым Adam и локальным счётчиком0…2000; прежние checkpoints не переписываются. Материалы, швы, кромка, глубина, UV и реконструкция обучаются по прежним objective. Генератор LR2e-5; головы распознавания/видимости/положения/фазы врага адаптируются отдельно с LR1e-5. Изображения атласа и прочие замороженные sprite weights проверяются хешем. Подставляются только патрульные фазы; навигационный режим запрещает врага, собственное предсказанное состояние может наложить veto. Пушка и руки композятся после врага.

Wall PatchGAN сохранён: D LR5e-6, GAN вес0 до500-го шага, затем линейный рост до0.002 за500 шагов, вклад adversarial-gradient ограничен5% от reconstruction-gradient на контрольном слое. Это не обещание устранения blur: качество границ и материалов проверяется по VAL и превью.

Входы заново закэшированы из Base:8192 TRAIN +1024 VAL. Source128, маски, state/map и материалы служат эталонами SR; SR на исполнении получает только RGB32. Для подготовки Base входов используются записанные Source pose/map; динамические appearance features вычисляются только из реального предыдущего RGB/state/action, с записанным navigation/combat профилем. Это paired renderer/SR experiment, не автономная оценка дрейфа динамики. Замороженный feature-world взят из существующего image_chain; обучение динамики и политики не возобновляется.

Превью и VAL каждые100 обновлений, checkpoint каждые25. Четыре ряда: Source128 / прямой Base32 / прежний SR на Base / адаптированный SR. Заголовок страницы явно сообщает отсутствие Flow. Первое превью шага0 — baseline, не результат дообучения.

## Повторный запуск и продолжение

Из корня проекта, при наличии существующих родительских assets и Source pairs:

```bash
.venv-ubuntu/bin/python -m turkeyworld.run_base_only_sr \
  --run-dir training_runs/sr_base_only_20261116 \
  --world training_runs/image_chain_20261114/assets/world.pt \
  --renderer training_runs/after_round_game_20261115T182837Z/assets/renderer.pt \
  --sr-source training_runs/sr_rrdb_wall_columns_20261115/checkpoints/sr_last.pt \
  --source-pairs training_runs/sr_semantic_npr_20261112/data/source \
  --steps 2000
```

Повторный запуск проверяет существующий cache и продолжает SR checkpoint, optimizer, sampler и RNG. Перед продолжением удалить `stop_training`, если остановка была запрошена; одновременно второй worker запускать нельзя. Родительские assets закреплены hard links, manifest и checksums.

Страница отдельно:

```bash
.venv-ubuntu/bin/python -m training_dashboard.local_server \
  --run-dir training_runs/sr_base_only_20261116 --port 8790
```

Проверка новой цепочки без окна:

```bash
.venv-ubuntu/bin/python -m turkeyworld.neural_play \
  --world training_runs/sr_base_only_20261116/assets/world.pt \
  --renderer-checkpoint training_runs/sr_base_only_20261116/assets/renderer.pt \
  --allow-unqualified-codec --base-only \
  --sr training_runs/sr_base_only_20261116/checkpoints/sr_last.pt \
  --device cuda --headless --steps 2 --warmup 0 --threads 2
```

Для интерактивного сравнения использовать `turkeyworld.compare_play` с теми же assets, `--base-only --display-size 1024 --smooth`. Не передавать `--rf-euler-steps`: в этой ветке RF solver отсутствует. Старый renderer bundle нужен для строгого извлечения связанного Base/геометрии, его flow/codec не исполняются. Существующая игровая ветка и прежние previews динамики автоматически не переключаются.

## Коррекция пропадающего врага

Текущая страница через `active_run.json` показывает `sr_base_patrol_grouped_20261116`, checkpoint2987. После Base SR1987 выполнено1000 обновлений sprite heads на Base TRAIN, затем включён явный `atlas_patrol_grouped` с проверкой на прежнем VAL без дополнительных SGD. Он сначала выбирает patrol/боевую группу по суммарной вероятности, затем позу внутри группы. Порог видимости0.9 и запрет боевых/навигационных спрайтов сохранены. Для исполнения нового варианта использовать `training_runs/sr_base_patrol_grouped_20261116/checkpoints/sr_last.pt`. Исправлен средний враг из исходного превью; дальний всё ещё может пропадать при низкой уверенности. Все7 поз пока распознаются неточно; см. журналH143 и validation.json.
