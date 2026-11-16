# Как повторять эксперименты TurkeyWorld

[README](../README_RU.md) · [English](REPRODUCING_EN.md) · [конфигурации опубликованного варианта](architecture-snapshot-2026-11-15.json)

## 1. Что именно воспроизводится

Здесь три разные задачи:

| Задача | Что требуется | Что получается |
|---|---|---|
| Новый baseline с нулевых весов | Репозиторий, игровые assets, новый Source-датасет, GPU | Новый кодек/SR/teacher-world через существующий `fresh_pipeline` |
| Повторение нынешней экспериментальной архитектуры | Baseline, последовательность миграций, новые datasets/plans для каждой стадии | Три RSSM, расширенные MoE, RF3 и RRDB с памятью фактур |
| Повторение опубликованных видео | Полные закреплённые world8873/SR7000/renderer checkpoints | То же семейство сцен и команды записи |

**В репозитории пока нет одной команды, которая с пустой папки автоматически воспроизводит всю позднюю историю RSSM/RF3/RRDB.** Есть рабочий исторический стартовый pipeline и отдельные проверяемые стадии продолжения. `ordered_image_pipeline`, motion-only, sprite-head и RRDB-continuation скрипты требуют родительские артефакты. Называть их готовым обучением нынешней системы с нуля было бы неверно.

Инструкция ниже даёт исполняемый старт без старых весов, порядок последующих стадий, точки обязательной адаптации и команды replay. Она не утверждает, что свежие веса получат те же ошибки или побитно совпадут с опубликованными.

## 2. Код, assets и окружение

Начинайте в отдельном clone/worktree и новой папке запуска, не в действующем `training_runs`. Сохраните `git rev-parse HEAD`, состояние `git diff`, версию Python/Torch/CUDA и драйвера. Git revision снимка указан в JSON; если на момент записи были незакоммиченные изменения, для точного повторения нужен и соответствующий diff.

Нужны `assets/turkey_atlas.png` и `assets/gun_atlas.png`. Их SHA есть в JSON. Нативные картинки входят в гибридный апскейлер: замена PNG меняет эксперимент даже при той же архитектуре. `training_runs`, datasets, `.pt` и `.npz` исключены из Git; обычный clone не скачивает checkpoints и исторические данные. Место на диске требуется и под данные, и под полные веса/Adam, и под snapshots; проверьте его до сбора.

Для Ubuntu24.04 с NVIDIA GPU:

```bash
bash scripts/setup_ubuntu.sh --dry-run
bash scripts/setup_ubuntu.sh
PY=.venv-ubuntu/bin/python
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
command -v ffmpeg
command -v ffprobe
```

Скрипт ожидает Python3.12 и устанавливает закреплённые Torch2.11/torchvision0.26 для CUDA12.8, а также зависимости проекта. Нужный NVIDIA driver устанавливается отдельно. Веса LPIPS могут скачиваться при первом использовании: это веса функции потерь, не готовая модель игры. CPU подходит для unit-тестов; длительные обучающие эксперименты рассчитаны на GPU. Установка среды не запускает worker.

`ffmpeg`/`ffprobe` должны быть в PATH: не предполагайте наличие `/usr/bin/ffmpeg`. На рабочей станции использовался бинарник из Anaconda. Для воспроизведения сохраните версию кодировщика.

## 3. Создание чистого Source-датасета

Все команды запускайте из корня проекта. Например, выберите ещё не существующий каталог:

```bash
RUN=training_runs/reproduce_clean_01
PY=.venv-ubuntu/bin/python
"$PY" -m turkeyworld.prepare_training_data "$RUN/data" \
  --rgb-train 4000 --rgb-val 256 --clip-frames 32 \
  --size 32 --hr-size 128 \
  --long-train 64 --long-val 16 --long-frames 450 \
  --hz 30 --seed 173 --workers 4
```

Это **пример нового baseline-датасета**, не точные исторические counts всех последующих фаз. 4000 коротких TRAIN-клипов нужны для существующего fresh-supervisor: он закрепляет subset128000 переходов, то есть4000×32. Пары HR должны быть настоящим Source128, а не увеличенным RGB32. Сохраняются карты, состояния, действия, RGB и маски; длинные эпизоды нужны для проверки rollout.

Проверьте `data/status.json`, завершённость manifest, engine `BEHAVIOR_VERSION`, state schema `single_enemy_v1` и отсутствие TRAIN/VAL overlap. Не меняйте seed, игровой код или assets внутри одного каталога с уже закреплённым manifest. Для поздних стадий дополнительно потребуются отдельные навигационные, quiet-enemy, боевые и статические наборы, включая ближнего/дальнего врага, пустые сцены, повороты, углы, стены, остановки и скольжение по одной оси. Их списки/квоты должны стать частью нового плана.

## 4. Исполняемый baseline с нуля

После подготовки данных:

```bash
bash scripts/run_training.sh --module turkeyworld.fresh_pipeline \
  --run-dir "$RUN/pipeline" --data-dir "$RUN/data" \
  --codec-run-dir "$RUN/codec" --sr-run-dir "$RUN/sr" \
  --world-run-dir "$RUN/world" --monitor-control-dir "$RUN/monitor" \
  --image-size 32 --device cuda --target-seconds 6 \
  --batch 4 --gradient-accumulation-steps 1 \
  --codec-batch 8 --sr-batch 8 --sr-steps 2000
```

Этот supervisor последовательно запускает VQGAN с нуля, исходный SR×4 и teacher-world/renderer с нуля. Стадия кодека имеет quality gate; `--max-steps 0` в дочерней команде означает ожидание критерия, а не ноль обучения. Поэтому завершение всего pipeline за определённое время не гарантировано. Не ослабляйте критерий, а затем не называйте результат прошедшим исходный gate.

**Этот baseline ещё не содержит нынешние три RSSM и RRDB wall-memory.** Его teacher нужен как начальный обученный графический/динамический источник для исторического маршрута миграций. Это важное различие между запуском с нуля и возобновлением уже развитой системы.

Команды дочерних фаз пишутся в `<pipeline>/*_command.json`; counters, stdout/stderr и checkpoints — в каталогах фаз. Читайте `--help` нужного модуля перед изменением флагов. Для локального монитора:

```bash
"$PY" -m training_dashboard.local_server --run-dir "$RUN/monitor" --port 8790
```

## 5. Переход от baseline к текущей архитектуре

Это **порядок реконструкции**, а не готовый shell-скрипт автоматической миграции. Каждая операция должна принимать свой новый полный checkpoint и создавать новый файл/план с SHA. Нельзя подставить номера8873/7000 в `training_steps`, чтобы пройти защиту продолжения.

| Порядок | Реализация | Обязательное действие |
|---|---|---|
| 1 | `rssm_migration.initialize_rssm_checkpoint` | Перенести обученный teacher в Player RSSM; сначала обучить/проверить новую динамику |
| 2 | `dual_rssm_migration.prepare_dual_rssm_continuation` | Добавить Enemy RSSM и продолжить обучение |
| 3 | `enemy_rssm_resize.prepare_enemy_rssm_resize` | Привести Enemy к h1344/hidden1408, если стартовая версия меньше |
| 4 | `rssm_fusion_migration.prepare_rssm_fusion_continuation` | Добавить общий d256/top-2 MoE |
| 5 | `rssm_combat_memory_migration.prepare_combat_memory_continuation` | Добавить Combat RSSM h1344/hidden1392 |
| 6 | `rssm_five_migration.prepare_rssm_five_continuation` | Добавить пять приватных ролей |
| 7 | `rssm_five_expanded_migration.prepare_expanded_five_continuation` | Расширить экспертов до1 005 088 параметров каждый |
| 8 | `rssm_spawn_conditioning_migration.prepare_spawn_conditioning_continuation` | Подключить текущую схему spawn conditioning |

Функции находятся в одноимённых файлах `turkeyworld/`; у них не у всех есть CLI. Проверяйте сигнатуры и их preconditions. Пример **только первой миграции** после получения совместимого teacher:

```python
from pathlib import Path
from turkeyworld.rssm_migration import initialize_rssm_checkpoint

initialize_rssm_checkpoint(
    Path("training_runs/reproduce_clean_01/world/checkpoints/teacher.pt"),
    Path("training_runs/reproduce_clean_01/rssm_initial.pt"),
    seed=173,
)
```

Файл teacher должен реально существовать и быть нужного формата; путь показан для baseline, это не скачивание pretrained-весов. Добавление случайно инициализированной ветви не заменяет её обучение. Некоторые миграции требуют сохранённые training ledgers/Adam определённой предыдущей фазы, а не просто `state_dict`. Если артефакт не удовлетворяет контракту, нужен явный адаптер нового эксперимента; отключать проверки и выдумывать ledger нельзя.

Дальнейшие trainer-стадии — `train_rssm_dynamics`, `train_rssm_moe`, `train_rssm_motion_only`. Для чистого нового опыта нужно создать свои планы датасетов/батчей и пройти их bind/activation checks. Текущая цель — H180/100, MAIN16=8 NAV+8 quiet NPC; прежние H16→…→180×30 и H48→112→180×40 — исторические режимы, а не требования к новому раунду. Полный проверенный target config дан в JSON, но сам JSON не содержит обученных весов или Adam.

## 6. Реконструкция нейрорендера

Соблюдайте зависимости: VQGAN encoder/quantizer/decoder → обученный teacher image model → distilled renderer/Base → графические пять ролей → expanded Base → RF3 → decoder overlay. Нужны пары Source с метками geometry/foreground и отдельный VAL.

Ключевые entrypoints: `distill_renderer`, `train_renderer_refine`, `train_base_geometry`, `train_decoder_rf_chain`, `ordered_image_pipeline`. Последний принимает `--base-run`, `--world`, `--source-recipe-parent`, `--decoder-parent`, `--sr-parent` и другие артефакты: он **не создаёт их из ничего**. Проверьте реальные CLI:

```bash
"$PY" -m turkeyworld.distill_renderer --help
"$PY" -m turkeyworld.train_base_geometry --help
"$PY" -m turkeyworld.train_decoder_rf_chain --help
"$PY" -m turkeyworld.ordered_image_pipeline --help
```

Не задавайте `--solver-steps 3` дистиллятору, если его CLI принимает только1/4. RF3 — отдельная последующая цепочка bundle, а не произвольная замена числа шагов у любой обученной сети. Заморозьте готовый bundle; проверьте его format/config/SHA и активный solver Euler3. При decoder adaptation сохраняйте encoder/quantizer и обновляйте только разрешённые префиксы. Номера32317/31709/4125 относятся к нашим артефактам, не являются целями нового run.

## 7. Пары и обучение экспериментального апскейлера

Сначала получите исходные Source32/128 пары и затем прогоните их через **свой замороженный нейрорендер**:

```bash
"$PY" -m turkeyworld.sr_neural_pairs_data \
  --output "$RUN/source_pairs" --source "$RUN/data" \
  --train-count 8192 --val-count 1024 --workers 4 --seed 173

"$PY" -m turkeyworld.cache_sr_neural_pairs \
  --source-root "$RUN/source_pairs" --output "$RUN/neural_pairs" \
  --world "$RUN/rssm_world.pt" --renderer "$RUN/renderer.pt" \
  --device cuda --batch 8 --seed 173
```

Вторая команда — **шаблон стадии после создания совместимых файлов** `rssm_world.pt`/`renderer.pt`; первая требует достаточный near/mid/far/guard inventory. Если ваш исходный сбор не покрывает страты, сначала дополните Source TRAIN/VAL отдельными непересекающимися условиями. Нельзя размножить один кадр, чтобы формально набрать нужный count. При использовании экспериментального неквалифицированного кодека opt-in должен быть явным (`--allow-unqualified-codec`) и результат должен оставаться помечен экспериментальным.

В новом опыте отдельно инициализируются RRDB RGB-веса и Adam, обучаются detection/box/phase/visibility головы врага. Затем подключаются weapon memories, depth/UV wall-memory и column-material/side heads. Встроенные assets сохраняются и проверяются по SHA. Нужны собственные панели VAL, а не обучение на скриншоте или выбранных видео.

**Граница автоматизации:** `train_rrdb_material_sr` в текущем дереве — экспериментальный trainer продолжения. Он читает `run-dir/assets/source.pt`, а путь к paired data и базовой VAL-панели пока зашит в коде (`image_chain_20261114/data/paired` и `sr_background_detail_20261114/training_plan.json`). `train_sprite_heads_sr` также опирается на исторические data/plan. Для чистого clone эти пути надо параметризовать и заново создать планы; документация не предлагает притвориться, что они уже существуют. Без этого команда RRDB не является законченной инструкцией обучения всей нынешней системы с нуля.

Сохраняйте установленные режимы: G LR2e−5, D5e−6, отдельные новые головы1e−4, GAN≤0.002 и контроль градиента на RGB-readout≤5%; отдельно проверяйте false green, depth, material accuracy и sprite dropouts. Source используется только для labels/loss, а не для runtime-подстановки рамок.

## 8. Политика и цикл

После проверки динамики обучается `train_dreamer_policy`. Сначала убедитесь, что TRAIN_DIRS и HEAD_EVENT_DIR указывают на **ваши** отдельные навигационные/quiet/combat datasets: текущие defaults исторические. Настройка только `--world-source` не заменяет эти данные. Используйте конечный `--updates`, фиксируйте seed и отдельный VAL.

`dreamer_motion_cycle` собирает Source-переходы политикой, обучает мир H180/100 и обновляет actor/critic. Его defaults тоже адресуют существующие каталоги. Для нового run явно задайте `--run-dir`, `--world-run`, `--policy-run`, `--motion-plan`, `--initial-world`. До запуска проверьте `--help` и подготовьте совместимые полные артефакты. Мир/рендер/SR заморожены во время actor/critic, а renderer/SR — во время motion-only обучения.

## 9. Проверка и запись после обучения

Перед игрой закрепите по одному полному snapshot world, renderer и SR. В таблице ниже — пример имён **ваших готовых** файлов:

```bash
"$PY" -m turkeyworld.compare_play \
  --world "$RUN/rssm_world.pt" --renderer-checkpoint "$RUN/renderer.pt" \
  --sr "$RUN/sr.pt" --rf-euler-steps 3 --device cuda \
  --display-size 1024 --smooth --initial-environment navigation
```

Для экспериментального bundle, требующего opt-in, добавляйте `--allow-unqualified-codec` осознанно. Enter начинает, T переключает среду, R сбрасывает обе ветки, Esc выходит. Проверяйте автономный режим, общую инициализацию, одинаковые controls, отсутствие engine corrections и последующей Source-обратной связи. Если вместо RSSM обучен только baseline teacher, не называйте его той же динамикой8873.

```bash
"$PY" -m turkeyworld.record_paired_cuts --profile navigation-six \
  --output-dir "$RUN/video_navigation" \
  --world "$RUN/rssm_world.pt" --renderer "$RUN/renderer.pt" --sr "$RUN/sr.pt" \
  --ffmpeg "$(command -v ffmpeg)"

"$PY" -m turkeyworld.record_quiet_chase --near --tracking \
  --output-dir "$RUN/video_enemy" \
  --world "$RUN/rssm_world.pt" --renderer "$RUN/renderer.pt" --sr "$RUN/sr.pt" \
  --ffmpeg "$(command -v ffmpeg)"
```

Для текущих форматов recorder включает экспериментальный codec opt-in. Quiet-chase пишет16 проб и выбирает6 полных фрагментов по опубликованному score, без Source state correction. Отбор не гарантирует отсутствие пропаданий: проверяйте `candidates.json`, `report.json`, полный decode, ffprobe, HP/events и все6 стартов. Каждая запись36с должна содержать1080 кадров30fps. Отбор и оставшиеся ограничения публикуйте вместе с видео.

## 10. Resume и критерии результата

Сохраняйте полный checkpoint: model/config, именной Adam и порядок групп, global step, Python/NumPy/Torch/CUDA RNG, private samplers, manifests, immutable plans и их SHA, историю миграций и image bundles. Для точного resume одного `state_dict` недостаточно. Не удаляйте stop markers автоматически; остановка должна пройти через завершённую границу оптимизатора.

Проверки перед принятием результата: неизменность TRAIN/VAL split и graphics lineage; воспроизводимая начальная сцена; XY-ошибка в клетках и yaw на H180; коллизии/застревание; направление/скорость NPC; HP/events/респаун; SSIM/MAE, texture/material/false-green и temporal flicker. Сравнивайте одинаковые VAL, H, действия и seed. После изменения функции потерь прежние TOTAL не являются сопоставимым качественным критерием.

Быстрые проверки кода без обучения:

```bash
PYTHONPATH=tests:. "$PY" -m unittest \
  test_rssm_first_contact test_rssm_motion_only_training \
  test_rrdb_material_sr test_wall_texture_memory_sr test_weapon_memory_sr \
  test_record_paired_cuts test_record_quiet_chase
```

Команды/сигнатуры этой инструкции сверены с кодом. Полный многодневный clean-room прогон позднего экспериментального стека при написании инструкции не выполнялся. Незавершённая автоматизация поздних стадий указана явно; это задача упаковки нового pipeline, а не основание обходить проверки исторических checkpoints.
