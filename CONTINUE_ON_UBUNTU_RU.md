# Продолжение TurkeyWorld на Ubuntu

Обучение приостановлено по запросу пользователя для сохранения и переноса.
**Автоматически возобновлять обучение после копирования нельзя.** Сначала нужно
проверить сохранённые границы, перенести привязки путей и получить явный запрос
на продолжение. Затем отдельно разрешена адаптация кода под Windows/Ubuntu;
она выполняется без обучения. Проверка на настоящей Ubuntu GPU пока не
подтверждена; её нельзя подменять тестами платформенных веток на Windows.
Код путей, bootstrap, управления процессами и защиты от сна теперь имеет
ветви Windows/Linux; далее приведены команды для проверки и явного продолжения.

## Что переносить

Перенести **весь workspace `E:\nau\TurkeyWorld`**, включая скрытые файлы,
неотслеживаемые исходники, `assets/`, `tests/`, `turkeyworld/`, обе страницы,
`training_dashboard/`, журналы и **весь `training_runs/`**. Обычный `git clone`
или только commit этого не восстановит: `.gitignore` исключает training runs,
NPZ-данные, PT-веса, локальные инструменты и временные артефакты; часть текущего
кода ещё не закоммичена. У `training_dashboard` отдельный Git-репозиторий.

Нужно сохранить не только последние model weights, но и полные checkpoints,
optimizer/Adam, RNG, планы, Source receipts, snapshots, validation panels,
исходные данные, checkpoints до миграций и всю цепочку parent/provenance.
Исходный workspace и его байты оставить резервной копией. При копировании
сохранять timestamps и по возможности hardlinks: VAL identity учитывает
`mtime_ns`, размер и содержимое. Изменение точности времени может изменить
identity даже при одинаковых NPZ-байтах.

Пример копирования **с уже доступной Linux/WSL-копии источника**, без удаления
чего-либо в источнике или назначения:

```bash
rsync -aH --info=progress2 /path/to/source/TurkeyWorld/ /path/to/destination/TurkeyWorld/
```

`-a` сохраняет в том числе время изменения, `-H` — hardlinks; всё равно сверить
SHA и `mtime_ns` после передачи. Обычный ZIP не гарантирует нужную точность
timestamps. См. [официальный rsync manual](https://download.samba.org/pub/rsync/rsync.1).

Сохранить снимок Python-окружения и локальных зависимостей. Windows Miniconda,
`.venv` и `.exe` не становятся Linux-окружением простым копированием: сохранить
их metadata/freeze, а исполняемые зависимости установить заново для Linux.
`.tools/node-v24.22.2-win-x64` — Windows Node; для UI-тестов на Ubuntu нужен
Linux Node. Данные и исходники внутри локальных инструментов не терять.

## Финальная сохранённая граница

Подтверждённая штатная остановка: **world8111**, внешний **round_index6**
(седьмой раунд), **158/180 updates раунда**, **H180/этап8/30**. Политика
завершена на **learner620 /actor400**. Контроллер54972 и worker мира остановлены.
Последняя фаза — `world_training`, а не новый сбор Source или обучение политики.

Арифметика:8111 =6873 исходных +1238 новых world updates; это6 полных раундов
по180 +158 текущего. **Осталось22 updates → world8133.** Продолжить нужно
именно H180/8, с текущими sampler/RNG, тем же Source-корпусом раунда6 и всеми
Adam moments. Не сбрасывать на H16, не повторять180 updates и не собирать новый
Source перед завершением оставшихся22.

Только после world8133 контроллер должен подключить этот завершённый мир
политике, выполнить20 head updates и50 actor/critic updates: будущие цели
**learner690 /actor450**. Это план следующей фазы, **не выполненный результат**.
Следующий Source-сбор идёт уже после неё.

Финальная квитанция:
[pause_receipt.json](training_runs/ubuntu_handoff_20261115/pause_receipt.json).
Все15 проверок прошли: полные часы world,1599 model tensors/442 Adam states/
7 групп, CPU/CUDA/PCG/private samplers, policy620/400 со slow critic/normalizers
и тремя оптимизаторами, хеши снимков и завершение всех обучающих процессов.
Read-only dashboards также остановлены. Перезапуска обучения не было.

Замороженные копии находятся в `training_runs/ubuntu_handoff_20261115/checkpoints`:

| Снимок | SHA256 |
| --- | --- |
| `rssm_world_step_008111.pt` | `f43d053ea2854bec13e6d7fa8458c964a94ec240e0aa99df258a08b3d710314a` |
| `policy_step_000620_actor_000400.pt` | `de8261d9a0ed153f6f59374feabf39fe87ae46303ed3c4b1660155119523a328` |
| `renderer_RF3_Base31709_decoder4125.pt` | `40f9373125c038a200b4386c25a375fa2c32d6902447a9f8b2b8cfa1e70985b6` |
| `sr_frozen_step_003000.pt` | `c7912165e80d555f9b09c521a3d0b6c46ac3915d1ff879fbdd35d815385dc388` |

[source_manifest.json](training_runs/ubuntu_handoff_20261115/source_manifest.json)
содержит исходные пути, размеры и SHA. Копии — hardlink snapshots внутри
исходного Windows workspace; при переносе проверить bytes/SHA повторно.
Сохранены7 Source receipts по32 эпизода: **224 эпизода /40320 переходов**.
Более ранние H112–H114 не подменяют эту границу. До проверки квитанции
не запускать SGD и не удалять stop markers.

Минимально необходимые группы файлов, помимо всего исходного workspace:

- Полный `training_runs/rssm_moe_nav15_enemy25_cap20_20261113`: текущий FULL
  checkpoint, исходный6873, планы/receipts, gate/samplers, VAL pins и preview-chain.
- Полный `training_runs/dreamer_actor_critic_20261114`: policy checkpoint с
  тремя оптимизаторами/RNG, slow critic, return/reward normalizers,
  world snapshots, TRAIN/VAL plans и validation panel. Model-only export
  не заменяет этот checkpoint.
- Полный `training_runs/dreamer_motion_cycle_20261114`: состояние контроллера,
  корпуса Source циклов0…6, collection plans/receipts/assets и все handoffs.
- Исходные `staged_single_20261110/data_navigation_recovery_v1/long/rgb/{train,val}`,
  `rssm_enemy_motion_20261113/data/full_game/rgb/{train,val}` и
  `combat_source_20261111/data` с policy replay, moving1 TRAIN/VAL-смертями
  и исходными validation snapshots/manifests.
- Финальные `image_chain_20261114/handoffs/decoder/renderer.pt` и frozen SR3000,
  а также **все старые preview parents и их assets**. Resolver проверяет25
  уникальных ссылок цепочки, включая прежние renderer/Atlas SR; эти старые файлы
  нужны для происхождения/identity, хотя выбранный современный SR не использует
  Atlas. Копии только последних RF3/SR недостаточно.
- Игнорируемая Git папка `training_runs/diagnostic_tools`, включая bootstrap,
  proof/recovery helpers, и все новые тесты/исходники. `code_snapshot.zip` —
  снимок кода, а не замена многогигабайтным данным/checkpoints.

## Действующая конфигурация, которую нужно сохранить

- Мир: три RSSM игрока/врага/боя, общий MoE trunk и пять Transformer-experts;
  state20, один постоянный NPC. Архитектуру и размеры не менять.
- Динамика: `rssm_motion_only_main16_v1`, **MAIN16 =8 навигации без врага
  +8 спокойного движения NPC**, накопление1. Combat loss и боевые MAIN/AUX/late
  выключены. Независимых contact actors нет; короткие ветви используют
  собственные MAIN-префиксы и полные h/z. Общая память боя сохранена и может
  получать motion-градиенты; чисто боевые readouts заморожены.
- Горизонты **[16,48,80,112,144,180]**, ровно **30 полных Adam updates на H**,
  **180 updates мира на раунд**. Полный BPTT текущего H; H180=6 секунд при30 Гц.
  Порога loss/серии нет. Не начинать текущий раунд или этап с нуля при переносе.
  Начальный наблюдаемый контекст1; FP32, TF32 выключен, CPU/OMP/MKL threads2.
- Последовательный цикл `dreamer_motion_Source_world_policy_cycle_v1`:
  Source → мир → политика. Source:16 NAV+16 quiet эпизодов,180 переходов каждый,
  всего32/5760; выстрел замаскирован, ε=0,30. Новые данные только TRAIN.
  Мир в каждой8-actor группе берёт4 исходных+4 новейших Source-примера.
- После законченного раунда мира политика подключает его полный checkpoint,
  выполняет20 reward/continue head updates и50 настоящих actor/critic updates.
  Policy MAIN18 остаётся6 NAV/6 quiet/6 combat; NAV и quiet смешивают3 новых
  +3 исходных старта. Воображаемый горизонт15 prior-переходов, мир заморожен.
  Не повторять уже выполненные actor updates или калибровку при resume.
- Source использует настоящий движок с текущими наблюдениями; будущие Source
  не исправляют воображаемые/автономные прогоны. Policy collector не заменяет
  ручное управление пользовательской игрой.
- Графика заморожена: **RF3/NPR32317, Base31709, VQGAN decoder4125,
  background-detail SR3000**. Native32 → нейроапскейлер128 → bilinear512.
  Восстановитель и Atlas в этом выбранном SR не применяются. Image-workers
  не возобновлять вместе с динамикой автоматически.
- Held-out VAL180 и её byte/mtime/panel identities сохранить. Старые VAL450,
  MAIN18 loss gates и архивные counters не пересчитывать по новой формуле.
  Завершение бюджетов и маленький imagined loss не означают quality-pass игры.

## Проверенное исходное окружение Windows

Снимок metadata перед переносом:

| Компонент | Версия |
| --- | --- |
| Python | 3.12.11, Anaconda, Windows x64 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA runtime / GPU | 12.8 / одна NVIDIA GeForce RTX 4090 |
| cuDNN reported version | 91900 |
| NumPy | 2.5.2 |
| Pillow | 12.3.0 |
| pygame | 2.6.1 |
| LPIPS | 0.1.4 |

Фактические снимки окружения:
[torch_environment.json](training_runs/ubuntu_handoff_20261115/torch_environment.json),
[installed_packages_windows_reference.json](training_runs/ubuntu_handoff_20261115/installed_packages_windows_reference.json).
Полный список пакетов Windows служит справочником; Windows-only пакеты нельзя
слепо устанавливать как Linux lockfile.

`requirements.txt` задаёт numpy≥1.26/Pillow≥10/pygame≥2.5;
`requirements-ml.txt` добавляет torch≥2.3; `requirements-vqgan.txt` —
torchvision≥0.18 и lpips==0.1.4. Это минимальные зависимости, **не точный lock**.
Для Linux официально опубликована пара **torch2.11.0/torchvision0.26.0 с CUDA12.8**.
Установщик проекта выбирает соответствующие Linux wheels; Windows conda не
переиспользуется. Драйвер и настоящий GPU Ubuntu ещё нужно проверить.
[Официальная таблица PyTorch](https://pytorch.org/get-started/previous-versions/#v2110)
подтверждает версии и index CUDA12.8; не заменять их случайным latest.
В policy/SR pickle встречается `numpy._core.multiarray`, поэтому NumPy2.x
сохраняется. Будущая численная эквивалентность всё равно требует проверки.

## Установка Ubuntu-окружения без запуска обучения

Для Ubuntu24.04 с Python3.12:

```bash
sudo apt-get update
sudo apt-get install python3.12-venv python3-pip rsync tmux libgl1 libsdl2-2.0-0
cd /path/to/TurkeyWorld
bash scripts/setup_ubuntu.sh --dry-run
bash scripts/setup_ubuntu.sh
```

[python3.12-venv](https://packages.ubuntu.com/noble/python3.12-venv),
[SDL2](https://packages.ubuntu.com/noble/libsdl2-2.0-0) и
[libgl1](https://packages.ubuntu.com/noble/libgl1) — пакеты официального Ubuntu archive.
`scripts/setup_ubuntu.sh` создаёт `.venv-ubuntu`, ставит указанную пару Torch/
torchvision из официального cu128 index и версии приложения из
`requirements-ubuntu.txt`, затем выполняет `pip check` и выводит состояние CUDA.
Он не запускает worker, не читает checkpoints и не удаляет stop markers.
Путь к существующему Python3.12 задаётся через `--python`; Windows venv отвергается.

CPU-окружение только для тестов: `bash scripts/setup_ubuntu.sh --cpu`; по умолчанию
оно отдельное, `.venv-ubuntu-cpu`. GPU-обучение в нём не запускать. Linux Node
понадобится отдельно для тестов dashboard; копия `node.exe` не подходит.
Проверить `nvidia-smi`, место на диске и фактический `torch.cuda.is_available()`.
Замеры Windows H180≈4,51GiB peak allocation не гарантируют вместимость/скорость
другого GPU. Здесь проверены Bash-синтаксис и dry-run установщика на Windows;
настоящая установка пакетов/Ubuntu GPU ещё не выполнялась.

## Проверка перенесённого workspace без обучения

В код добавлены `turkeyworld.workspace_paths`, кроссплатформенный
`turkeyworld.bootstrap` и `turkeyworld.portability_check`.
`workspace_portability.json` сохраняет **исходный identity-root**
`E:\nau\TurkeyWorld`; менять его на новый Linux-путь не нужно. Native I/O
разрешает пути внутри фактической папки проекта, а прежние JSON/планы сохраняют
свои исходные строки и SHA. Обрабатываются и относительные Windows `\`.

После полного копирования, установки окружения и без удаления stop markers:

```bash
cd /path/to/TurkeyWorld
.venv-ubuntu/bin/python -m turkeyworld.bootstrap \
  --module turkeyworld.portability_check --resume --snapshots
```

Это read-only preflight: он не создаёт optimizer, не делает SGD и не запускает
worker. `--snapshots` читает SHA исходных четырёх assets и четырёх копий;
`--resume` проверяет clocks, state/Adam/RNG, gate, Source-корпуса, VAL и всю
preview ancestry. На исходном Windows workspace фактически прошли **8 SHA-checks
+22 resume-checks** для8111/620. Точный состав и результаты — в
[portability_preflight_windows.json](training_runs/ubuntu_handoff_20261115/portability_preflight_windows.json);
исходные JSON/планы остались прежними.
На целевой Ubuntu эта же команда **обязана пройти заново**. Проверка на
Windows или синтетических POSIX fixtures не доказывает работу Ubuntu GPU.

## Платформенные привязки и границы проверки

1. Windows API проверки процессов/`CREATE_NO_WINDOW` теперь ограничены Windows;
   POSIX использует свою ветку. `process_runtime` сверяет platform/hostname,
   Linux boot/start identity: совпавший номер чужого PID не считается нашим
   worker. Старые PID-файлы требуют явного `--discard-foreign-pids`, а не
   автоматического удаления. `.exe` и старый Windows bootstrap не используются
   как Linux launch path.
   Для нового запуска используется `turkeyworld.bootstrap`, который задаёт
   OMP/MKL threads до импорта Torch и не запускает ничего при простом import.
2. В планах, checkpoints, исходных ledgers, corpus receipts, preview revisions,
   policy/world bindings и validation identities лежат абсолютные
   **`E:\nau\TurkeyWorld\...`**. Проверки связывают их с SHA256 других документов.
   **Не делать глобальную замену путей и не перезаписывать старые планы.**
   Реализованный `workspace_paths` отделяет native paths от исходной identity.
   Не заменять его глобальным редактированием JSON и не создавать новую
   lineage без необходимости. Проверять всю цепочку через read-only preflight.
3. Копирование может изменить `mtime_ns`. Несовпадение VAL-панели нельзя
   молча принять, ослабить validator или объявить прежним fingerprint;
   требуется явная запись переноса с доказательством одинакового содержимого.
4. Linux `KeepAwake` и `scripts/run_training.sh` используют sleep-only
   `systemd-inhibit`, без изменения дисплея/блокировки. Внутренний guard
   подтверждает готовность и освобождает lock при выходе или смерти родителя.
   Ошибки inhibitor видимы; wrapper без него не продолжает молча.
   `--allow-sleep` — явное разрешение запуску без такой защиты, не стандартный
   режим. `tmux` удерживает терминальную сессию, но сам по себе не запрещает
   suspend. [systemd-inhibit manual](https://www.freedesktop.org/software/systemd/man/latest/systemd-inhibit.html)
   описывает жизненный цикл lock. Блокировку экрана отключать не требуется.
5. Аудит pickle последних world/policy/RF3/SR не обнаружил глобалов
   `pathlib.WindowsPath`/`PosixPath`. Не вводить ненужный monkeypatch сериализации:
   подтверждённая проблема — абсолютные строки и их identity-граф.

Перенос tensor-checkpoints возможен в принципе, но смена ОС/GPU/ядер и версии
Torch не гарантирует побитно одинаковые **будущие** SGD. Сохранённые веса,
Adam moments и RNG перед первым update должны совпасть с исходным checkpoint.
Правила Source при переносе не менять: закреплённые `simulation_clock.py`,
`structured_state.py`, `engine.py`, `spawn.py`, `anchor_spawn.py`,
`spawn_render_training.py`, `runtime_checks.py` сверить с code/manifest и
сохранить. Миграция ОС не должна вводить исправления координат движком,
сбросы памяти или дополнительные настоящие кадры в нейросетевой rollout.

## Порядок следующей сессии

Итоговые артефакты переноса:
[код ZIP](training_runs/ubuntu_handoff_20261115/code_snapshot.zip),
[SHA и состав кода](training_runs/ubuntu_handoff_20261115/code_snapshot_manifest.json),
[отчёт проверки](training_runs/ubuntu_handoff_20261115/portability_verification.json),
[128 прошедших тестов](training_runs/ubuntu_handoff_20261115/portability_tests_windows.json).
ZIP содержит исходники и диагностические скрипты; данные и веса переносить отдельно.

1. Прочитать финальные pause receipt/manifest и H115, проверить отсутствие
   работающих старых workers. Stop markers оставить до явного возобновления.
2. Проверить полноту workspace, hashes, время VAL-файлов, code snapshot и
   окружение. Не ограничиваться tracked Git-файлами.
3. Проверить реализованный POSIX controller и `workspace_paths`, сохранив
   оригинальные планы/receipts и SHA provenance. Запустить preflight выше.
   Старые Windows PID/launch-records сохранить как историю и оформить перенос;
   не трактовать совпавшие номера как живые Ubuntu-процессы.
4. Без SGD загрузить полные world/policy checkpoints и сверить шаги, все веса,
   Adam groups/moments/name order, CPU/CUDA/PCG и private samplers, текущие
   round/stage counters, Source corpus и policy absolute actor target.
5. Прогнать соответствующие CPU-тесты и закреплённую VAL180; показать различия
   с последней Windows-валидацией. Пробный update, если понадобится, выполнять
   только в одноразовом клоне и не подключать его к production.
6. Только после явного запроса пользователя продолжить незавершённую фазу
   с полной сохранённой границы командами ниже. Обучение всё ещё поставлено
   на паузу; наличие готовых launchers не является разрешением запустить их.

## Явное продолжение после успешной проверки

Ubuntu, внутри полной копии проекта с установленным `.venv-ubuntu`:

```bash
tmux new -s turkeyworld
bash scripts/run_training.sh --module turkeyworld.dreamer_motion_cycle \
  --resume-after-user-stop --discard-foreign-pids --cycles 1
```

Это завершит только текущий внешний раунд: **22 оставшихся world updates**,
затем20 head+50 actor/critic; следующий Source-сбор не начинается из-за
`--cycles 1`. Без этого флага контроллер продолжает последовательные раунды.
Обычный запуск не очищает stop markers. Явный `--resume-after-user-stop`
архивирует их; `--discard-foreign-pids` архивирует подтверждённые чужие/legacy
PID records. Живой собственный worker не должен приниматься за чужой.

Wrapper предпочитает активный `VIRTUAL_ENV`, затем `.venv-ubuntu`, `.venv`,
потом `python3`; `TURKEYWORLD_PYTHON` задаёт интерпретатор явно.
Оставаться в той же пользовательской сессии; screen lock не равен выходу
пользователя, выключению или принудительному suspend.

Windows: основная команда напрямую через Python, без зависимости от
PowerShell ExecutionPolicy:

```powershell
python -X utf8 -u -m turkeyworld.bootstrap --module turkeyworld.dreamer_motion_cycle `
  --resume-after-user-stop --discard-foreign-pids --cycles 1
```

`scripts/run_training.ps1` — необязательная оболочка с теми же target/flags,
если существующая политика разрешает её запуск. На этой Windows стандартная
ExecutionPolicy блокирует `.ps1`; менять защитные настройки не требуется.

На исходной Windows фактически проверен один CUDA inference-step мира8111
с RF3/SR3000 до1024, без SGD и GUI. Это не тест настоящего Ubuntu GPU,
не длинное прохождение и не quality-pass. Все checkpoints остаются на паузе.
Платформенные ветки проверены45 focused CPU-тестами; установщик — ещё3
syntax/dry-run проверками. Они не заменяют проверку на целевой Ubuntu.

Короткое задание следующему агенту: «Прочитай `CONTINUE_ON_UBUNTU_RU.md`,
финальные pause receipt/manifest и H115. Проверь перенос и подготовь точное
продолжение на Ubuntu без сброса весов/Adam/RNG/gates. Пока не запускай обучение».
