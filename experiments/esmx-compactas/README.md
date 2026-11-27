# Prototipo acústico compacto de seis órdenes

Este corte recibe PCM mono s16le a 16 kHz. Por cada turno obtiene una firma int8 de 24 filas temporales y 20 bandas. Reutiliza los filtros y la FFT de P31, recorta extremos de baja energía y ajusta el ritmo con DTW en una banda de tres filas. El modelo contiene plantillas de seis intenciones y una clase de rechazo. Exige todas las clases y comprueba distancia, margen y empates. Cada intención aceptada conserva el escenario y parámetro de CircuitMess, incluido D6. El reconocedor se prueba por separado del turno de escucha del robot.

El modelo de prueba contiene **16 plantillas y 7,696 bytes**. Los dos hablantes sintéticos de entrenamiento se usan solo para generar las firmas. La voz de desarrollo se usó para ajustar los límites. Las comparaciones con Flo, con Ald de Piper y con fragmentos de radio se registran por separado. Guardar grabaciones, pesos, WAV y resultados detallados fuera de Git. [Origen de logmel, importado sin cambios de ButterBotDocs](https://github.com/cris-mangxo/ButterBotDocs/tree/2018a9b2c6571b2159098e183ebcd5c754dc8e15/experiments/asr/cpp).

## Prueba en la Mac

Desde la raíz del fork, en fish. Se necesita Apple clang, Python 3 y las voces de macOS que enumera `pilot.py`.

```fish
set -l lab (mktemp -d)
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I experiments/esmx-commands/host -I experiments/esmx-compactas/guard \
  -I experiments/esmx-compactas/autonomous -I main/src \
  -I components/ButterBot-Common/src -I components/CMF/src \
  experiments/esmx-compactas/host/main.cpp \
  experiments/esmx-compactas/guard/Dice6Guard.cpp \
  experiments/esmx-compactas/guard/PhraseGate.cpp \
  experiments/esmx-compactas/autonomous/AcousticDice.cpp \
  experiments/esmx-compactas/autonomous/WordDice.cpp \
  main/src/Audio/EsMxKws/Compact.cpp main/src/Audio/EsMxKws/logmel.cpp \
  -o "$lab/kws"
python3 -B experiments/esmx-compactas/pilot.py \
  --program "$lab/kws" --out "$lab/modelo" > "$lab/resultados.json"
```

Opcionalmente, pasar `--negative-dir` con WAV públicos, revisados fuera del repertorio, de hasta cuatro segundos. El programa no agrega esos ejemplos al modelo. El JSON reporta los resultados incluidos los rechazos y errores. Se pueden cargar las plantillas de la prueba de forma local para nuevas mediciones con `kws score <templates.bin> 6200 600`, pasando exclusivamente PCM s16le de un turno por entrada estándar.

El banco `esp32s3` permite construir un control con `-D KWS_WITH_MODEL=OFF` y el candidato con `-D KWS_WITH_MODEL=ON -D PILOT_MODEL_DIR=<salida de pilot.py>`. Usar ESP-IDF 5.5.3, `-D IDF_TARGET=esp32s3`, un directorio y un `SDKCONFIG` distintos por build. El modelo generado y ambas imágenes quedan en el laboratorio local. El banco reserva un PCM pequeño de 8,000 muestras para forzar el enlace de la cadena y el flujo habitual del robot conserva su configuración. La asignación de PCM real, latencia y convivencia con visión y audio se medirán con el robot después de su ensayo de restauración.

## Grabación opcional del hogar

Una tanda en la Mac por persona que acepte participar. Posición inicial a unos 45 cm, lugar tranquilo y robot apagado. Usar seudónimos como `P01`. Para desarrollo, repetir otro día con `--split dev` y una sesión nueva. Un clip de otra partición se graba de nuevo y la primera tanda queda como piloto de Mac.

1. Listar entradas de audio con `ffmpeg -f avfoundation -list_devices true -i ""`. El listado puede terminar con un mensaje de salida normal tras enumerar. Leer el índice en **AVFoundation audio devices**.
2. Ejecutar con el índice elegido, por ejemplo `python3 -B experiments/esmx-compactas/record.py --person P01 --mic 0 --split train`.
3. La persona que habla escribe `ACEPTO` si autoriza el uso local. Para cada toma, pulsar Enter, esperar a **AHORA**, hablar una vez con ritmo natural y guardar silencio hasta **Guardada**. Usar `o` para omitir una toma o `q` para terminar. El piloto pide seis órdenes, dos frases de rechazo y un silencio.
4. Los WAV y el manifiesto quedan en la carpeta privada del proyecto dentro de `Library/Application Support`. Para seguimiento, compartir solo el número de tomas, la partición y el seudónimo si hace falta. La recogida formal con el micrófono del robot se hará en sesiones aparte.

Conservar los datos del hogar en almacenamiento privado. El piloto sintético tiene su propia ruta de generación. Un corpus humano público da voces de referencia sin pedir muestras del hogar y la verificación con participantes queda disponible si se decide personalizar.

## Comparación Ald y Common Voice es-MX

El corte del 26 de noviembre usa voces públicas y sintéticas. La voz del propietario queda excluida de los datos y de la síntesis. La guía de captura anterior se conserva como referencia histórica.

`ald_round.py` sintetiza pares de alta y evaluación separados, con Ald fijado por SHA-256. Usa Python 3.12, el binario `kws` y las 16 plantillas congeladas de la primera ronda. Conserva sus orígenes, añade ocho plantillas y compara ambos modelos con los mismos turnos y umbrales. `ald_cases.json` añade ejemplos positivos. `ald_contrast_cases.json` reparte las ocho altas entre positivos y rechazos. Los tamaños, audios y modelos se guardan en un directorio de resultados nuevo, fuera del repositorio.

```fish
python3.12 -B experiments/esmx-compactas/ald_round.py \
  --program "$lab/kws" --baseline "$lab/modelo/templates.bin" \
  --piper-python "$piper_python" --ald "$ald_onnx" --out "$ald_out"
```

`common_voice.py` requiere un archivo es-MX 26.0 descargado de Mozilla Data Collective. Comprueba el tamaño y el SHA-256 publicados antes de leerlo. Usa exclusivamente `test.tsv`, toma un fragmento de hasta cuatro segundos por hablante y conserva solo conteos de rechazo y activaciones en su salida. La opción `--alternate` compara un tercer modelo en los mismos clips. Los filtros léxicos excluyen frases cercanas al repertorio para medir conversación fuera de comando. Para reconocer órdenes positivas en voces humanas se requiere un conjunto etiquetado distinto. El muestreador no extrae rutas del tar al sistema ni incorpora voces al modelo. Los IDs de hablantes y las transcripciones permanecen fuera del informe.

```fish
python3.12 -B experiments/esmx-compactas/common_voice.py \
  --dataset female --archive "$archivo_mozilla" --program "$lab/kws" \
  --baseline "$lab/modelo/templates.bin" --candidate "$ald_out/templates.bin"
```

Descargar los archivos desde las [fichas femenina](https://mozilladatacollective.com/datasets/cmr3zunmk00flnt07p7ai9p43) y [masculina](https://mozilladatacollective.com/datasets/cmr3jjevj0041nt07vb3sm35d). Mozilla pide autenticar el acceso y aceptar los términos. Mantener los archivos, las plantillas y los audios fuera de Git.

## Verificación acotada de D6

`d6_corpus.py` genera dos bancos Ald. El banco amplio reserva construcciones nuevas para medir la detección de D6 y sus controles. Las cuatro órdenes previas son comprobaciones de regresión y sus textos y tomas a velocidad 1.0 también aparecen entre las altas de habla fuera de orden. El banco `--anchor` separa las tomas y velocidades de dos frases fijas, «Tira un dado de seis» y «Avienta un dado de seis lados». Entre calibración y prueba repite el texto en 24/24 ejemplos de otros dados y 4/20 negaciones. También usa el mismo segundo de silencio como control del frontend. El banco usa dos firmas de turno completo y fija el límite a partir de la calibración. Ambos métodos conservan las decisiones de la base para mantequilla, chiste, dato y hora. Las rutas D6 se deciden en el verificador con el mismo mapeo al escenario original.

**Resultado de laboratorio, 26 de noviembre de 2026.** Las dos frases fijas acertaron 3/4 tomas de la misma voz y texto a otras velocidades, y 0/8 órdenes con texto nuevo. En el banco amplio rechazaron 24/24 negaciones D6 y 48/48 ejemplos de otros dados. En el banco fijo rechazaron las 20/20 negaciones y los 24/24 ejemplos de otros dados. Rechazaron 56/56 controles humanos de habla leída fuera del repertorio. La versión cuantizada queda descartada. Activó D6 en 4/24 negaciones Ald y en 1/32 controles Common Voice. Las cuatro órdenes anteriores se conservaron. Ambos verificadores heredan el error de la base que activa «Dato curioso» al oír «No me digas un dato curioso». Common Voice se evaluó con una copia local de SHA distinto del catálogo. Falta evaluar órdenes positivas y negativas cercanas en voces humanas. La distancia DTW existente se expone para su uso en el banco. La ruta del verificador se enlaza únicamente en la compilación experimental.

Desde la raíz del fork, en fish, con el binario `$lab/kws` anterior y el modelo compacto congelado en `$baseline`:

```fish
python3.12 -B experiments/esmx-compactas/d6_corpus.py --anchor \
  --piper-python "$piper_python" --ald "$ald_onnx" --out "$anchor_dir"
for operation in fit ald
    python3.12 -B experiments/esmx-compactas/d6_guard.py "$operation" --anchor \
      --program "$lab/kws" --baseline "$baseline" \
      --weights "$anchor_dir/anchor.bin" --corpus "$anchor_dir"
end
python3.12 -B experiments/esmx-compactas/d6_guard.py common-voice --anchor --new-cohort \
  --program "$lab/kws" --baseline "$baseline" \
  --weights "$anchor_dir/anchor.bin" --archive "$archivo_mozilla"
```

Omitir `--anchor` para repetir el ensayo cuantizado descartado con otro directorio de corpus y el Python que tiene NumPy instalado. `--anchor --broad` cruza el verificador de frases fijas con el banco amplio. Cada ejecución de `ald` imprime los conteos por categoría junto con las decisiones individuales. Common Voice acepta aquí el archivo oficial o el SHA local conocido del diagnóstico, que se identifica como provisional. Sin `--new-cohort` usa los hablantes 33–64 y con la opción usa los 65–88 de la selección ordenada.

Para repetir la paridad, compilar el `kws` del [commit anterior](https://github.com/cris-mangxo/ButterBot-Firmware-Public/tree/f91f87b2d2f65bb0379df44a6b67a9cb560f9f82/experiments/esmx-compactas) siguiendo su README. Pasarlo como `$kws_reference` y apuntar `$broad_dir` al banco amplio generado sin `--anchor`.

```fish
python3.12 -B experiments/esmx-compactas/d6_verify.py --reference "$kws_reference" \
  --program "$lab/kws" --baseline "$baseline" --anchor "$anchor_dir/anchor.bin" \
  --corpus "$anchor_dir" --corpus "$broad_dir" --positive-wav "$anchor_dir/audio/train-0000.wav"
python3.12 -B -m unittest discover -s experiments/esmx-compactas -p 'test_*.py'
```

Para enlazar el verificador en el banco ESP32-S3, generar `dice6_anchors.h` con `d6_guard.py header --anchor --header <ruta>` y los mismos argumentos de programa, base y pesos. Agregar `-D DICE6_ANCHOR_DIR=<directorio del encabezado>` al comando de compilación del banco.

## Verificador ampliado de D6 y veto de Dato curioso

El banco de 9,676 bytes usa doce firmas D6 con radios independientes y seis firmas de negaciones de Dato curioso. Una negación puede cancelar Dato curioso si su distancia de turno completo cae en el radio de una firma negativa y mejora por al menos 256 la distancia a las dos firmas afirmativas. El banco verifica únicamente decisiones D6, desconocido y Dato curioso. La capa experimental sigue en `experiments/esmx-compactas`.

**Resultado de escritorio.** El verificador bloqueó el audio exacto de la regresión «No me digas un dato curioso» y conservó las 22 tomas afirmativas que la versión de dos frases acertaba en los conjuntos comparados. D6 pasó de 3/24 a 24/24 en las doce frases dadas de alta, a ritmos de prueba distintos, y obtuvo 0/8 en cuatro formulaciones inéditas. El turno original de negación de D6 que activó una variante de desarrollo se incorporó a calibración. El replay final conserva 20/20 rechazos de negaciones y 4/4 aciertos D6 del banco anterior. Common Voice aportó 45 clips dentro de cuatro segundos y sin activaciones, con 121 clips largos contabilizados aparte. El audio público es habla leída ajena al repertorio, de una copia local provisional. Falta medir órdenes humanas positivas y negativos cercanos. El banco compiló para ESP32-S3. El estudio no acredita uso en el robot.

Con `$fixed_anchor` apuntando a la referencia anterior de dos frases y `$anchor_dir` a su corpus:

```fish
python3.12 -B experiments/esmx-compactas/phrases_corpus.py \
  --piper-python "$piper_python" --ald "$ald_onnx" --out "$phrases_dir"
python3.12 -B experiments/esmx-compactas/phrases.py fit \
  --program "$lab/kws" --baseline "$baseline" --bank "$phrases_dir/bank.bin" --corpus "$phrases_dir"
for operation in ald confirm enrolled
    python3.12 -B experiments/esmx-compactas/phrases.py "$operation" \
      --program "$lab/kws" --baseline "$baseline" --anchor "$fixed_anchor" \
      --bank "$phrases_dir/bank.bin" --corpus "$phrases_dir"
end
python3.12 -B experiments/esmx-compactas/phrases.py replay \
  --program "$lab/kws" --baseline "$baseline" --anchor "$fixed_anchor" \
  --bank "$phrases_dir/bank.bin" --corpus "$anchor_dir"
python3.12 -B experiments/esmx-compactas/phrases_human.py --fresh \
  --program "$lab/kws" --baseline "$baseline" --anchor "$fixed_anchor" \
  --bank "$phrases_dir/bank.bin" --archive "$archivo_mozilla"
```

`ald` repite el panel usado durante el desarrollo. `confirm` mide textos nuevos. `enrolled` mantiene los textos de alta D6 y cambia el ritmo de síntesis. `replay` vuelve al banco anterior. La negación «No avienta un dado de seis lados» se incluyó en calibración a la velocidad que mostró el fallo durante ese replay. La toma se conserva como regresión conocida.

`--fresh` excluye los hablantes y las frases seleccionadas en el corte exploratorio de esta ronda y prioriza las frases cortas. Los clips largos se contabilizan aparte. Se analiza habla leída del corpus, sin publicar audio, transcripciones ni identidad de voces.

Para compilar el banco, generar `phrase_bank.h` con `phrases.py header --header <ruta>` y los mismos argumentos de programa, base y banco. Configurar `-D PHRASE_BANK_DIR=<directorio del encabezado>` y omitir `DICE6_ANCHOR_DIR`. Las pruebas directas de C++ se compilan con `host/test_phrase_gate.cpp`, `guard/PhraseGate.cpp`, `Compact.cpp` y `logmel.cpp` y las rutas de inclusión del reconocedor.

## D6 sin registrar el texto completo

La gramática de C++ en `transcript/` interpreta una petición afirmativa y completa de un solo dado de seis caras. Sigue después del banco anterior y solo se consulta cuando la base y el verificador devuelven desconocido. Un veto de Dato curioso y una orden ya reconocida conservan su resultado. Las transcripciones se calculan en la Mac con Whisper; los modelos de esta prueba empiezan en 32,152,673 bytes, por encima de la memoria flash de 16 MiB del robot. La opción `-D TRANSCRIPT_GRAMMAR_BENCH=ON` enlaza solo la gramática con una entrada de texto fija en el banco de ESP32-S3.

**Resultado de escritorio.** Whisper Small con ventana completa reconoció 5/10 formulaciones D6 nuevas de Ald en un panel reservado. El banco acústico anterior acertó 0/10. La gramática interpretó las diez al recibir el texto de referencia. Los 16 controles de otros dados, negaciones y menciones quedaron rechazados. Esta comparación deja fuera del banco del robot el motor de transcripción.

Desde la raíz del fork, con whisper.cpp 1.9.6 instalado, pesos multilingües `tiny-q5_1`, `base-q5_1` y `small-q5_1` fuera de Git y la base y el banco acústico fijados del ensayo anterior:

```fish
set -lx PYTHONPATH experiments/esmx-compactas
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I experiments/esmx-compactas/transcript -I main/src \
  experiments/esmx-compactas/host/text_gate.cpp \
  experiments/esmx-compactas/transcript/DiceGrammar.cpp -o "$lab/text-gate"
python3.12 -B experiments/esmx-compactas/transcript/render_cases.py --suite holdout \
  --piper-python "$piper_python" --ald "$ald_onnx" --out "$asr_corpus"
set -l asr_opts --whisper (command -s whisper-cli) --base "$whisper_base" \
  --tiny "$whisper_tiny" --small "$whisper_small" --program "$lab/kws" \
  --grammar "$lab/text-gate" --baseline "$baseline" --bank "$phrases_dir/bank.bin"
python3.12 -B experiments/esmx-compactas/transcript/evaluate.py holdout $asr_opts --corpus "$asr_corpus"
python3.12 -B experiments/esmx-compactas/transcript/evaluate.py human $asr_opts --archive "$archivo_mozilla"
```

También se puede usar `predict` con `--wav <PCM mono 16 kHz>`, `replay` con los corpus anteriores `--corpus` y `--older-wide`, o generar y evaluar `panel`, `confirm` y `final`. `--short-window` reproduce la comparación con ventana de contexto de 256. El script comprueba los pesos y el banco por SHA-256. Ejecuta Whisper con Metal y sin red ni acceso serie. Los clips de Common Voice, el audio sintético y las transcripciones temporales quedan fuera de Git.

## D6 acústico autónomo

El detector temporal de `autonomous/` se entrena en la Mac con las alineaciones fonéticas de Ald. En ejecución requiere solo audio PCM, el frontend existente y un banco de 9,140 bytes. Acepta señales de verbo, «dado» y «seis» en orden. Rechaza cuando también detecta negación, otro número o señales de cita y contexto. El banco anterior conserva la prioridad sobre toda decisión conocida, incluidas sus negaciones de Dato curioso.

**Resultado de laboratorio.** Obtuvo 5/16 D6 nuevos en un banco Ald y 4/10 al repetir el panel previo de transcripción, con 0/26 para el banco anterior. Rechazó los 60 controles sintéticos y los 55 clips humanos públicos de hasta cuatro segundos. El corpus de voces humanas aporta controles de rechazo. La evaluación con órdenes completas de personas y en el robot queda pendiente. El ejecutable Xtensa produjo las cinco decisiones de la muestra seleccionada en QEMU. La versión con audio fijo es exclusiva para pruebas en el emulador.

Requisitos de entrenamiento. Python 3.12 con PyTorch 2.9.1, NumPy 2.4.2 y un entorno separado con Piper 1.8.0, NumPy y ONNX 1.20.1. Usar `$lab/kws` compilado con el comando anterior.

```fish
set -lx PYTHONPATH experiments/esmx-compactas
"$piper_python" -B experiments/esmx-compactas/autonomous/corpus.py \
  --ald "$ald_onnx" --out "$unaligned_dir"
"$piper_python" -B experiments/esmx-compactas/autonomous/corpus.py --aligned \
  --ald "$ald_onnx" --out "$aligned_dir"
"$train_python" -B experiments/esmx-compactas/autonomous/train.py fit \
  --program "$lab/kws" --corpus "$unaligned_dir" --bank "$autonomous_dir/turn.bin"
"$train_python" -B experiments/esmx-compactas/autonomous/word_train.py fit \
  --program "$lab/kws" --corpus "$aligned_dir" --bank "$autonomous_dir/words.bin"
set -l word_opts --program "$lab/kws" --baseline "$baseline" \
  --prior "$phrases_dir/bank.bin" --turn "$autonomous_dir/turn.bin" --words "$autonomous_dir/words.bin"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py test $word_opts \
  --corpus "$aligned_dir" > "$autonomous_dir/test.json"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py replay $word_opts \
  --corpus "$phrases_dir" --older-wide "$broad_dir" > "$autonomous_dir/replay.json"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py human $word_opts \
  --archive "$archivo_mozilla"
"$train_python" -B experiments/esmx-compactas/autonomous/word_train.py header \
  --bank "$autonomous_dir/words.bin" --header "$autonomous_dir/word_bank.h"
```

Compilar para ESP32-S3 con `-D WORD_DICE_DIR=<directorio de word_bank.h>` junto con `KWS_WITH_MODEL` y `PHRASE_BANK_DIR`. Omitir `TRANSCRIPT_GRAMMAR_BENCH`. Para QEMU, generar `word_pcm.h` con `autonomous/qemu_fixture.py` y usar `-D WORD_DICE_TEST_PCM_DIR=<directorio de word_pcm.h>` en otro build. Ejecutar con `idf.py qemu --qemu-extra-args="-nic none"` apuntando al build de prueba. La salida esperada es cinco líneas `WORD_D6_CASE` y `WORD_D6_DONE 0`. [QEMU para ESP32-S3](https://docs.espressif.com/projects/esp-idf/en/v5.5/esp32s3/api-guides/tools/qemu.html).

## Más voces sintéticas y prueba con órdenes públicas

El banco complementario usa Ald y dos locutores sintetizados de [Sharvard](https://huggingface.co/rhasspy/piper-voices/blob/f5b7f7cfcab751578ac223f7decb9781b72d676a/es/es_ES/sharvard/medium/MODEL_CARD). Comparte las tramas con el detector anterior y solo se consulta después de su rechazo. La unión conserva decisiones anteriores y el veto de Dato curioso. Como en el primer experimento, la detección de negaciones de D6 es empírica y depende del audio.

El banco adicional aportó 1 reconocimiento sobre 10 órdenes sintéticas del panel previo y 2 sobre las 40 tomas repetidas. El panel Ald nuevo quedó en 9/12 y conservó el rechazo de 21/21 controles. Con voces humanas públicas obtuvo 0/2 peticiones directas de un D6 y 0/1 cláusula imperativa de un D6 extraída de una explicación. Rechazó los dos controles de varios dados. Una segunda variante con ejemplos sintéticos abreviados obtuvo 0/2 y 0/1 en esas mismas pruebas humanas y produjo una activación falsa en el panel sintético previo. La compilación de laboratorio usa el banco sin abreviaturas. El reconocimiento en personas queda abierto.

Usar las rutas del experimento anterior y crear directorios nuevos. La variable `sharvard_onnx` apunta al ONNX de la revisión enlazada. Colocar su `.onnx.json` junto al modelo. `public_sources` contiene los dos MP3 locales bajo los nombres `tirando-rol.mp3` y `oneshots.mp3`. Las ubicaciones y sumas se publican en [public_orders.json](autonomous/public_orders.json). La prueba usó FFmpeg 9.1. Comparar `cases[].wav_sha256` del manifiesto generado con `public_pcm_sha256` del recibo publicado usando sus identificadores. Los recortes se mantienen localmente.

```fish
set -l v2 "$lab/autonomous-v2"
mkdir -p "$v2/selected-bank"
"$piper_python" -B experiments/esmx-compactas/autonomous/corpus.py --aligned \
  --sharvard "$sharvard_onnx" --out "$v2/sharvard"
"$train_python" -B experiments/esmx-compactas/autonomous/word_train.py fit \
  --program "$lab/kws" --corpus "$aligned_dir" --sharvard-corpus "$v2/sharvard" \
  --bank "$v2/words-extra.bin"
"$train_python" -B experiments/esmx-compactas/autonomous/word_train.py header --supplement \
  --bank "$v2/words-extra.bin" --header "$v2/selected-bank/word_extra.h"
python3.12 -B experiments/esmx-compactas/autonomous/public_orders.py \
  --sources "$public_sources" --out "$v2/public-holdout"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py public-orders $word_opts \
  --extra "$v2/words-extra.bin" --corpus "$v2/public-holdout"
```

Para reproducir la variante de abreviaturas, generar otros dos corpus alineados con `--shorthand`, uno con Ald y otro con Sharvard, y pasarlos a `word_train.py fit` mediante `--shorthand-ald` y `--shorthand-sharvard`. Para generar el panel sintético nuevo usar `corpus.py --ald "$ald_onnx" --fresh --out <directorio nuevo>`. Evaluar ese directorio con `measure.py fresh`. `measure.py --extra` requiere un ejecutable anfitrión recompilado con el modo `word-pair`.

Para el banco doble, agregar `-D WORD_DICE_EXTRA_DIR=<directorio de word_extra.h>`. `qemu_fixture.py --public-dir <directorio de cortes>` incorpora los cinco clips públicos al build de prueba. Agregar `--supplement-hit` con el resultado de `measure.py replay --extra` para usar como caso D6 una toma que reconoce únicamente el banco adicional. Ambos argumentos juntos producen diez líneas `WORD_D6_CASE` y `WORD_D6_DONE 3` para este candidato. `WORD_D6_CASE` compara cada salida con su etiqueta y `WORD_D6_DONE` cuenta errores de reconocimiento. Comprobar por separado la paridad con el ejecutable anfitrión. Mantener el binario que contiene grabaciones en el directorio local de QEMU.

## Adaptación con voces humanas de MSWC

El entrenamiento puede incorporar palabras reales de la edición española del [Multilingual Spoken Words Corpus](https://mlcommons.org/datasets/multilingual-spoken-words/) de Mazumder y colaboradores, publicado por MLCommons bajo CC BY 4.0. Se reservan los identificadores de las tablas train, dev y test del archivo Common Voice es-MX usado para regresión. La auditoría recorre todas las tablas del archivo local. Solo contiene train, dev y test, con 314 identificadores y sin coincidencias con los dos grupos MSWC. La comprobación entre corpus se limita a los identificadores compatibles. Dentro de MSWC se separan entrenamiento y calibración por identificador de locutor. El manifiesto comprueba también que ningún archivo original se use en ambas particiones. Los dos episodios de la evaluación anterior siguen reservados y sus audios nunca se entregan al entrenador.

`human_words.py` verifica los ocho bloques de audio, el archivo de particiones y el de alineaciones. Toma palabras cortas del audio recortado y rellenado por el corpus, y construye turnos compuestos. `mix_human_words.py` sustituye algunas palabras por esos mismos recortes dentro de turnos continuos de Ald, conservando el resto de la frase. Esos turnos híbridos siguen siendo audio de entrenamiento; el panel de órdenes humanas completas usa audio continuo de los episodios.

Para reproducir, usar FFmpeg 9.1, las rutas del experimento anterior y un directorio de salida nuevo. En `mswc_dir` colocar estos archivos de la [revisión fijada](https://huggingface.co/datasets/MLCommons/ml_spoken_words/tree/0bc9df68e92fd6bb54176bf7eb29e2b9e97cb218).

- `data/splits/es/splits.tar.gz` como `mswc-es-splits.tar.gz`.
- `data/opus/es/train/audio/{2,3,4,8,9,10,13,14}.tar.gz` como `mswc-es-train-N.tar.gz`.
- [Alineaciones de español](https://mswc.mlcommons-storage.org/alignments/es.tar.gz) como `mswc-es-alignments.tar.gz`.

```fish
set -l v3 "$lab/autonomous-v3"
mkdir -p "$v3/adapted-bank"
"$train_python" -B experiments/esmx-compactas/autonomous/human_words.py \
  --sources "$mswc_dir" --reference "$archivo_mozilla" --out "$v3/human-words"
"$train_python" -B experiments/esmx-compactas/autonomous/mix_human_words.py \
  --human "$v3/human-words" --carrier "$aligned_dir" --out "$v3/human-mixed"
"$train_python" -B experiments/esmx-compactas/autonomous/audit_human_words.py \
  --reference "$archivo_mozilla" --corpus "$v3/human-words" "$v3/human-mixed"
"$train_python" -B experiments/esmx-compactas/autonomous/word_train.py fit \
  --program "$lab/kws" --corpus "$aligned_dir" --sharvard-corpus "$v2/sharvard" \
  --human-word-corpus "$v3/human-mixed" --bank "$v3/words-mixed.bin"
"$train_python" -B experiments/esmx-compactas/autonomous/word_train.py header --adapted \
  --bank "$v3/words-mixed.bin" --header "$v3/adapted-bank/word_adapted.h"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py public-orders $word_opts \
  --extra "$v2/words-extra.bin" --adapted "$v3/words-mixed.bin" --corpus "$v2/public-holdout"
```

Para evaluar la concatenación directa, entrenar con `--human-word-corpus "$v3/human-words"` y medir ese banco con `--extra`. La concatenación produjo una activación falsa ante una negación en las pruebas. La ruta híbrida se compara con `--extra` apuntando al banco suplementario anterior y `--adapted` al banco nuevo. El ejecutable anfitrión debe recompilarse para agregar `word-adapt`.

Compilar el banco de pruebas con `-D WORD_DICE_ADAPTED_DIR=<directorio de word_adapted.h>` junto con las opciones anteriores, incluido `WORD_DICE_EXTRA_DIR`. Conservar los tres bancos en su orden. Para QEMU, generar otro `word_pcm.h` con `--adapted-hit`, el resultado de `measure.py test --adapted` y el directorio de los cinco recortes públicos. El banco adaptado añadió dos aciertos en el panel sintético anterior, conservó las decisiones correctas de los dos bancos y siguió omitiendo los tres D6 humanos. QEMU produjo diez resultados y `WORD_D6_DONE 3`. Los pesos, las grabaciones y los binarios de prueba se generan localmente.

## Entrenamiento con instrucciones humanas continuas

`autonomous/whole_orders.json` fija seis grabaciones públicas, sus grupos de voces y los 16 intervalos usados. El entrenador recibe cada intervalo completo como una sola señal. Los cuatro D6 positivos de entrenamiento incluyen tres instrucciones del narrador para su propia tirada y una instrucción leída durante una partida. Las menciones de D6, otros dados y varias tiradas sirven de contraste. Los locutores se separaron por programa. Las grabaciones de Tirando Rol y OneShots, declaradas en `public_orders.json`, se usaron exclusivamente para prueba.

Descargar los enlaces `enclosure` de la ficha y nombrar cada archivo local con su clave más `.mp3`. La etiqueta y la posición se obtuvieron de audio continuo con Whisper local y contexto del episodio. El código comprueba las sumas de las fuentes y rechaza los episodios reservados por enlace o hash. Usa FFmpeg 9.1. Solo en entrenamiento se generan velocidades de 0.92, 1.00 y 1.08 dentro del límite de cuatro segundos. Cada variante entra ocho veces en el conjunto de ajuste. La calibración permanece a velocidad original.

Desde la raíz del fork y con las variables del experimento anterior disponibles, el banco temporal se entrena con el corpus Ald sin alineaciones y los intervalos humanos fijados.

```fish
set -l v4 "$lab/autonomous-v4"
mkdir -p "$v4/continuous-bank"
python3.12 -B experiments/esmx-compactas/autonomous/whole_orders.py \
  --sources "$public_training_sources" --out "$v4/continuous"
"$train_python" -B experiments/esmx-compactas/autonomous/train.py fit \
  --program "$lab/kws" --corpus "$unaligned_dir" --continuous "$v4/continuous" \
  --bank "$v4/whole.bin"
"$train_python" -B experiments/esmx-compactas/autonomous/train.py header \
  --bank "$v4/whole.bin" --header "$v4/continuous-bank/acoustic_dice.h"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py public-orders $word_opts \
  --extra "$v2/words-extra.bin" --adapted "$v3/words-mixed.bin" \
  --continuous "$v4/whole.bin" --corpus "$v2/public-holdout"
```

Recompilar el ejecutable anfitrión para agregar `word-full`. Con `--prioritize-human`, se intentó priorizar el único positivo humano en calibración y se obtuvieron los mismos bytes. La evaluación con `--continuous` conserva la comparación con los tres bancos previos. La CNN final solo añade un D6 después de que esos bancos se quedan en desconocido. El veto de Dato curioso termina la decisión antes de evaluar la CNN.

En ESP32-S3, el banco experimental requiere los tres bancos anteriores y `-D CONTINUOUS_DICE_DIR=<directorio de acoustic_dice.h>`. La variante existe en el banco de laboratorio. El resultado con voces humanas sigue en 0/3 y produjo tres activaciones falsas en Common Voice. Los pesos y el firmware de prueba quedan locales. Para QEMU, usar `qemu_fixture.py --continuous-hit` con el informe de `measure.py test --continuous` y los recortes de prueba. La ejecución esperada produce diez líneas `WORD_D6_CASE` y `WORD_D6_DONE 3`.

## Más órdenes y negaciones de tiradas humanas

Las fichas `whole_orders_expanded.json` y `public_orders_expanded.json` fijan el corpus ampliado y el panel nuevo de locutores. El entrenamiento incluye cinco instrucciones D6 nuevas de Alone in the ROL y seis negaciones de tirada sin número de caras. La calibración añade dos negaciones de Doblajes y Mazmorras. Se reservan por programa, enlace y hash tanto los dos episodios del panel anterior como Todo es Rock And Roll y Nadie Sabe Nada. Las tres negaciones del panel nuevo incluyen una de **varios** D6. Las otras dos no indican cuántas caras tiene el dado.

Cada enlace `enclosure` apunta a la publicación de origen. Al descargar, usar la clave de la fuente como nombre y la extensión declarada en `format`, con `.mp3` como valor predeterminado. El preparador usa FFmpeg 9.1 y verifica las sumas. Con las variables de las secciones anteriores, el experimento se reproduce en fish así.

```fish
set -l v5 "$lab/autonomous-v5"
mkdir -p "$v5/continuous-bank"
python3.12 -B experiments/esmx-compactas/autonomous/whole_orders.py --expanded \
  --sources "$training_sources" --out "$v5/expanded-train"
python3.12 -B experiments/esmx-compactas/autonomous/public_orders.py --expanded \
  --sources "$test_sources" --out "$v5/expanded-test"
"$train_python" -B experiments/esmx-compactas/autonomous/train.py fit \
  --program "$lab/kws" --corpus "$unaligned_dir" --continuous "$v5/expanded-train" \
  --bank "$v5/expanded.bin"
"$train_python" -B experiments/esmx-compactas/autonomous/train.py header \
  --bank "$v5/expanded.bin" --header "$v5/continuous-bank/acoustic_dice.h"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py public-orders \
  $word_opts --extra "$v2/words-extra.bin" --adapted "$v3/words-mixed.bin" \
  --continuous "$v5/expanded.bin" --expanded-public --corpus "$v5/expanded-test"
```

Recompilar el programa anfitrión para incorporar `word-guarded`. El informe compara la CNN continua sola y con la confirmación adicional. Esta confirmación solo examina un D6 nuevo propuesto por la CNN. Exige evidencia sostenida de «seis» y rechaza si el banco de palabras detecta una negación, otro número o contexto. Las órdenes previas y el veto del clasificador original conservan su prioridad. En ESP32-S3 se activa únicamente con `-D CONTINUOUS_DICE_GUARD=ON` junto con `CONTINUOUS_DICE_DIR`. La configuración anterior sigue siendo la referencia.

Para reproducir el panel nuevo en QEMU, usar `qemu_fixture.py --expanded-public --public-dir <directorio de los recortes>`. `--guard-control <wav>` usa un control local con etiqueta de rechazo y conserva las pruebas de negación, orden previa y veto de Dato curioso. La medición usó un falso positivo de Common Voice seleccionado después de observar la diferencia en la Mac. Se conserva como diagnóstico separado del panel nuevo. Mantener las grabaciones y los binarios de emulación fuera de Git.

## Recuperación D6 con orden temporal

El modo experimental `word-rescue` conserva la propuesta de la CNN y las decisiones anteriores. Ante una señal de contexto, requiere evidencia sostenida de verbo y «seis» en ese orden, separados entre 7 y 70 tramas de 20 ms. Una negación u otro número sostenidos siguen cancelando la propuesta. Las reglas del banco y el veto de Dato curioso se conservan.

El clasificador de palabras copia una ventana acotada al orden de las bandas del banco para que los productos int8 sean contiguos. Respeta los ceros al inicio y al final. `host/bench_word_logits.cpp` permite comparar el mismo cálculo en dos revisiones con `-O2`, incluyendo ventanas cortas y extremos. Compilarlo en cada revisión con `autonomous/WordDice.cpp`, `main/src/Audio/EsMxKws/logmel.cpp` y las rutas de inclusión del clasificador. La huella de salida debe coincidir. La comparación en escritorio no estima la latencia física.

Para preparar el [panel público](autonomous/public_orders_ordered.json), guardar cada fuente con su clave y extensión MP3 o WAV. Se distinguen las instrucciones continuas, los controles continuos y las cuatro composiciones de una negación explícita. Cada composición concatena la pronunciación de «no» de Lingua Libre a una instrucción humana con 70 ms de separación. Una de las instrucciones procede de una regresión conocida. El código protege los orígenes de este panel también al preparar el entrenamiento.

```fish
python3.12 -B experiments/esmx-compactas/autonomous/public_orders.py --ordered \
  --sources "$test_sources" --out "$v6/ordered-test"
python3.12 -B experiments/esmx-compactas/autonomous/measure.py public-orders \
  $word_opts --extra "$v2/words-extra.bin" --adapted "$v3/words-mixed.bin" \
  --continuous "$v5/expanded.bin" --rescue --ordered-public --corpus "$v6/ordered-test"
```

Para repetir las pruebas antiguas se agrega `--rescue` a los comandos de evaluación que usan `--continuous`. La columna `evaluated` excluye los clips fuera del máximo de cuatro segundos. En el banco Xtensa, habilitar `CONTINUOUS_DICE_GUARD=ON` y `CONTINUOUS_DICE_RESCUE=ON`. En el generador QEMU se puede elegir `--ordered-public` y repetir `--public-id` para ajustar los clips a la partición de laboratorio. El firmware del producto no incorpora este experimento.
