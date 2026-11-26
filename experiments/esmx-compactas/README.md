# Prototipo acústico compacto de seis órdenes

Este corte recibe PCM mono s16le a 16 kHz. Por cada turno obtiene una firma int8 de 24 filas temporales y 20 bandas. Reutiliza los filtros y la FFT de P31, recorta extremos de baja energía y ajusta el ritmo con DTW en una banda de tres filas. El modelo contiene plantillas de seis intenciones y una clase de rechazo. Exige todas las clases y comprueba distancia, margen y empates. Cada intención aceptada conserva el escenario y parámetro de CircuitMess, incluido D6. El reconocedor se prueba por separado del turno de escucha del robot.

El modelo de prueba contiene **16 plantillas y 7,696 bytes**. Los dos hablantes sintéticos de entrenamiento se usan solo para generar las firmas. La voz de desarrollo se usó para ajustar los límites. Las comparaciones con Flo, con Ald de Piper y con fragmentos de radio se registran por separado. Guardar grabaciones, pesos, WAV y resultados detallados fuera de Git. [Origen de logmel, importado sin cambios de ButterBotDocs](https://github.com/cris-mangxo/ButterBotDocs/tree/2018a9b2c6571b2159098e183ebcd5c754dc8e15/experiments/asr/cpp).

## Prueba en la Mac

Desde la raíz del fork, en fish. Se necesita Apple clang, Python 3 y las voces de macOS que enumera `pilot.py`.

```fish
set -l lab (mktemp -d)
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I experiments/esmx-commands/host -I experiments/esmx-compactas/guard -I main/src \
  -I components/ButterBot-Common/src -I components/CMF/src \
  experiments/esmx-compactas/host/main.cpp \
  experiments/esmx-compactas/guard/Dice6Guard.cpp \
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

`d6_corpus.py` genera dos bancos Ald. El banco amplio reserva construcciones nuevas para medir la detección de D6 y sus controles. Las cuatro órdenes previas son comprobaciones de regresión y sus textos y tomas a velocidad 1.0 también aparecen entre las altas de habla fuera de orden. El banco `--anchor` separa las tomas y velocidades de dos frases fijas, «Tira un dado de seis» y «Avienta un dado de seis lados». Calibración y prueba usan el mismo segundo de silencio como control del frontend. El banco usa dos firmas de turno completo y fija el límite a partir de la calibración. Ambos métodos conservan las decisiones de la base para mantequilla, chiste, dato y hora. Las rutas D6 se deciden en el verificador con el mismo mapeo al escenario original.

**Resultado de laboratorio, 26 de noviembre de 2026.** Las dos frases fijas acertaron 3/4 tomas de la misma voz y texto a otras velocidades, y 0/8 órdenes con texto nuevo. Rechazaron 44/44 negaciones D6, 72/72 ejemplos de otros dados y 56/56 controles humanos de conversación. La versión cuantizada queda descartada. Activó D6 en 4/24 negaciones Ald y en 1/32 controles Common Voice. Las cuatro órdenes anteriores se conservaron. Ambos verificadores heredan el error de la base que activa «Dato curioso» al oír «No me digas un dato curioso». Common Voice se usó para medir rechazo de conversación con una copia local de SHA distinto del catálogo. Falta evaluar órdenes positivas y negativas cercanas en voces humanas. Los resultados se limitan al laboratorio, con el verificador fuera del firmware del producto.

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
