# Prototipo acústico compacto de seis órdenes

Este corte recibe PCM mono s16le a 16 kHz. Por cada turno obtiene una firma int8 de 24 filas temporales y 20 bandas. Reutiliza los filtros y la FFT de P31, recorta extremos de baja energía y ajusta el ritmo con DTW en una banda de tres filas. El modelo contiene plantillas de seis intenciones y una clase de rechazo. Exige todas las clases y comprueba distancia, margen y empates. Cada intención aceptada conserva el escenario y parámetro de CircuitMess, incluido D6. El reconocedor se prueba por separado del turno de escucha del robot.

El modelo de prueba contiene **16 plantillas y 7,696 bytes**. Los dos hablantes sintéticos de entrenamiento se usan solo para generar las firmas. La voz de desarrollo se usó para ajustar los límites. Las comparaciones con Flo, con Ald de Piper y con fragmentos de radio se registran por separado. Guardar grabaciones, pesos, WAV y resultados detallados fuera de Git. [Origen de logmel, importado sin cambios de ButterBotDocs](https://github.com/cris-mangxo/ButterBotDocs/tree/2018a9b2c6571b2159098e183ebcd5c754dc8e15/experiments/asr/cpp).

## Prueba en la Mac

Desde la raíz del fork, en fish. Se necesita Apple clang, Python 3 y las voces de macOS que enumera `pilot.py`.

```fish
set -l lab (mktemp -d)
c++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I experiments/esmx-commands/host -I main/src \
  -I components/ButterBot-Common/src -I components/CMF/src \
  experiments/esmx-compactas/host/main.cpp \
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
