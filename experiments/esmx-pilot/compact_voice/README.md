# Experimento de voz sin Flite

La variante requiere `BB_ESMX_PILOT=ON` y `BB_ESMX_COMPACT_VOICE=ON`. Usa Ald es-MX y clips locales G.711 μ-law a 16 kHz. Reproduce PCM a través de `SpeechAudioSource` y el generador de audio del fabricante. No requiere AAC para los primeros diagnósticos de la cámara y la tarjeta de motores. Mantiene el orden de las llamadas y los mensajes del firmware y deja los efectos de sonido existentes.

El catálogo cubre los diagnósticos de arranque, avisos de energía, apagado, sensores y Actividad, los turnos de Bola de la suerte, los avisos de dados, el anuncio de una tirada y su resultado hasta diez mil, junto a los demás grupos que enumera el manifiesto. Las entradas fijas que comparten intención pueden usar la misma frase de voz. La selección y los identificadores del protocolo siguen en el código del fabricante. Las rutas del piloto que ya reproducen AAC, como Hora y D6 de una tirada, siguen usando esos recursos.

Esta variante es un experimento de tamaño y reproducción, todavía incompleto. Hace falta cubrir las frases restantes, incluidas las clases de la cámara, y elegir cómo leer texto libre de notificaciones y títulos/artistas del teléfono, nombres IR persistidos como fonemas, y otros textos parametrizados. El backend termina la salida de un texto desconocido y registra un aviso sin incluir el texto. No cambia la lógica de las rutinas, las particiones, los datos ni el reconocimiento. Una compilación satisfactoria solo acredita el enlace y el tamaño. La prueba de escritorio usa la fuente de audio y la rutina de Actividad del fabricante con dobles de los servicios físicos. Comprueba las frases de arranque, el orden de guardado y sincronización de Actividad, los límites de los números y la cancelación entre clips.

## Generar y comprobar localmente

Con Python 3.12, Piper, FFmpeg y el modelo Ald con su archivo `.onnx.json` en el mismo directorio:

```fish
python3.12 experiments/esmx-pilot/compact_voice/generate.py \
  --phrases components/ButterBot-Common/src/Phrases.cpp --piper "$piper" \
  --model "$ald_model" --ffmpeg (command -v ffmpeg) --out "$compact_voice"
bash experiments/esmx-pilot/compact_voice/test_host.sh "$compact_voice"
```

Usa un directorio de compilación vacío por variante. Añade `-D BB_ESMX_COMPACT_VOICE=ON -D BB_ESMX_COMPACT_VOICE_ASSETS="$compact_voice"` al comando de compilación ya fijado para el piloto. Los activos, imágenes y binarios quedan locales. El manifiesto fija las huellas del catálogo, el modelo y las frases del fabricante.
