# Voz local del piloto es-MX

El piloto usa la voz de un solo locutor, Ald es-MX medium, para todos sus mensajes. Piper sintetiza fuera de la placa. El ESP32-S3 conserva el decodificador AAC de fábrica para reproducir clips en mono a 16 kHz. El componente empaca los clips dentro de la aplicación para el perfil opcional. [Ficha de Ald](https://huggingface.co/rhasspy/piper-voices/blob/main/es/es_MX/ald/medium/MODEL_CARD).

Mantequilla y chiste usan los primeros cinco identificadores de cada catálogo de fábrica para caber en la aplicación. Cada frase conserva su identificador de fábrica en audio y datos al mando. Se cubren las tres respuestas sin reloj configurado. La hora configurada reúne fragmentos para las horas y los minutos con la misma voz. D6 conserva la animación, el efecto de sonido y el resultado numérico del firmware de fábrica, con audios para anunciar una tirada y sus seis resultados. Se conservan los mensajes y datos que recibe el mando. La variante de D6 se activa solo al reconocer la orden singular por voz.

## Generar

Indica la ruta del modelo `es_MX-ald-medium.onnx` junto con su configuración `es_MX-ald-medium.onnx.json`. El directorio de destino local es el que se pasará al firmware con `-D BB_ESMX_ASSETS`.

```sh
python3 experiments/esmx-pilot/voice_generate.py \
  --piper /ruta/al/entorno/bin/piper \
  --model /ruta/a/es_MX-ald-medium.onnx \
  --output-dir /ruta/a/artefactos/locales

bash experiments/esmx-pilot/voice_validate.sh /ruta/a/artefactos/locales
```

El generador requiere Python 3.9 o posterior, Piper con el soporte de CLI para WAV individuales y FFmpeg. Al generar, comprueba todas las cabeceras AAC y muestra el tamaño y los hashes del modelo, su configuración y el audio. La segunda orden comprueba que el mismo decodificador Helix usado por el firmware procesa todos los cuadros en el host.

La generación es local. Los WAV intermedios se descartan y el modelo, el encabezado generado y las imágenes compiladas quedan fuera de Git. Antes de distribuir una imagen con la voz se debe conservar la procedencia de los pesos y comprobar sus términos. El piloto cubre las frases del catálogo de esta fase. La inteligibilidad, las pausas entre fragmentos y la reproducción física se medirán en el robot.

La primera generación con Piper 1.8.0, ONNX Runtime 1.32.0 y FFmpeg 9.1 produjo 67 clips con 116 700 bytes AAC. La verificación host del decodificador de fábrica procesó los 1 356 cuadros del paquete.
