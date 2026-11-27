# Primer piloto es-MX del robot

Perfil experimental de escritorio para el proyecto completo de CircuitMess. Usa el mismo AFE de Espressif y una tarea separada para reconocer un turno completo, con memoria de audio acotada en PSRAM. El consenso D6 reutiliza los bancos fijados en los ensayos previos. El despacho usa la tabla del fabricante para mantequilla, chiste y hora. D6 singular usa una tirada de un dado de seis mediante la misma rutina de resultado, animación, sonido y datos al mando.

| Orden | Acción original | Respuesta del piloto |
| --- | --- | --- |
| Pásame la mantequilla | `PassTheButter` | Cinco variantes de frase con sus ID existentes |
| Échame un chiste | `Joke` | Cinco chistes con sus ID existentes |
| ¿Qué hora es? | `CurrentTime` | Hora compuesta y aviso si el RTC no está configurado |
| Tira un dado de seis | `DiceRoll`, D6, cantidad 1 | Aviso y resultado de 1 a 6 con la voz del piloto |

El perfil está desactivado de forma predeterminada. Con el piloto activo el estado de escucha atiende estas cuatro órdenes es-MX. Las demás opciones de voz quedan para fases siguientes. El mando conserva sus acciones y datos. Las pantallas que deducen la voz a partir de un ID pueden seguir mostrando su catálogo anterior. El piloto usa voz Ald generada en el escritorio y conserva el decodificador de fábrica. [Preparación y validación de voz](voice.md).

## Construcción de escritorio

Desde la raíz del fork, con ESP-IDF 5.5.3 ya activado. `experiment_root` apunta al directorio donde se generaron los bancos históricos, `piper` al ejecutable Piper 1.8.0 y `ald` al archivo `es_MX-ald-medium.onnx` que tiene al lado su configuración. El destino `lab` debe ser un directorio local nuevo, absoluto, ajeno al índice de Git. Los repositorios contienen las recetas de generación de los bancos y el manifiesto de sus encabezados fijados, y usan archivos locales para generar y compilar. [Experimentos de reconocimiento](../esmx-compactas/README.md) · [Manifiesto de entradas](model-assets.sha256).

```sh
mkdir -p "$lab"
python3 experiments/esmx-pilot/prepare_model.py "$experiment_root" "$lab/generated"
python3 experiments/esmx-pilot/voice_generate.py \
  --piper "$piper" --model "$ald" --output-dir "$lab/generated"
cp sdkconfig "$lab/pilot.sdkconfig"
idf.py -B "$lab/pilot" -D "SDKCONFIG=$lab/pilot.sdkconfig" \
  -DBB_ESMX_PILOT=ON -D "BB_ESMX_ASSETS=$lab/generated" build
```

Para comparar con el perfil predeterminado, usar un segundo directorio de build y otra copia de `sdkconfig`, con `BB_ESMX_PILOT=OFF`. Ambos perfiles usan la tabla de particiones de 16 MiB del fork. Las imágenes generadas quedan locales y el procedimiento termina al compilar.

## Verificación en Mac

```sh
bash experiments/esmx-pilot/build_host.sh "$lab/generated" "$lab/recognize"
python3 experiments/esmx-pilot/replay.py "$lab/recognize" "$experiment_root"
python3 experiments/esmx-pilot/test_capture.py --sanitize
bash experiments/esmx-pilot/test_dispatch.sh
bash experiments/esmx-pilot/voice_validate.sh "$lab/generated"
```

El ejecutable en Mac recibe PCM16 mono a 16 kHz por entrada estándar y llama al mismo reconocedor que el firmware. Devuelve el valor de `EsMx::Kws::Intent`. `replay.py` contrasta paneles públicos y sintéticos que ya están materializados localmente. Publicar solo evidencia reducida. La prueba de captura simula tramas y tareas, la prueba de despacho usa tipos del fabricante y la de audio usa el decodificador Helix. Micrófono, pausas de negación, altavoz, latencia, memoria en funcionamiento y movimiento se verificarán en placa en la fase autorizada.

Una evaluación posterior debe reservar locutores antes del ajuste y conservar aparte los casos históricos consultados 03, 37 y 81. Los casos 88 y 89 permanecen reservados. El veto de «Dato curioso» sigue en el banco de frases y el perfil solo despacha las cuatro acciones enumeradas. [Frontend de audio de Espressif](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/audio_front_end/README.html).
