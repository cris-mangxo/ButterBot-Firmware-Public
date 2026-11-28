# Primer piloto es-MX del robot

**Candidato experimental parcial de escritorio.** Perfil opcional para el proyecto completo de CircuitMess. Usa el mismo AFE de Espressif y una tarea separada para reconocer un turno completo, con memoria de audio acotada en PSRAM. El consenso D6 reutiliza los bancos fijados en los ensayos previos. El despacho usa la tabla del fabricante para mantequilla, chiste y hora. D6 singular usa una tirada de un dado de seis mediante la misma rutina de resultado, animación, sonido y datos al mando.

| Orden | Acción original | Respuesta del piloto |
| --- | --- | --- |
| Pásame la mantequilla | `PassTheButter` | Cinco variantes de frase con sus ID existentes |
| Échame un chiste | `Joke` | Cinco chistes con sus ID existentes |
| ¿Qué hora es? | `CurrentTime` | Hora compuesta y aviso si el RTC no está configurado |
| Tira un dado de seis | `DiceRoll`, D6, cantidad 1 | Aviso y resultado de 1 a 6 con la voz del piloto |

El perfil está desactivado de forma predeterminada. Con el piloto activo el estado de escucha puede despachar esas cuatro intenciones. En la prueba Ald de los textos de la tabla se reconocieron Mantequilla y D6. Chiste y Hora se abstuvieron y su ruta al despachador se verificó con la intención inyectada. En la regresión anterior sí pasó «Aviéntate un chiste». Las demás opciones de voz quedan para fases siguientes. El mando conserva sus acciones y datos. Las pantallas que deducen la voz a partir de un ID pueden seguir mostrando su catálogo anterior. El piloto usa voz Ald generada en el escritorio y conserva el decodificador de fábrica. [Preparación y validación de voz](voice.md).

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

El ejecutable en Mac recibe PCM16 mono a 16 kHz por entrada estándar y llama a la misma función de inferencia que compila el firmware. La prueba alimenta el WAV completo. En el firmware la entrada se obtiene del AFE con ganancia, caché y recorte por actividad de voz. Devuelve el valor de `EsMx::Kws::Intent`. `replay.py` contrasta paneles públicos y sintéticos que ya están materializados localmente. Publicar solo evidencia reducida. La prueba de captura simula tramas y tareas, la prueba de despacho usa tipos del fabricante y la de audio usa el decodificador Helix. Micrófono, pausas de negación, altavoz, latencia, memoria en funcionamiento y movimiento se verificarán en placa en la fase autorizada.

| Panel | Resultado |
| --- | --- |
| Regresiones públicas consultadas | 03 y 81 activan D6. 37 se abstiene. El acierto 61 corresponde también a desarrollo. |
| D6 reservado | 88 y 89 se abstienen. |
| Demás positivos públicos | 0/15 reconocidos. El conjunto público total suma 3/20. |
| Controles negativos públicos | 33/33 se abstienen. |
| Hora reservada | 0/2 con personas de Monterrey ajenas al ajuste. |
| D6 sintético | 39/40. Falla `test-0292`. |
| Veto original de Dato curioso | Se abstiene en `test-0379`. |

Mantequilla y Chiste siguen sin una muestra humana es-MX reservada apta. Queda por medir la espera al cancelar durante inferencia y confirmar en la pareja física la animación y el valor de D6 sin pregunta de cantidad. La revisión del código del mando muestra un constructor que acepta `RollAnim` como dato inicial y reconstruye la vista con `Result`.

Una evaluación posterior debe reservar locutores antes del ajuste y conservar aparte los casos históricos consultados 03, 37 y 81. Los casos 88 y 89 permanecen reservados. El veto de «Dato curioso» sigue en el banco de frases y el perfil solo despacha las cuatro acciones enumeradas. [Frontend de audio de Espressif](https://docs.espressif.com/projects/esp-sr/en/latest/esp32s3/audio_front_end/README.html).

## Referencia de transcripción en Mac

Este diagnóstico compara la inferencia compacta con dos modelos Whisper locales seguidos de una gramática cerrada y el selector de acciones del fabricante. Requiere las muestras locales selladas del piloto, Python 3.12, `whisper-cli` 1.9.6, los pesos fijados por el script y los encabezados generados en la construcción de escritorio. Los modelos permanecen en la Mac. La salida guarda decisiones, rutas y hashes. El audio capturado y las transcripciones que consume el diagnóstico se procesan en un directorio temporal.

```sh
bash experiments/esmx-pilot/build_host.sh "$lab/generated" "$lab/recognize"
bash experiments/esmx-pilot/build_host.sh "$lab/generated" "$lab/capture" --capture
bash experiments/esmx-pilot/build_transcript_reference.sh "$lab/rule"
python3.12 experiments/esmx-pilot/transcript_reference.py \
  --whisper "$(command -v whisper-cli)" \
  --small "$experiment_root/asr-d6/ggml-small-q5_1.bin" \
  --turbo "$experiment_root/autonomous-v4/ggml-large-v3-turbo-q5_0.bin" \
  --reference "$lab/rule" --recognizer "$lab/recognize" --capture "$lab/capture" \
  --experiment-root "$experiment_root" > "$lab/reference.json"
```

En cinco tomas humanas reservadas o complementarias, una segunda ejecución de cada brazo simulado extrae el PCM entregado a la inferencia. Las modalidades `--pcm` y `--pcm-with-hangover` producen el audio binario para el diagnóstico local. En este panel los dos brazos entregaron PCM idéntico por toma y las decisiones de los modelos de referencia coincidieron con las del WAV completo. La anotación textual de tres tomas de Hora permite comparar por separado palabras transcritas y cobertura de gramática. Para D6 no se cuenta con esa anotación y se conserva la procedencia de la etiqueta humana. Los alias adicionales de Hora forman parte de este diagnóstico, que consultó las tres anotaciones. Las cinco tomas siguen excluidas del entrenamiento y calibración del compacto. Los resultados de la referencia sobre ellas son diagnósticos.

Dentro del subconjunto humano sin casos marcados como desarrollo (19 órdenes), el compacto acertó 0, Small 1 y Turbo 4. Este tamaño no establece superioridad estadística. El panel de 36 negativos quedó en abstención para las tres rutas, incluido el veto de «Dato curioso». Micrófono, AFE, movimiento y altavoz físicos quedan fuera de la medición.

## Repetición a través de la captura y comparación de un respaldo

El segundo ejecutable de Mac hace pasar WAV de laboratorio por `PilotCapture` y el reconocedor real. Simula un detector de energía con ventanas de 20 ms. El informe compara el WAV completo, el detector inmediato y el mismo detector con 1 s de retención tras la última trama activa. Son comprobaciones de recorte, pre-roll y entrega de PCM. El recorte habilitó `public-78`, una ganancia frágil que ya se consultó, y perdió `public-81`, que pasa con el WAV completo. En el checkpoint anterior, la retención de un segundo y la confirmación de fin de 320 ms dejan un máximo de aproximadamente 2.7 s de voz continua dentro del búfer de 4 s, incluso antes de contar pre-roll. En el checkpoint anterior, entre 112 casos se descartaron 6 por capacidad con VAD inmediato y 22 con retención. Entre los 33 negativos públicos, el reconocedor recibió 29 con VAD inmediato y 16 con retención. Ambos conjuntos clasificados quedaron en abstención y los demás controles se descartaron antes de inferencia. Ganancia, VAD de Espressif, latencia y acústica del robot requieren placa.

```sh
bash experiments/esmx-pilot/build_host.sh "$lab/generated" "$lab/capture" --capture
python3 experiments/esmx-pilot/capture_replay.py "$lab/recognize" "$lab/capture" "$experiment_root"
```

El respaldo int8 de `fallback_lab.py` es un experimento de Mac. Usa la CNN y el formato binario ya probados en D6, con cuatro clases, y calibra los márgenes con voces sintéticas separadas. Los paneles públicos se abren después de fijar el banco. Su salida se compara solo cuando el reconocedor congelado se abstiene. La variante rescató 5 órdenes sintéticas y añadió 10 activaciones incorrectas sobre voces sintéticas. En público activó Chiste en 37 y 64, dos controles, y Hora en 78, una orden D6. Ambas tomas reservadas de Hora quedaron en abstención. El panel público histórico carece de afirmaciones etiquetadas de Mantequilla, Chiste u Hora. El respaldo queda fuera del perfil de firmware.

Con el Python 3.12 de entrenamiento con PyTorch 2.9.1 y NumPy 2.4.2, el entorno Piper de Ald, las voces de Mac de la ficha y los paneles locales previos:

```sh
"$training_python" experiments/esmx-pilot/fallback_lab.py \
  --feature-tool "$kws" --reference "$lab/recognize" \
  --experiment-root "$experiment_root" --ald "$ald" \
  --piper-python "$piper_python" --output "$lab/fallback"
```

`kws` es el ejecutable de los [experimentos compactos](../esmx-compactas/README.md), con los modos `features` y `acoustic-scores`. Los pesos y parámetros de cuantización ocupan 6 468 bytes. Se guardan localmente con cabecera de laboratorio, fuera del formato de banco D6, junto con el registro previo a la prueba y los WAV. La toma SpanishPod A0056 procede de Praxis Language y se usa para ajuste con licencia CC BY 3.0. [Ficha y licencia de la toma](https://commons.wikimedia.org/wiki/File:SpanishPod_newbie_lesson_A0056_dialogue.ogg).

## Retención del turno y cancelación tardía

El piloto espera 1,2 s de silencio anunciado por el VAD y conserva las pausas interiores. Guarda hasta 100 000 muestras PCM16 en PSRAM. Antes de inferir, recorta solo ceros digitales de los extremos. El reconocedor sigue limitado a 64 000 muestras. Un turno demasiado largo o incompleto se descarta entero. Para D6, un prefijo que ya clasifica como una orden completa y va seguido de más señal con energía tras una pausa de al menos 240 ms deja el turno en abstención. La detección marca ambigüedad acústica y cubre la cancelación débil probada. También puede abstenerse ante una aclaración legítima o al aparecer un tercer hueco de esa duración.

Los dos brazos de escritorio procesaron los 112 casos y recuperaron `public-81` con el relleno de ceros digitales del arnés. Los dos brazos coincidieron bajo ese relleno. Los 33 controles públicos alcanzaron el reconocedor sin generar una orden. El arnés enlaza el despachador real y observa la clave de rutina y su fábrica con dobles de los servicios. La reproducción y los efectos físicos siguen pendientes.

```sh
python3 experiments/esmx-pilot/late_cancel_replay.py "$lab/capture" "$d6_ald_wav" "$cancelacion_ald_wav"
```

El control toma dos WAV de Ald, una orden afirmativa de D6 y una frase de cancelación. Concatena las voces con pausas de 250, 880 y 950 ms, tres niveles de amplitud para la segunda frase y ambos brazos de VAD. En la evaluación se usaron «No, mejor no lo tires», «No, cancela» y «No lo tires». De 54 controles, 30 llegaron a inferencia y 24 excedieron la ventana del modelo. Las 30 inferencias quedaron en abstención. En una comparación host con la misma captura y el reconocedor de `451d185`, previo a la guardia, 14 de esas 30 inferencias despacharon D6. Se clasifica ambigüedad acústica en las tomas sintéticas. Falta medir las cancelaciones humanas.

El arnés añade una cola digital de ceros. Si el AFE real produce ruido no nulo durante el segundo de retención y los 1,2 s de confirmación, esa cola puede ocupar unos 2,2 s de la ventana de cuatro. Quedarían unos 1,8 s para la voz y sus pausas, frente a unos 2,7 s con la ventana anterior bajo los mismos supuestos. Un segundo de cola artificial de pico de 1 unidad PCM sobre `public-81` bastó para que el nuevo capturador se abstuviera por longitud en ambos brazos. Se requiere audio del AFE para decidir el recorte tolerable y verificar latencia en placa. Las cancelaciones fuera del mismo turno se quedan para otra prueba.
