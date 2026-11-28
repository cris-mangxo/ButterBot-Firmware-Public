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

## Diagnóstico de etapas de D6

```sh
bash experiments/esmx-pilot/build_host.sh "$lab/generated" "$lab/stages" --dice-stages
python3.12 -B experiments/esmx-pilot/dice_stage_replay.py \
  --stages "$lab/stages" --recognizer "$lab/recognize" --capture "$lab/capture" \
  --experiment-root "$experiment_root" > "$lab/dice-stages.json"
```

En Mac, la sonda ejecuta el reconocedor de siempre y consulta de forma separada las mismas funciones del clasificador, el veto de frase, los bancos léxicos y el consenso previo a la guarda de seguimiento ambiguo. Las comprobaciones hipotéticas de cada banco describen qué decidiría por separado. La sonda carece de un registro de ramas ejecutadas. El verificador compara la decisión final con el ejecutable normal en 112 casos y con dos modalidades de captura en siete casos fijados. En cada PCM con firma exige también que el consenso reconstruido coincida con la decisión final. Si un caso nuevo activa la guarda de seguimiento y cambia ese resultado, este verificador exige diagnosticarlo por separado. El informe guarda decisiones, banderas y hashes, sin incluir muestras ni palabras inferidas.

## Repetición a través de la captura y comparación de un respaldo

El segundo ejecutable de Mac hace pasar WAV de laboratorio por `PilotCapture` y el reconocedor real. Simula un detector de energía con ventanas de 20 ms. El informe compara el WAV completo, el detector inmediato y el mismo detector con 1 s de retención tras la última trama activa. Son comprobaciones de recorte, pre-roll y entrega de PCM. En el checkpoint anterior a la retención del turno, el recorte habilitó `public-78`, una ganancia frágil que ya se consultó, y perdió `public-81`. El runtime actual recupera `public-81` en las dos modalidades simuladas. En el checkpoint anterior, la retención de un segundo y la confirmación de fin de 320 ms dejan un máximo de aproximadamente 2.7 s de voz continua dentro del búfer de 4 s, incluso antes de contar pre-roll. Entre 112 casos se descartaron 6 por capacidad con VAD inmediato y 22 con retención. Entre los 33 negativos públicos, el reconocedor recibió 29 con VAD inmediato y 16 con retención. Ambos conjuntos clasificados quedaron en abstención y los demás controles se descartaron antes de inferencia. Ganancia, VAD de Espressif, latencia y acústica del robot requieren placa.

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

## Banco binario de Hora para diagnóstico

`time_lab.py` reutiliza los WAV sintéticos de `fallback_lab.py` y las firmas de 24 por 20 del piloto. Usa distancia temporal de turno completo frente a ejemplos de Hora y controles. La fase `freeze` usa desarrollo sintético para fijar radio y margen entre clases. Opcionalmente agrega la toma de ajuste SpanishPod. Escribe el banco y los hashes de procedencia antes de ejecutar `replay` con las voces reservadas. Separa duplicados exactos de los casos sintéticos históricos del panel sin esos duplicados.

```sh
mkdir -m 0700 "$lab/time"
"$training_python" -B experiments/esmx-pilot/time_lab.py freeze \
  --root "$experiment_root" --features "$kws" --wav "$lab/fallback/wav" --bank-dir "$lab/time"
"$training_python" -B experiments/esmx-pilot/time_lab.py replay \
  --root "$experiment_root" --features "$kws" --wav "$lab/fallback/wav" --bank-dir "$lab/time" \
  --reference "$lab/recognize" --capture "$lab/capture" > "$lab/time/replay.json"
```

El banco de laboratorio mide 19 220 bytes en disco. El estimado de 200 bytes corresponde solo a dos filas de enteros de 32 bits para comparar una plantilla por vez. Quedan aparte el extractor, la firma de consulta y la pila. No se midieron memoria total ni latencia en el microcontrolador. En las dos voces reservadas de Hora hubo cero rescates. En el panel histórico introdujo cinco activaciones de Hora en otros casos, entre ellos una orden D6 reservada. El resultado descarta integrar este banco.

## Clasificador temporal conjunto en Mac

`keyword_lab.py` reutiliza el frontend C++ de palabras, sobre la salida de `kws word-frames`. Aprende tres señales, presencia de vocabulario, intención de turno y negación. La red pondera posiciones relativas en el turno. Clasifica turnos completos con umbrales fijados antes del replay. El experimento no determina tiempos ni orden explícito de palabras individuales.

```sh
mkdir -m 0700 "$lab/keywords"
"$training_python" -B experiments/esmx-pilot/keyword_lab.py freeze \
  --root "$experiment_root" --wav "$lab/fallback/wav" --features "$kws" --output "$lab/keywords"
"$training_python" -B experiments/esmx-pilot/keyword_lab.py replay \
  --root "$experiment_root" --wav "$lab/fallback/wav" --features "$kws" --output "$lab/keywords" \
  --reference "$lab/recognize" --capture "$lab/capture" > "$lab/keywords/report.json"
```

La ruta usa 4 898 bytes de pesos int8 más escalas en disco y opera con los pesos de nuevo en flotante en Python. No se midieron memoria, latencia ni equivalencia de inferencia entera en ESP32-S3. El panel público incluye casos históricos de desarrollo del compacto. El modelo nuevo no ajustó sobre esas grabaciones. Las falsas activaciones en ese panel y en las negaciones sintéticas impiden integrarlo. `keyword_public_replay.py` permite repetir el contraste adicional con los manifiestos sellados de CIEMPIESS LIGHT y el material suplementario de FSI. Sus audios permanecen locales.

## Transferencia acústica de palabras humanas

`word_transfer.py` preentrena la parte local de la red con el corpus de palabras humanas alineadas de MSWC y la ajusta después con órdenes completas. Sus dos comprobaciones para D6 exigen que el turno sea afirmativo y que haya evidencia acústica de verbo, dado y seis en ese orden. La preparación de MSWC usa `experiments/esmx-compactas/autonomous/human_words.py`. Los modos de congelación y replay están separados.

```sh
mkdir -m 0700 "$lab/transfer"
"$training_python" -B experiments/esmx-pilot/word_transfer.py freeze \
  --root "$experiment_root" --wav "$lab/fallback/wav" --features "$kws" \
  --human "$experiment_root/autonomous-v3/human-words" --output "$lab/transfer"
"$training_python" -B experiments/esmx-pilot/word_transfer.py replay \
  --root "$experiment_root" --wav "$lab/fallback/wav" --features "$kws" \
  --output "$lab/transfer" --reference "$lab/recognize" --capture "$lab/capture"
```

La variante con `--anahuac-words` y `--ciempiess-words` añade recortes humanos de desarrollo. Se entrena la presencia de la palabra a la vez que se etiqueta el turno como rechazo. Usa un directorio distinto para congelar cada variante. `cv27_word_prep.py` genera el manifiesto de Anáhuac a partir de la revisión fijada del corpus, su `manifest.json` de auditoría y los ocho TAR numerados según el script. El programa verifica dos alineaciones automáticas y guarda solo clips pequeños y metadatos locales. Las marcas de tiempo son estimaciones para desarrollo. El conjunto no tiene locutores verificados mediante identificadores y queda fuera de las pruebas de aceptación. [Anáhuac](https://huggingface.co/datasets/skeletonsword/fonetica-cv27/tree/2a9d611ee5c268fe8fd155adbb37eba790ef8e08/anahuac) · [Marcas de palabra en Whisper.cpp](https://github.com/ggml-org/whisper.cpp#word-level-timestamp-experimental).

Las dos variantes se descartaron y quedaron fuera del firmware. Ambas obtuvieron 0/8 en Mantequilla, Chiste y Hora y 0/12 en D6 en el panel sintético de prueba. Obtuvieron 0/2 en Hora humana y 0/2 en D6 humano. Asignaron Chiste a la orden D6 `public-89` y activaron indebidamente una negación sintética. Los 112 casos históricos incluyen los cuatro casos de Hora y D6 que se reportan como desgloses. Los cuatro quedaron fuera del ajuste y ya se consultaron, por lo que una comparación futura ciega requiere tomas nuevas. Los 5 081 bytes son de datos de pesos y escalas en disco. No se midió inferencia entera, memoria ni latencia en ESP32-S3. El replay registra abstención del compacto cuando el PCM supera su ventana y conserva ese caso en el panel.

### Alineación con la transcripción de Anáhuac

`mfa_cv27_compare.py` usa la transcripción del corpus para estimar el comienzo y el final de cada palabra. Reutiliza la selección fijada de `cv27_word_prep.py` y compara el resultado con el `alignment.json` de los dos decodificadores. Requiere [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html) 3.4.0, el [modelo acústico español 2.0.0a](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-spanish_mfa-v2.0.0a) y el [diccionario para Latinoamérica 2.0.0](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-spanish_latin_america_mfa-v2.0.0). Los modelos y la referencia se verifican por SHA-256. El subproceso `mfa align` usa el perfil `SANDBOX` de [`cv27_word_prep.py`](cv27_word_prep.py), que bloquea red y acceso a puertos serie. La preparación de PCM con ffmpeg y `mfa version` quedan fuera de ese perfil.

```sh
"$training_python" -B experiments/esmx-pilot/mfa_cv27_compare.py \
  --source "$anahuac" --reference "$lab/anahuac/alignment.json" \
  --ffmpeg "$ffmpeg" --mfa "$mfa" --acoustic "$acoustic" --dictionary "$dictionary" \
  --mswc "$experiment_root/autonomous-v3/human-words" --output "$lab/anahuac-mfa"
python3.12 -B -m unittest discover -s experiments/esmx-pilot -p test_mfa_cv27_compare.py
```

Se admitieron 48 de 49 recortes para desarrollo, siete de ellos de «mantequilla». La opción `--mswc` mide el solapamiento y error de límites con 16 montajes de palabras de la partición de calibración de MSWC. Obtuvo al menos media unión temporal en 51 de 55 palabras y 42,281 ms de error absoluto mediano por límite. Esa medición corresponde a montajes, mientras que Anáhuac contiene habla continua sin marcas manuales. Dos ejecuciones aceptaron los mismos recortes pero variaron los límites de cuatro entre 10 y 60 ms. El experimento de transferencia admite el manifiesto fijado de la primera ejecución. Sus WAV y manifiesto se conservan localmente. Una nueva alineación puede producir hashes diferentes. Ningún recorte cuenta como orden completa ni como prueba de un locutor reservado.

La variante con el manifiesto fijado y los dos recortes CIEMPIESS produjo rasgos válidos para 46 de los 50 recortes candidatos. Como respaldo, recuperó una orden pública histórica de D6 y una sintética. Sumó tres despachos incorrectos con WAV y dos con la captura de escritorio. En el desglose de voces reservadas obtuvo 0/2 para Hora y 0/2 para D6, con una activación incorrecta de Chiste. También activó una negación sintética. Como clasificador nuevo en el panel sintético obtuvo 0/8 en Mantequilla, 1/8 en Chiste, 0/8 en Hora y 1/12 en D6. Esta variante queda fuera de la aplicación.

## Destilación de fonemas en escritorio

`phoneme_teacher_lab.py` usa [Wav2Vec2 XLSR con inventario eSpeak](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft/tree/2c733782da5604684829819a5eb744c193fe9398). Verifica el peso oficial y proyecta las probabilidades a los fonemas que aparecen en el texto sintético de entrenamiento. La etiqueta vacía de CTC y el grupo de símbolos restantes se mantienen separados. Usa eSpeak NG 1.52.0 con `es-419` y el extractor nativo `word-frames`. Conserva los centros temporales del extractor al interpolar las probabilidades del maestro. El ajuste y desarrollo solo usan las voces sintéticas de esas particiones.

`phoneme_student_lab.py` entrena convoluciones temporales separables con señales de fonemas, intención y negación. Exporta los pesos int8 por canal junto con escalas y sesgos. El replay de escritorio convierte los pesos otra vez a flotante. La inferencia entera y el consumo en la placa quedan pendientes.

```sh
"$training_python" -B experiments/esmx-pilot/phoneme_teacher_lab.py \
  --root "$experiment_root" --wav "$lab/fallback/wav" --teacher "$teacher" \
  --espeak "$espeak_ng" --features "$kws" --output "$lab/phone-targets"
"$training_python" -B experiments/esmx-pilot/phoneme_student_lab.py \
  --cache "$lab/phone-targets" --output "$lab/phone-student"
"$training_python" -B experiments/esmx-pilot/phoneme_replay_lab.py \
  --root "$experiment_root" --features "$kws" --cache "$lab/phone-targets" \
  --frozen "$lab/phone-student" --reference "$lab/recognize" --capture "$lab/capture"
"$training_python" -B -m unittest discover -s experiments/esmx-pilot -p test_phoneme_lab.py
```

Requiere Transformers 4.57.6 y PyTorch 2.9.1. El maestro se carga desde el directorio local con `weights_only=True`. El replay verifica el sello del banco y del extractor y compara con el compacto que ya usa el piloto. El respaldo solo actúa si el piloto se abstiene. El resultado de escritorio distingue las nuevas órdenes acertadas de las activaciones incorrectas.

El banco de este experimento usó 9 180 bytes de pesos, escalas y sesgos. El respaldo reconoció una de dos tomas humanas reservadas de Hora en ambas entradas. En el panel histórico añadió 16 activaciones incorrectas por WAV y 17 con la captura retenida. Esos conteos se mantienen al retirar tres duplicados sintéticos exactos del ajuste o desarrollo. El resultado impide incorporarlo al firmware. La opción `--episode` emite la decisión de una toma humana sellada para pasarla al [arnés de rutinas](routine_host/README.md).

## Rutinas de fábrica en Mac

El [arnés de rutinas](routine_host/README.md) ejecuta los métodos `tick` de Mantequilla, Chiste, Hora y D6 con servicios sustituidos. Coteja el selector de intención, las estructuras que se enviarían al mando y los bytes que recibe el reproductor. Admite también PCM por entrada estándar para enlazar el reconocedor real con esa prueba.

## Retención del turno y cancelación tardía

El piloto espera 1,2 s de silencio anunciado por el VAD y conserva las pausas interiores. Guarda hasta 100 000 muestras PCM16 en PSRAM. Antes de inferir, recorta solo ceros digitales de los extremos. El reconocedor sigue limitado a 64 000 muestras. Un turno demasiado largo o incompleto se descarta entero. Para D6, un prefijo que ya clasifica como una orden completa y va seguido de más señal con energía tras una pausa de al menos 240 ms deja el turno en abstención. La detección marca ambigüedad acústica y cubre la cancelación débil probada. También puede abstenerse ante una aclaración legítima o al aparecer un tercer hueco de esa duración.

Los dos brazos de escritorio procesaron los 112 casos y recuperaron `public-81` con el relleno de ceros digitales del arnés. Los dos brazos coincidieron bajo ese relleno. Los 33 controles públicos alcanzaron el reconocedor sin generar una orden. El arnés enlaza el despachador real y observa la clave de rutina y su fábrica con dobles de los servicios. La reproducción y los efectos físicos siguen pendientes.

```sh
python3 experiments/esmx-pilot/late_cancel_replay.py "$lab/capture" "$d6_ald_wav" "$cancelacion_ald_wav"
```

El control toma dos WAV de Ald, una orden afirmativa de D6 y una frase de cancelación. Concatena las voces con pausas de 250, 880 y 950 ms, tres niveles de amplitud para la segunda frase y ambos brazos de VAD. En la evaluación se usaron «No, mejor no lo tires», «No, cancela» y «No lo tires». De 54 controles, 30 llegaron a inferencia y 24 excedieron la ventana del modelo. Las 30 inferencias quedaron en abstención. En una comparación host con la misma captura y el reconocedor de `451d185`, previo a la guardia, 14 de esas 30 inferencias despacharon D6. Se clasifica ambigüedad acústica en las tomas sintéticas. Falta medir las cancelaciones humanas.

El arnés añade una cola digital de ceros. Si el AFE real produce ruido no nulo durante el segundo de retención y los 1,2 s de confirmación, esa cola puede ocupar unos 2,2 s de la ventana de cuatro. Quedarían unos 1,8 s para la voz y sus pausas, frente a unos 2,7 s con la ventana anterior bajo los mismos supuestos. Un segundo de cola artificial de pico de 1 unidad PCM sobre `public-81` bastó para que el nuevo capturador se abstuviera por longitud en ambos brazos. Se requiere audio del AFE para decidir el recorte tolerable y verificar latencia en placa. Las cancelaciones fuera del mismo turno se quedan para otra prueba.
