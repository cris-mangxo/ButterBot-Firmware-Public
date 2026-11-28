# Validación acústica de Mantequilla, R37

Este perfil experimental comparte el decoder PocketSphinx CI32-Q8 con Hora. El vocabulario de R37 se agrega al diccionario de Hora. La entrada acentuada de «mantequilla» usa un alias interno para conservar la entrada de Hora. La opción de compilación queda apagada por defecto.

La ruta pide un testigo acústico de 24 a 128 tramas para «mantequilla» o «la mantequilla», con puntuación desde −1800. Si falta, termina antes de ejecutar ambos FSG. El inicio y el final deben coincidir con los segmentos FSG en un máximo de 12 tramas. El final se recorta al último marco con voz. La cola puede medir hasta 24 tramas. Un término de cancelación desde −1900 que alcance dos tramas antes del final del objeto veta la solicitud. El FSG de petición debe superar por más de tres puntos por trama cualquier rival con oposición semántica, incluidas declaraciones de ubicación. El banco fonético nativo R24 confirma la intención con posterior ≥0.80, margen ≥0.30 y logit de negación <−0.5.

La ruta estándar evalúa Mantequilla después de los respaldos originales en un turno con `NoMatch`. La opción separada `BB_ESMX_BUTTER_WEAK_CONFLICT` habilita una prueba local de Mantequilla sobre una abstención `DiceNegationConflict`. `DiceNegationVeto`, `FactVeto` y demás causas terminales siguen bloqueadas. Se conserva explícitamente la variante estricta para medir cada caso.

## Preparar el banco local

Usa el CI32-Q8 expandido de Hora, las cuatro fuentes locales de contraste locativo de R37 y la biblioteca PocketSphinx compilada. Python 3.12 genera el directorio para escritorio y la cabecera de activos de Xtensa. `--validation` verifica las huellas de los siete archivos acústicos, el diccionario de Hora y las cuatro fuentes de R37 antes de habilitar la ruta.

```fish
python3.12 experiments/esmx-pilot/butter_acoustic/pack.py \
  --clock-model "$clock_model" --r36-searches "$r37_searches" \
  --validation --out "$butter_assets"
set -gx BB_ESMX_JOKE_FALLBACK 1
set -gx BB_ESMX_CLOCK_FALLBACK 1
set -gx BB_ESMX_CLOCK_ACOUSTIC 1
set -gx BB_ESMX_BUTTER_ACOUSTIC 1
set -gx BB_ESMX_BUTTER_ASSETS "$butter_assets"
set -gx BB_ESMX_CLOCK_ACOUSTIC_DIR "$butter_assets"
bash experiments/esmx-pilot/butter_acoustic/build_probe.sh "$probe"
"$probe" < "$pcm"
bash experiments/esmx-pilot/butter_acoustic/build_probe.sh "$gate_tests" --test-gate
"$gate_tests"
```

Define `BB_ESMX_SPHINX_INCLUDE`, `BB_ESMX_SPHINX_GENERATED_INCLUDE` y `BB_ESMX_SPHINX_LIBRARY` como en el arnés de Hora. `probe` publica medidas acústicas y verifica que las búsquedas no alteren el resultado de Hora entre llamadas. `build_host.sh` produce un ejecutable que recibe PCM por stdin y entrega su intención final. Sus opciones `--decision` y `--butter-phonemes` exponen el resultado base y el filtro fonético separadamente.

```fish
env BB_ESMX_BUTTER_WEAK_CONFLICT=1 \
  bash experiments/esmx-pilot/build_host.sh "$pilot_assets" "$host_weak"
env BB_ESMX_BUTTER_WEAK_CONFLICT=1 \
  bash experiments/esmx-pilot/routine_host.sh "$voice_assets" "$pilot_assets" butter < "$pcm"
```

El arnés de rutinas recibe PCM e infiere la intención antes de invocar el código original. El argumento `butter` indica el resultado esperado.

## Replay en QEMU

`prepare_target.py` recibe `--cases` con un JSON de casos `id`, `pcm` relativo al manifiesto, `sha256`, `intent` esperado y `upstream` esperado. Pasa `--model "$butter_assets/manifest.json"` y la ruta local de salida `--header` llamada `ClockTargetCases.h`. Para compilar el arnés aislado de Hora en ESP32-S3, fija su `BB_ESMX_CLOCK_TARGET_CASES` a ese directorio y usa los parámetros anteriores de PocketSphinx y el mismo CI32-Q8. Añade `-D BB_ESMX_BUTTER_ACOUSTIC=ON -D BB_ESMX_BUTTER_WEAK_CONFLICT=ON -D BB_ESMX_BUTTER_ASSETS="$butter_assets"` para probar la excepción experimental. El arnés imprime `BB_BUTTER_BUDGET` por turno, con el tiempo de inferencia y el resultado del límite de tres segundos.

Mantén locales los modelos, clips, cabeceras generadas y binarios. La ejecución aislada verifica inferencia y presupuesto emulado. La latencia con servicios activos y las pruebas físicas del robot quedan pendientes.
