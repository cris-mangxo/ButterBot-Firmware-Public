# Reconocimiento experimental de un dado de seis

Este experimento usa el decodificador CI16 compartido de Hora. La compilación de escritorio lo habilita con `BB_ESMX_D6_SCOPE_HOST=1`. La aplicación completa lo deja fuera. La inferencia sigue devolviendo el identificador que activa la rutina original de un dado.

El código consulta la evidencia acústica únicamente si el clasificador base deja un conflicto débil de negación de dado. Los vetos terminales, la detección de una segunda intervención y las rutas anteriores tienen prioridad. El decodificador localiza una orden completa y contrasta hipótesis afirmativas y negativas en ventanas alrededor de ella. También rechaza menciones detectadas cerca de la orden. El léxico conserva las pronunciaciones del diccionario existente. Si falta el archivo de palabras clave o alguna pronunciación existente entra en conflicto, esta ruta se abstiene y las búsquedas previas siguen disponibles.

## Reproducción local

Se necesitan el banco CI16 ya preparado para Hora, los encabezados generados para el piloto, los dos directorios de encabezados de PocketSphinx y su biblioteca estática local. Copia el banco a un directorio nuevo y agrega `dice-scope.kws` a ese directorio. Conserva el banco de referencia. Los audios y modelos quedan en el almacenamiento local según las condiciones de cada fuente.

```bash
mkdir -p "$model_dir"
cp -R "$reference_ci16/." "$model_dir/"
cp experiments/esmx-pilot/dice_scope/dice-scope.kws "$model_dir/"
BB_ESMX_JOKE_FALLBACK=1 BB_ESMX_CLOCK_ACOUSTIC=1 BB_ESMX_D6_SCOPE_HOST=1 \
  BB_ESMX_CLOCK_ACOUSTIC_DIR="$model_dir" \
  BB_ESMX_SPHINX_INCLUDE="$sphinx_source/include" \
  BB_ESMX_SPHINX_GENERATED_INCLUDE="$sphinx_build/include" \
  BB_ESMX_SPHINX_LIBRARY="$sphinx_build/libpocketsphinx.a" \
  BB_ROUTINE_HOST_AAC_OUT="$output_aac" \
  bash experiments/esmx-pilot/routine_host.sh "$pilot_assets" "$pilot_assets" dice6 < "$sample_pcm"
```

La entrada para esta prueba es PCM16 mono a 16 kHz. El archivo de salida debe ser nuevo. Una ejecución válida reconoce el identificador 6, ejecuta un ciclo de la rutina original, verifica su mensaje y coteja los clips AAC. Primero comprueba el caso público conocido 88 en las vistas cruda y retenida por el capturador. Repite con la opción apagada y con `dice-scope.kws` ausente. En estos dos modos 88 se abstiene y Hora y Chiste conservan sus resultados.

La política de alcance se fijó con datos de desarrollo antes de volver a evaluar 88 y 89. La política limitada recuperó 88 en las dos vistas, mantuvo 03 y 81, y dejó 37, 89 y los vetos de Dato curioso sin activación. 88 y 89 son diagnósticos históricos. Todavía se necesita una orden y una cancelación espontánea de locutores reservados de español mexicano y una prueba de memoria y latencia en el ESP32-S3 antes de ofrecer esta ruta en el dispositivo.


## Emulación de ESP32-S3

El arnés aislado de `clock_target` acepta `-D BB_ESMX_D6_SCOPE_BENCH=ON`.
Conserva el modelo CI16 de Hora y añade al sistema de archivos virtual la
lista KWS del mismo commit, con su SHA-256 verificado al configurar. Usa
`prepare_target.py` del experimento Mantequilla para producir la cabecera
de casos locales. El arnés informa intención, disposición anterior,
tiempo por turno y memoria. Un turno conserva el plazo de 3 s. La opción
queda reservada a esta compilación de escritorio y se excluye del firmware
completo del robot.
