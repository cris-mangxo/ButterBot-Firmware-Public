# Experimento acústico de Mantequilla, R37

El experimento comparte el modelo CI32 y el decoder PocketSphinx con Hora. El diccionario combina las entradas originales de Hora y las de R36. La pronunciación acentuada de «mantequilla» usa un alias interno para mantener la entrada de Hora intacta. El detector independiente de palabras debe coincidir con el segmento FSG del objeto y con el final del habla. La ruta afirmativa completa compite con frases semánticas de negación, sustitución y mención. Las palabras de cancelación detectadas durante la orden vetan la salida.

El arnés separa la extracción de medidas y la aceptación. La generación de activos omite el umbral por defecto para poder recolectar evidencias sin aceptar comandos. La opción de diagnóstico de escritorio puede evaluar `DiceNegationConflict` exclusivamente para Mantequilla. `DiceNegationVeto`, `FactVeto` y los demás vetos siguen terminales. En ESP32-S3, el build solo acepta activos con la salida acústica de Mantequilla deshabilitada.

## Preparar el banco local

Usa el modelo CI32 expandido de Hora, las cuatro fuentes R36 locales y PocketSphinx ya compilado. Con Python 3.12, el mismo generador produce el directorio para escritorio y `ButterModelAssets.h` para el build experimental.

```fish
python3.12 experiments/esmx-pilot/butter_acoustic/pack.py \
  --clock-model "$clock_model" --r36-searches "$r36_searches" --out "$butter_assets"
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

Define las rutas `BB_ESMX_SPHINX_INCLUDE`, `BB_ESMX_SPHINX_GENERATED_INCLUDE` y `BB_ESMX_SPHINX_LIBRARY` como en el arnés de Hora. El ejecutable `probe` imprime rangos y puntuaciones, además de comprobar que una búsqueda de Hora conserve el resultado tras las tres búsquedas de Mantequilla.

Para una demostración local, `pack.py --gate` lee un archivo JSON explícito con `keywordMinimum`, `onsetSlack`, `articleLead`, `endSlack`, `tailSlack`, `floorPerFrame`, `contrastPerFrame` y `oppositionKeywordMinimum`. El manifiesto registra valores y huellas. El arnés real de rutinas lee PCM e infiere la etiqueta. El argumento `butter` es la expectativa comprobada por el arnés.

```fish
env BB_ESMX_BUTTER_DIAGNOSE_DICE_CONFLICT=1 \
  bash experiments/esmx-pilot/routine_host.sh "$voice_assets" "$pilot_assets" butter < "$pcm"
```

Para verificar el enlace Xtensa, agrega `-D BB_ESMX_BUTTER_ACOUSTIC=ON -D BB_ESMX_BUTTER_ASSETS="$butter_assets"` al build acústico de Hora y usa un banco generado sin `--gate`. Mantén locales los clips, los modelos y los binarios resultantes. La prueba Xtensa solo verifica el enlace de los componentes. Las mediciones de audio y la demostración de la rutina se ejecutan en la Mac. El detector queda pendiente de un filtro de falsos positivos y de las pruebas reales en el robot.

El arnés recoge los datos de ambos FSG incluso cuando no detecta la palabra objetivo. Antes de probar la latencia en placa, omite esas búsquedas cuando falte un testigo acústico elegible y verifica que la decisión se conserve.
