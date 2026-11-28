# Prueba de rutinas del piloto en la Mac

El ejecutable compila el despacho piloto, los cuatro `tick` del robot, la tabla de frases del fabricante, `Voice.cpp` y `MemoryAudioSource.cpp`. El sustituto de audio lee el AAC de cada fuente de memoria y lo coteja byte a byte con las selecciones esperadas del banco local. La prueba verifica también las estructuras de escenario que recibe el sustituto de `Com`.

Desde la raíz del firmware, indica el directorio generado que contiene `voice_data.h`.

```bash
experiments/esmx-pilot/routine_host.sh "$voice_dir"
```

Se ejercitan los mensajes de mantequilla y chiste, el reloj sin configurar y tres horas concretas, las seis caras del dado y el orden de su animación, efecto y resultado. Se verifica el diálogo original de cantidad para D6 y el comportamiento cuando falta el servicio AAC. En ese caso se solicita la respuesta inglesa de fábrica.

La segunda forma compila también el reconocedor del piloto. Lee por entrada estándar un turno PCM s16le mono de 16 kHz de hasta cuatro segundos. Usa un directorio local con los encabezados de modelos y una expectativa `butter`, `joke`, `time`, `dice6` o `none`. Un desacierto emite las etiquetas esperada y obtenida, devuelve 1 y deja el conteo de ejecuciones en cero. Una coincidencia de orden ejecuta su rutina y verifica el mensaje y el AAC. Para `none` verifica la abstención sin ejecutar rutinas.

```bash
experiments/esmx-pilot/routine_host.sh "$voice_dir" "$model_dir" dice6 < "$pcm_file"
```

El directorio de compilación es temporal. El binario recibe muestras por entrada estándar y la salida se limita a resultados JSON.

La prueba de un reconocedor externo de escritorio usa `--observed-intent`. Lee una etiqueta numérica de la entrada estándar, la coteja con la expectativa independiente y ejecuta la misma comprobación de rutina y AAC. El resultado cubre el recorrido en la Mac entre el reconocedor externo y el despacho. Esa entrada solo existe en el arnés de escritorio. `phoneme_replay_lab.py --episode ID_RESERVADO` emite la etiqueta que el respaldo congelado predijo después de la captura simulada, conservando cualquier decisión del compacto.

```bash
experiments/esmx-pilot/routine_host.sh "$voice_dir" --observed-intent time < "$observed_label_file"
```

Para una traza de un episodio acertado, `BB_ROUTINE_HOST_AAC_OUT` indica un archivo nuevo donde el arnés guarda los bytes de todas las reproducciones AAC del episodio, en el orden original. `BB_ROUTINE_HOST_BINARY_HASH_OUT` indica un archivo nuevo para el SHA-256 del ejecutable de esa corrida. Las dos variables son opcionales. Los archivos existentes provocan un error y conservan sus datos.

Las pruebas sustituyen los servicios del SDK, reloj, comunicación, reproductor, entrada de escucha en inglés, archivo del efecto y generador aleatorio del ESP32. El punto de entrada de la tabla OEM se sustituye por las tres filas ejercitadas. El firmware ESP32-S3 contiene su tabla completa. El reloj y el resultado del dado son deterministas en la prueba. El alcance del resultado es la rutina y sus solicitudes de reproducción, con verificación separada del códec mediante `voice_validate.sh`. El micrófono, el procesamiento de señal del ESP32, la comunicación por radio, el altavoz y el funcionamiento con el mando siguen pendientes de comprobación física.

## Repetir un panel sellado de voces es-MX

Usa `sealed_episode.py` con tres ejecutables de escritorio compilados desde el mismo código y perfil de reconocimiento. Genera el reconocedor con `build_host.sh`, la captura con `build_host.sh … --capture` y el ejecutable de rutinas con `routine_host.sh` y `BB_ROUTINE_HOST_EXECUTABLE_OUT`. Registra el commit y las opciones de compilación por separado. Los binarios reciben el mismo PCM; el resultado de captura debe coincidir con el que da el reconocedor cuando recibe su PCM retenido.

```bash
python3 experiments/esmx-pilot/sealed_episode.py \
  --manifest "$panel" --manifest-sha256 "$panel_sha" \
  --recognizer "$recognizer" --capture "$capture" --routine "$routine" \
  --ffmpeg "$ffmpeg" --clock-dir "$clock_model"
```

El JSON local usa `version: 1`, `source_sha` del firmware evaluado, `artifact_sha256` con las claves `recognizer`, `capture`, `routine` y `ffmpeg`, y `clock_files_sha256` con todos los nombres y huellas de archivos regulares del banco CI16. Para un perfil sin el banco, usa un mapa vacío y omite `--clock-dir`. `adjustment` contiene los pares `source_group` y `speaker_group` que participaron en ajuste. Cada entrada de `episodes` contiene `wav` (nombre local junto al JSON), `wav_sha256`, `locale: "es-MX"`, `role` (`heldout` o `regression`), `source_id`, `source_group`, `speaker_group`, `rights` y `expected` (0, 1, 3, 5 o 6 respectivamente para abstención, Mantequilla, Chiste, Hora y D6).

Comprueba el origen de la grabación, el español mexicano, el permiso de uso y el turno completo antes de sellar el panel. Genera su SHA-256 antes de ejecutar inferencia. El arnés solo verifica los metadatos declarados, coteja las huellas y detecta traslapes declarados de locutor o fuente. Usa `regression` para casos conocidos. La grabación del propietario queda excluida.

El recibo muestra por separado audio crudo y retenido. Una captura incompleta o un audio inválido se registra como inconcluso. Un acierto positivo exige la rutina original, sus datos y la decodificación del AAC. La única salida publicada contiene números de episodio, resultados y huellas. El proceso devuelve 0 si todas las vistas coinciden, 1 si falta una vista válida o hay un desacierto y 2 si falla el manifiesto, las huellas o un ejecutable. Los WAV y el manifiesto quedan locales.
