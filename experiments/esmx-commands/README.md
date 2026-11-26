# Primer corte de órdenes es-MX

Prototipo de escritorio sobre la base de CircuitMess. La entrada puede ser una frase de texto o el WAV PCM de un turno (mono, 16 bits, 16 kHz y duración de 0.25 a 8 segundos). El reconocimiento de audio usa Whisper multilingüe en la Mac, en modo español y sin vocabulario forzado. El modelo cuantizado pesa 32,152,673 bytes y sirve de referencia de escritorio.

El texto completo pasa por `EsMx::matchCommand`. La lista de frases cubre las dos rutinas de mantequilla, chiste, dato curioso, objeto, hora, dado genérico, D6, D100 y Bola de la suerte. Se admite un conjunto pequeño de variantes. Normaliza mayúsculas, acentos de las vocales precompuestas y espacios. El envoltorio de escritorio convierte Unicode a NFC. El cotejo de un turno completo permite rechazar negaciones transcritas, órdenes concatenadas y palabras sueltas ajenas a la tabla. Un resultado de varias secciones del transcriptor se rechaza completo. Una transcripción errónea puede cambiar la intención. La prueba con grabaciones del hogar medirá esos errores.

`EsMx::routineForTranscript` usa los pares de escenario y parámetro del fabricante y delega en `routineForScenario`. El ejecutable de escritorio reporta el par y el verificador lo coteja contra la tabla real compilada para Xtensa. La rutina de dado sin número conserva el parámetro `None`. La fábrica determina qué rutina usa. El despacho habitual de `ListenState` conserva el flujo de fábrica. La futura integración necesita un reconocedor de destino y su conexión al turno de escucha, además de pruebas de cancelación, temporización y memoria. La orden «deja de bailar» pertenece a una sesión dentro de la rutina de baile y queda fuera de este conjunto.

## Reproducir en la Mac

Desde la raíz del fork. Los ejemplos guardan binarios y modelos en un directorio temporal. Requiere Apple clang con C++20 y Python 3. El modo audio también usa `whisper-cli`, `say` y `afconvert`.

```fish
set -l lab (mktemp -d)
c++ -std=c++20 -Wall -Wextra -Werror \
  -I experiments/esmx-commands/host -I main/src \
  -I components/ButterBot-Common/src -I components/CMF/src \
  experiments/esmx-commands/router.cpp main/src/Audio/EsMxCommand.cpp \
  -o "$lab/router"

python3 -B experiments/esmx-commands/recognize.py \
  --router "$lab/router" --text 'Aviéntate un chiste.'
python3 -B experiments/esmx-commands/smoke.py --router "$lab/router"

curl --fail --location --output "$lab/ggml-tiny-q5_1.bin" \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/f281eb4/ggml-tiny-q5_1.bin
printf '%s  %s\n' '818710568da3ca15689e31a743197b520007872ff9576237bda97bd1b469c3d7' \
  "$lab/ggml-tiny-q5_1.bin" | shasum -a 256 -c -
python3 -B experiments/esmx-commands/smoke.py \
  --router "$lab/router" --model "$lab/ggml-tiny-q5_1.bin"
```

La prueba de audio sintetiza las frases de `cases.json` usando Paulina (es-MX) y añade silencio. El JSON conserva todos los aciertos y desaciertos. El proceso devuelve 1 si hay un desacierto. Para una grabación local propia, se puede usar `recognize.py --wav <archivo> --model <modelo> --router <binario>`. Devuelve la evaluación de la intención junto con el escenario y el parámetro cuando coincide.

Para comprobar las fábricas, compilar con ESP-IDF 5.5.3 y usar `verify_factories.py --router <binario> --build <directorio-build> --objdump <xtensa-esp32s3-elf-objdump>` con el Python del SDK y `pyelftools`. Este comando inspecciona la referencia compilada al dispatcher y las fábricas de la tabla original. El host usa una cabecera mínima para declarar los niveles de log de ESP, y la compilación Xtensa usa las cabeceras reales del SDK.
