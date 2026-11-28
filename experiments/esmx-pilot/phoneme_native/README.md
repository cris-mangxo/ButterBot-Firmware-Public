# Banco de inferencia nativa del estudiante fonético

Este banco comparte el kernel C++ entre Mac y un proyecto ESP-IDF aislado para QEMU. Consume tramas int8 de 20 bandas en orden de trama, como las de `word-frames`. Lee los pesos int8 por canal, recupera sus valores float con las escalas selladas y usa activaciones float. Reproduce las salidas de fonemas, intención y negación del estudiante PyTorch.

La prueba verifica la aritmética sobre matrices artificiales y rasgos sintéticos sellados. El entero que devuelve el ejecutable corresponde a la guardia del estudiante base. Los filtros de captura, CTC, contexto y despacho se ejecutan por separado en sus respectivos arneses. Los arreglos artificiales de ceros son pruebas de igualdad del kernel y no representan audio aceptado por el robot.

## Mac

Se usa el Python de entrenamiento con las dependencias ya instaladas. `python_deps` apunta al directorio local de dependencias de la prueba fonética si éstas se instalaron fuera del entorno de entrenamiento.

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$python_deps"
bash experiments/esmx-pilot/phoneme_native/check_host.sh \
  "$training_python" "$bank" "$synthetic_features"
```

Para usar el ejecutable con otras tramas, genera un encabezado local y compila los mismos archivos.

```sh
native=experiments/esmx-pilot/phoneme_native
"$training_python" -B "$native/export_bank.py" "$bank" "$scratch/student_bank.h"
c++ -std=c++20 -O2 -ffp-contract=off -I"$native" -I"$scratch" \
  "$native/Kernel.cpp" "$native/main.cpp" -o "$scratch/student"
"$scratch/student" < "$scratch/frames-int8.raw"
```

`--logits` devuelve, en este orden, 32 valores float32 por trama para fonemas, cinco valores float32 para intenciones, un valor float32 para negación y un byte con el identificador de intención. El transporte se verifica en Mac arm64 con orden little-endian.

## ESP32-S3 emulado

Después de cargar el entorno ESP-IDF del workspace, agrega a `PATH` el directorio local que contiene `qemu-system-xtensa` y crea un directorio de salida vacío. QEMU genera su propia imagen de emulación en ese directorio.

```sh
native=experiments/esmx-pilot/phoneme_native
bash "$native/build_qemu.sh" "$training_python" "$bank" "$output" "$synthetic_features"
IDF_COMPONENT_MANAGER=0 idf.py -C "$native/qemu" -B "$output/build" \
  -D "SDKCONFIG=$output/sdkconfig" -D IDF_TARGET=esp32s3 \
  -D "PHONEME_ASSETS=$output/assets" qemu --qemu-extra-args="-nic none"
```

La línea `PHONEME_DONE` informa el número de pruebas y el número de discrepancias. Las líneas `PHONEME_CASE` incluyen el tiempo que reporta el emulador. El tiempo real en el robot y el incremento del ejecutable de fábrica requieren medición propia.

Si guardas la salida en un registro, se puede reducir a un informe que sella el banco, el binario y la imagen emulada.

```sh
"$training_python" -B "$native/summarize_qemu.py" "$output/qemu.log" "$bank/report.json"   "$output/build/ButterBotPhonemeNativeBench.bin"   --emulated-flash "$output/build/qemu_flash.bin"
```
