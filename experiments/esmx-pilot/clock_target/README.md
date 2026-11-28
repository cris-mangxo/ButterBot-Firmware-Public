# Hora acústica en C++ y ESP32-S3

El perfil opcional `BB_ESMX_CLOCK_ACOUSTIC` conserva los reconocedores previos.
Solo intenta «qué hora es» después de `NoMatch` y de que el respaldo
fonético se abstenga. El binario normal conserva el perfil anterior.

## Preparación local

1. Usa PocketSphinx 5.0.4 de cmusphinx y prepara dos directorios nuevos con
   `build_pocketsphinx.sh CHECKOUT NUEVO_DIRECTORIO host` y
   `build_pocketsphinx.sh CHECKOUT NUEVO_DIRECTORIO esp32s3` con ESP-IDF 5.5.3
   ya cargado. El script extrae el commit fijado y aplica el parche exclusivamente
   a la copia Xtensa.
2. Usa el modelo, el diccionario y los informes de la investigación CIEMPIESS.
   El modelo de 90 estados y 16 gaussianas por estado usa las tablas de ocho bits
   verificadas. Ejecuta `../clock_model_pack.py` con los informes de huellas
   para crear localmente `ClockModelAssets.h`. Conserva los activos
   CIEMPIESS y las grabaciones en directorios fuera de Git.
3. Ejecuta `prepare_cases.py` con el panel Heroico y el informe acústico para
   crear `ClockTargetCases.h` en un directorio local. Compara los valores
   esperados del arnés contra el registro de referencia antes de evaluar un
   banco nuevo.

## Recorrido en Mac

Usa los comandos `../build_host.sh` y `../routine_host.sh` de la manera habitual.
Activa `BB_ESMX_JOKE_FALLBACK=1`, `BB_ESMX_CLOCK_FALLBACK=1` y
`BB_ESMX_CLOCK_ACOUSTIC=1`. Indica los directorios `source/include` y
`build/include` de PocketSphinx en `BB_ESMX_SPHINX_INCLUDE` y
`BB_ESMX_SPHINX_GENERATED_INCLUDE`. Pasa el archivo
`build/libpocketsphinx.a` en `BB_ESMX_SPHINX_LIBRARY`. Al ejecutar los
binarios, `BB_ESMX_CLOCK_ACOUSTIC_DIR` apunta a los siete archivos acústicos
expandidos y a `clock.dict`. El arnés de rutinas puede conservar su ejecutable
con `BB_ROUTINE_HOST_EXECUTABLE_OUT` para repetir el panel.

`../clock_acoustic_replay.py` valida el panel Heroico, el PCM retenido, el
despacho original, las tramas AAC y el histórico. Una salida de etiqueta
numérica se verifica por separado del arnés de rutina. El arnés de rutina
recibe PCM y llama al mismo reconocedor C++ antes de iniciar la rutina.
`../clock_acoustic_controls.py` comprueba controles adicionales crudos.
Las frases del panel Heroico ya eran conocidas al diseñar la gramática.

## Compilación y emulación

Los argumentos comunes de CMake son `BB_ESMX_ASSETS`, la tríada
`BB_ESMX_SPHINX_LIBRARY`, `BB_ESMX_SPHINX_INCLUDE` y
`BB_ESMX_SPHINX_GENERATED_INCLUDE`, y `BB_ESMX_CLOCK_PACK` con el
directorio de `ClockModelAssets.h`. Usa la biblioteca para ESP32-S3.

- En el firmware completo activa las opciones CMake `BB_ESMX_PILOT`,
  `BB_ESMX_JOKE_FALLBACK`, `BB_ESMX_CLOCK_FALLBACK` y
  `BB_ESMX_CLOCK_ACOUSTIC`. Usa la configuración del candidato optimizado
  por tamaño.
- En este arnés indica también `BB_ESMX_CLOCK_TARGET_CASES` con la
  cabecera de muestras. Usa `IDF_TARGET=esp32s3` y un directorio de build
  y archivo SDK separados. El proyecto configura PSRAM octal y una pila
  de 8 KiB.
- Genera un archivo de imagen de 4 MiB para QEMU desde los binarios con
  `esptool merge_bin`. En el emulador usa `-M esp32s3 -m 8M` y
  `-global driver=ssi_psram,property=is_octal,value=true`. El informe
  requiere `BB_CLOCK_MODEL_OK`, siete resultados y `BB_CLOCK_PASS`.
  Registra calentamiento, tiempos por turno y memoria. [PSRAM octal en QEMU](https://github.com/espressif/esp-toolchain-docs/blob/main/qemu/esp32s3/README.md#enabling-octal-psram).

El directorio virtual contiene únicamente los activos de Hora en lectura.
El emulador verifica que los bytes reconstruidos sean los mismos que en
escritorio. El decodificador calienta en una tarea independiente y la
escucha solo lo consulta si está listo. El enlace y la emulación tienen
un alcance de escritorio. La latencia junto con el resto de servicios
y el margen de memoria durante el funcionamiento físico quedan por medir.

## D6 acústico experimental

`BB_ESMX_DICE_ACOUSTIC` reutiliza el decodificador de Hora. La búsqueda
compacta exige una orden completa de un solo dado de seis y un contraste
de afirmación y negación del turno. Solo se ejecuta ante un conflicto
débil de negación de D6. Los vetos terminales y las decisiones
confirmadas tienen prioridad. La opción admite abstención y queda
desactivada de forma predeterminada.

Genera los activos de D6 en un directorio nuevo con
`../dice_acoustic_pack.py --source DICCIONARIO_CIEMPIESS_GZ --clock CLOCK_DICT --output NUEVO_DIRECTORIO`.
Este comando comprueba el diccionario de Hora fijado, corrige las entradas
MEXBET de «usted» y «lance», y conserva las pronunciaciones que comparte
con Hora. Coloca en ese directorio el `ClockModelAssets.h` local y los
archivos expandidos del mismo modelo de Hora. Activa
`BB_ESMX_DICE_ACOUSTIC=ON` en el firmware completo y usa ese directorio
en `BB_ESMX_CLOCK_PACK`. En Mac activa `BB_ESMX_DICE_ACOUSTIC=1` y usa
ese directorio en `BB_ESMX_CLOCK_ACOUSTIC_DIR` al compilar y al ejecutar.

Para probar una orden de D6 con una entrada local de 16 kHz, mono, PCM16,
usa `../routine_host.sh DIRECTORIO_VOZ DIRECTORIO_MODELOS dice6 < entrada.pcm`.
El arnés debe recibir audio real y devolver intención 6, llamada a la
rutina original, mensaje comprobado y los clips AAC esperados. También
debe comprobarse que el reconocedor se abstiene en las cancelaciones.
La opción no extiende por sí sola la cobertura a cualquier frase o a
otros tipos de dado.
