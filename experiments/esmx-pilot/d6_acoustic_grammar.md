# Comparación acústica D6 en escritorio

La sonda conserva como referencia el perfil es-MX completo del commit
`f25ae6a`. Ejecuta el reconocedor nativo y el capturador de ese perfil.
Abre la gramática solo en las tomas elegibles para respaldo cuyo resultado
final sigue vacío. Cuenta las activaciones para PCM crudo, captura nominal
simulada y coincidencia entre ambas vistas. Esta última comparación es un
diagnóstico de estabilidad, ya que el despacho trabaja con el PCM retenido.

- `d6_acoustic_grammar_lab.py` recibe el modelo local de PocketSphinx 5.0.4,
  el diccionario Mexbet comprimido, los binarios de referencia del perfil
  activo, el histórico y el corpus CHM150 con su división sellada. Usa el
  perfil acústico original en `BB_ESMX_CLOCK_ACOUSTIC_DIR`. La salida pública
  de la herramienta son conteos por grupo. El directorio de salida nuevo
  contiene hipótesis por toma y se conserva local.
- `d6_acoustic_pack4.py` transforma un CI32-Q8 derivado localmente en dos
  variantes de cuatro bits. Conserva los archivos auxiliares y prepara
  matrices expandidas para PocketSphinx y sus equivalentes empacadas para
  medir tamaño. La variante `global` usa una escala por dimensión y la
  variante `state` usa una por estado y dimensión.
- `d6_acoustic_clock_compat.py` compara el perfil original y el modelo de
  escritorio alternativo sobre el panel Heroico ya atestado. Copia
  `clock.dict` del modelo de referencia al directorio alternativo antes de
  ejecutar la comparación.

`--help` detalla las rutas de entrada. Los tres programas son herramientas
de laboratorio. Los resultados de esta rama dejan el perfil instalado en
el código de referencia. Las frases históricas son regresiones conocidas.
Las hipótesis y las grabaciones permanecen locales.
