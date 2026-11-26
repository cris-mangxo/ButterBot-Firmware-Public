// Front-end experimental `logmel-v1` (P31). Espejo de experiments/asr/logmel.py (float32 en lugar de float64).
// Sin heap en ninguna ventana: toda la memoria está en `LogMel`, que posee el llamador (estática o pila).
// Parámetros: Hann periódica 480, salto 160, FFT 512, 40 bandas mel HTK 125–7500 Hz, ln(e + 1e-6).
#pragma once
#include <cstddef>
#include <cstdint>

namespace bb {
namespace asr {

constexpr int kSampleRate = 16000;
constexpr int kWin = 480;
constexpr int kHop = 160;
constexpr int kFft = 512;
constexpr int kBins = kFft / 2 + 1;  // 257
constexpr int kBands = 40;
constexpr int kMaxWeights = 512;     // 455 usados (se comprueba en logmel_init)
constexpr float kFloor = 1e-6f;
constexpr double kFMinHz = 125.0;
constexpr double kFMaxHz = 7500.0;
constexpr double kScale = 0.125;
constexpr int kZeroPoint = 0;
constexpr size_t kMaxSamples = 10u * kSampleRate;  // límite C2

// Estado completo del front-end. sizeof(LogMel) está medido en el README (xtensa y host).
struct LogMel {
  float hann[kWin];
  float tw_cos[kFft / 2];
  float tw_sin[kFft / 2];
  float weights[kMaxWeights];
  uint16_t band_start[kBands];
  uint16_t band_len[kBands];
  uint16_t band_off[kBands];
  float re[kFft];
  float im[kFft];
  bool ready;
};

// Rellena las tablas. Devuelve false si la envolvente de pesos no cabe (nunca con los parámetros fijos).
bool logmel_init(LogMel* lm);

// Número de ventanas completas: 0 si n < 480, si no 1 + (n - 480) / 160.
size_t logmel_num_frames(size_t n_samples);

// Una ventana de 480 muestras int16 -> 40 valores ln(energía). Sin heap.
void logmel_frame(LogMel* lm, const int16_t* window, float* out40);

// Todas las ventanas, cuantizadas a int8 (ventana-mayor, banda-menor). `out_capacity` en valores.
// Devuelve el número de valores escritos, o -1 si n_samples > kMaxSamples o no cabe.
int64_t logmel_int8(LogMel* lm, const int16_t* samples, size_t n_samples, double scale, int zero_point,
                    int8_t* out, size_t out_capacity);

}  // namespace asr
}  // namespace bb
