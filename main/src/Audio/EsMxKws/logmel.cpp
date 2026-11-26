#include "logmel.h"

#include <cmath>

#include "cuantizar.h"

namespace bb {
namespace asr {
namespace {

constexpr double kPi = 3.14159265358979323846;

double mel_of(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
double mel_inverse(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

// FFT compleja radix-2 (decimación en tiempo) in situ, permutación de bits sin tabla.
void fft512(LogMel* lm) {
  float* re = lm->re;
  float* im = lm->im;
  for (int i = 1, j = 0; i < kFft; ++i) {
    int bit = kFft >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const float tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const float ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  for (int size = 2; size <= kFft; size <<= 1) {
    const int half = size / 2;
    const int step = kFft / size;
    for (int start = 0; start < kFft; start += size) {
      for (int k = 0; k < half; ++k) {
        const float c = lm->tw_cos[k * step];
        const float s = lm->tw_sin[k * step];
        const int i = start + k;
        const int j = i + half;
        const float tr = re[j] * c - im[j] * s;
        const float ti = re[j] * s + im[j] * c;
        re[j] = re[i] - tr;
        im[j] = im[i] - ti;
        re[i] = re[i] + tr;
        im[i] = im[i] + ti;
      }
    }
  }
}

}  // namespace

bool logmel_init(LogMel* lm) {
  lm->ready = false;
  for (int n = 0; n < kWin; ++n) lm->hann[n] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * n / kWin));
  for (int k = 0; k < kFft / 2; ++k) {
    lm->tw_cos[k] = static_cast<float>(std::cos(2.0 * kPi * k / kFft));
    lm->tw_sin[k] = static_cast<float>(-std::sin(2.0 * kPi * k / kFft));
  }
  double edges[kBands + 2];
  const double lo = mel_of(kFMinHz);
  const double hi = mel_of(kFMaxHz);
  for (int i = 0; i < kBands + 2; ++i) edges[i] = mel_inverse(lo + (hi - lo) * i / (kBands + 1));
  int used = 0;
  for (int m = 0; m < kBands; ++m) {
    const double a = edges[m], c = edges[m + 1], b = edges[m + 2];
    int start = -1, len = 0;
    lm->band_off[m] = static_cast<uint16_t>(used);
    for (int k = 0; k < kBins; ++k) {
      const double f = static_cast<double>(k) * kSampleRate / kFft;
      double w;
      if (a < f && f <= c) {
        w = (f - a) / (c - a);
      } else if (c < f && f < b) {
        w = (b - f) / (b - c);
      } else {
        continue;
      }
      if (w > 0.0) {
        if (start < 0) start = k;
        if (used >= kMaxWeights) return false;
        lm->weights[used++] = static_cast<float>(w);
        ++len;
      }
    }
    lm->band_start[m] = static_cast<uint16_t>(start < 0 ? 0 : start);
    lm->band_len[m] = static_cast<uint16_t>(len);
  }
  lm->ready = true;
  return true;
}

size_t logmel_num_frames(size_t n_samples) {
  return n_samples < static_cast<size_t>(kWin) ? 0 : 1 + (n_samples - kWin) / kHop;
}

void logmel_frame(LogMel* lm, const int16_t* window, float* out40) {
  for (int n = 0; n < kWin; ++n) lm->re[n] = static_cast<float>(window[n]) / 32768.0f * lm->hann[n];
  for (int n = kWin; n < kFft; ++n) lm->re[n] = 0.0f;
  for (int n = 0; n < kFft; ++n) lm->im[n] = 0.0f;
  fft512(lm);
  for (int m = 0; m < kBands; ++m) {
    float e = 0.0f;
    const float* w = lm->weights + lm->band_off[m];
    for (int j = 0; j < lm->band_len[m]; ++j) {
      const int k = lm->band_start[m] + j;
      e += w[j] * (lm->re[k] * lm->re[k] + lm->im[k] * lm->im[k]);
    }
    out40[m] = std::log(e + kFloor);
  }
}

int64_t logmel_int8(LogMel* lm, const int16_t* samples, size_t n_samples, double scale, int zero_point,
                    int8_t* out, size_t out_capacity) {
  if (!lm->ready || n_samples > kMaxSamples) return -1;
  const size_t frames = logmel_num_frames(n_samples);
  if (frames * kBands > out_capacity) return -1;
  float fila[kBands];
  for (size_t v = 0; v < frames; ++v) {
    logmel_frame(lm, samples + v * kHop, fila);
    for (int m = 0; m < kBands; ++m) out[v * kBands + m] = quantize_value(static_cast<double>(fila[m]), scale, zero_point);
  }
  return static_cast<int64_t>(frames * kBands);
}

}  // namespace asr
}  // namespace bb
