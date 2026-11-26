// Cuantización int8 y comparación descuantizada (P31). Espejo de experiments/asr/cuantizar.py.
// Solo cabecera, sin heap. Cuantiza CARACTERÍSTICAS; no es un modelo acústico cuantizado.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace bb {
namespace asr {

constexpr int kInt8Min = -128;
constexpr int kInt8Max = 127;
constexpr double kCompareRelEps = 1e-12;

// Redondeo al entero más cercano; las mitades se alejan de cero.
inline double round_half_away(double x) { return x >= 0 ? std::floor(x + 0.5) : -std::floor(-x + 0.5); }

inline int8_t quantize_value(double x, double scale, int zero_point) {
  double q = round_half_away(x / scale) + zero_point;
  if (q < kInt8Min) q = kInt8Min;
  if (q > kInt8Max) q = kInt8Max;
  return static_cast<int8_t>(q);
}

inline double dequantize_value(int8_t q, double scale, int zero_point) {
  return (static_cast<double>(q) - zero_point) * scale;
}

struct QTensor {
  const int8_t* values;
  size_t n;
  double scale;
  int zero_point;
};

struct CompareResult {
  bool overflow;  // un valor descuantizado o una diferencia no es finito: ni equivalente ni discrepancia (error)
  bool equivalent;
  bool length_mismatch;
  bool has_max;  // false si el prefijo común está vacío
  double max_abs_diff;
  int64_t first_mismatch_index;  // -1: no hay (solo con `equivalent`)
};

inline bool within_tolerance(double diff, double tol) {
  if (diff <= tol) return true;
  const double mag = std::fabs(diff) > std::fabs(tol) ? std::fabs(diff) : std::fabs(tol);
  return std::fabs(diff - tol) <= kCompareRelEps * mag;  // equivalente a math.isclose(rel_tol=1e-12, abs_tol=0)
}

inline CompareResult compare(const QTensor& ref, const QTensor& cand, double tol) {
  CompareResult r{false, true, false, false, 0.0, -1};
  // Como Python: se descuantizan TODOS los elementos de ambos tensores antes de comparar; uno no finito es error.
  for (size_t i = 0; i < ref.n; ++i)
    if (!std::isfinite(dequantize_value(ref.values[i], ref.scale, ref.zero_point))) {
      r.overflow = true;
      r.equivalent = false;
      return r;
    }
  for (size_t i = 0; i < cand.n; ++i)
    if (!std::isfinite(dequantize_value(cand.values[i], cand.scale, cand.zero_point))) {
      r.overflow = true;
      r.equivalent = false;
      return r;
    }
  const size_t n = ref.n < cand.n ? ref.n : cand.n;
  r.has_max = n != 0;
  int64_t first = -1;
  for (size_t i = 0; i < n; ++i) {
    const double d = std::fabs(dequantize_value(ref.values[i], ref.scale, ref.zero_point) -
                               dequantize_value(cand.values[i], cand.scale, cand.zero_point));
    if (!std::isfinite(d)) {  // la resta de dos finitos puede desbordar: error, nunca `inf <= inf`
      r.overflow = true;
      r.equivalent = false;
      return r;
    }
    if (d > r.max_abs_diff) r.max_abs_diff = d;
    if (first < 0 && !within_tolerance(d, tol)) first = static_cast<int64_t>(i);
  }
  if (ref.n != cand.n) {
    r.equivalent = false;
    r.length_mismatch = true;
    r.first_mismatch_index = first >= 0 ? first : static_cast<int64_t>(n);
  } else if (first >= 0) {
    r.equivalent = false;
    r.first_mismatch_index = first;
  }
  return r;
}

}  // namespace asr
}  // namespace bb
