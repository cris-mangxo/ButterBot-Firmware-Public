#pragma once

#include <cstdint>
#include <span>

namespace EsMx::Pilot {

// Only call on a complete turn eligible for fallback.
bool matchesJokeFallback(std::span<const int16_t> pcm);

}
