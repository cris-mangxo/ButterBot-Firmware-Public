#pragma once

#include <cstdint>
#include <span>

// Host-only tap. Enable before capture starts and read only after the worker joins.
void enablePcmProbe();
std::span<const int16_t> recordedPcm();
