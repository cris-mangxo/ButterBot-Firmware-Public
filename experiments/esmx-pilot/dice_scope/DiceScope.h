#pragma once

#if (defined(BB_ESMX_D6_SCOPE_HOST) && !defined(ESP_PLATFORM)) || defined(BB_ESMX_D6_SCOPE_BENCH)
#ifndef BB_ESMX_CLOCK_ACOUSTIC
#error "D6 scope requires the shared acoustic clock decoder"
#endif
#if defined(BB_ESMX_D6_SCOPE_BENCH) && !defined(ESP_PLATFORM)
#error "D6 scope benchmark requires ESP32-S3"
#endif
#include <cstdint>
#include <pocketsphinx.h>
#include <span>

namespace EsMx::Pilot {

bool installDiceScopeSearches(ps_decoder_t* decoder, const char* keywords);
bool confirmsDiceScope(std::span<const int16_t> pcm, bool requireFiniteVerb = false);

}
#endif
