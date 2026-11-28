#pragma once

#if defined(BB_ESMX_D6_SCOPE_HOST) && !defined(ESP_PLATFORM)
#include <cstdint>
#include <pocketsphinx.h>
#include <span>

namespace EsMx::Pilot {

bool installDiceScopeSearches(ps_decoder_t* decoder, const char* keywords);
bool confirmsDiceScope(std::span<const int16_t> pcm);

}
#endif
