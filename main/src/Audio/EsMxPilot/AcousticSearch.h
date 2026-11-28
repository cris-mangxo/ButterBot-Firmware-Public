#pragma once

#include <cstdint>
#include <pocketsphinx.h>
#include <span>

namespace EsMx::Pilot {

// Both searches run only on the same recognition worker.
ps_decoder_t* sharedAcousticDecoder();
bool decodeAcoustic(ps_decoder_t* decoder, const char* search, std::span<const int16_t> pcm);

}
