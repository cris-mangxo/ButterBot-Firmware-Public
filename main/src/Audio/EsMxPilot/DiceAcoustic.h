#pragma once
#include <cstdint>
#include <pocketsphinx.h>
#include <span>

namespace EsMx::Pilot {

bool runAcousticSearch(ps_decoder_t* decoder, const char* search, std::span<const int16_t> pcm);
bool addAcousticDice(ps_decoder_t* decoder, const char* keywords);
bool confirmsAcousticDice(ps_decoder_t* decoder, std::span<const int16_t> pcm);

}
