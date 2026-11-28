#pragma once

#include <pocketsphinx.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct bb_ps_cepstra;
struct bb_ps_cepstra* bb_ps_capture_cepstra(ps_decoder_t* decoder, const int16_t* pcm, size_t samples);
int bb_ps_replay_cepstra(ps_decoder_t* decoder, struct bb_ps_cepstra* cepstra);
void bb_ps_free_cepstra(struct bb_ps_cepstra* cepstra);

#ifdef __cplusplus
}
#endif
