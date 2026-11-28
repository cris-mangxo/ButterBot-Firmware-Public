#include "cepstra_bridge.h"

#include "pocketsphinx_internal.h"
#include <stdlib.h>
#include <string.h>

#ifdef FIXED_POINT
#error "The cepstra experiment requires the pinned floating-point PocketSphinx build."
#endif

struct bb_ps_cepstra {
    float32* original;
    float32* work;
    float32** rows;
    int width;
    int frames;
};

static void* alloc(size_t bytes) {
#ifdef ESP_PLATFORM
    extern void* bb_clock_malloc(size_t bytes);
    return bb_clock_malloc(bytes);
#else
    return malloc(bytes);
#endif
}

void bb_ps_free_cepstra(struct bb_ps_cepstra* cepstra) {
    if (cepstra == NULL)
        return;
    free(cepstra->rows);
    free(cepstra->work);
    free(cepstra->original);
    free(cepstra);
}

struct bb_ps_cepstra* bb_ps_capture_cepstra(ps_decoder_t* decoder, const int16_t* pcm, size_t samples) {
    enum { MaxFrames = 512, MaxWidth = 32 };
    int frames = 0, tail = 0;
    if (decoder == NULL || pcm == NULL || samples == 0 ||
        (decoder->acmod->state != ACMOD_IDLE && decoder->acmod->state != ACMOD_ENDED))
        return NULL;
    ps_config_t* config = decoder->acmod->config;
    // Such models change the front end's state between decodes or need a raw PCM log.
    if (ps_config_bool(config, "dither") || ps_config_bool(config, "remove_noise") ||
        ps_config_str(config, "rawlogdir"))
        return NULL;

    fe_t* fe = decoder->acmod->fe;
    const int width = fe_get_output_size(fe);
    if (fe_start_utt(fe) < 0 || width < 1 || width > MaxWidth ||
        fe_process_frames(fe, NULL, &samples, NULL, &frames) < 0 || frames < 1 || frames >= MaxFrames)
        return NULL;

    const size_t buffer_bytes = (size_t)(frames + 1) * (size_t)width * sizeof(float32);
    struct bb_ps_cepstra* cepstra = alloc(sizeof(*cepstra));
    if (cepstra == NULL)
        return NULL;
    *cepstra = (struct bb_ps_cepstra){.width = width, .original = alloc(buffer_bytes),
        .work = alloc(buffer_bytes), .rows = alloc((size_t)(frames + 1) * sizeof(float32*))};
    if (cepstra->original == NULL || cepstra->work == NULL || cepstra->rows == NULL) {
        bb_ps_free_cepstra(cepstra);
        return NULL;
    }
    for (int frame = 0; frame <= frames; ++frame)
        cepstra->rows[frame] = cepstra->original + (size_t)frame * (size_t)width;
    if (fe_process_frames(fe, &pcm, &samples, cepstra->rows, &frames) < 0 || samples != 0 ||
        fe_end_utt(fe, cepstra->rows[frames], &tail) < 0) {
        bb_ps_free_cepstra(cepstra);
        return NULL;
    }
    cepstra->frames = frames + tail;
    for (int frame = 0; frame < cepstra->frames; ++frame)
        cepstra->rows[frame] = cepstra->work + (size_t)frame * (size_t)width;
    return cepstra;
}

int bb_ps_replay_cepstra(ps_decoder_t* decoder, struct bb_ps_cepstra* cepstra) {
    if (decoder == NULL || cepstra == NULL || cepstra->width != fe_get_output_size(decoder->acmod->fe))
        return -1;
    // Complete-utterance normalization mutates the input rows in place.
    memcpy(cepstra->work, cepstra->original, (size_t)cepstra->frames * (size_t)cepstra->width * sizeof(float32));
    return ps_process_cep(decoder, cepstra->rows, cepstra->frames, 0, 1);
}
