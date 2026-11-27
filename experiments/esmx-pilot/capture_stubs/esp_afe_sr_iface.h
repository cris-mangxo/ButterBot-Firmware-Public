#pragma once

#include <cstdint>

enum vad_state_t { VAD_SILENCE, VAD_SPEECH };

struct afe_fetch_result_t {
    int16_t* data;
    int data_size;
    int16_t* vad_cache;
    int vad_cache_size;
    vad_state_t vad_state;
    int ret_value;
};
