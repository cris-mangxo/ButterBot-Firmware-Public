#ifdef BB_ESMX_COMPACT_VOICE
#include "SpeechGen.h"

#ifdef ESP_PLATFORM
#include "esp_log.h"
#endif

namespace {

void reportUnavailable(){
#ifdef ESP_PLATFORM
    ESP_LOGW("SpeechGen", "Text is outside the experimental es-MX voice catalogue");
#endif
}

}

SpeechGen::SpeechGen() = default;
SpeechGen::~SpeechGen() = default;

int SpeechGen::generate(std::span<int16_t> output, bool& last, TickType_t){
    return static_cast<int>(stream.generate(output, last));
}

void SpeechGen::parseText(std::string text){
    if(!stream.start(text)){
        reportUnavailable();
    }
}

void SpeechGen::parsePhonemes(const std::string&){
    cancel();
    reportUnavailable();
}

void SpeechGen::parseSSML(std::string){
    cancel();
    reportUnavailable();
}

void SpeechGen::cancel(){
    stream.reset();
}
#endif
