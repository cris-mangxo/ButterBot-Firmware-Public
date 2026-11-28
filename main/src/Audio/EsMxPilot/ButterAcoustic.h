#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace EsMx::Pilot {

struct ButterGate {
    bool enabled = false;
    int keywordMinimum = 0;
    int onsetSlack = 0;
    int articleLead = 0;
    int endSlack = 0;
    int tailSlack = 0;
    int floorPerFrame = 0;
    int contrastPerFrame = 0;
    int oppositionKeywordMinimum = 0;
};

struct ButterSegment {
    int start = -1;
    int end = -1;
    int score = 0;
};

enum class ButterTerm { Object, ArticleObject, Opposition };
struct ButterHit {
    ButterTerm type;
    ButterSegment segment;
};

struct ButterEvidence {
    std::array<ButterHit, 32> hits{};
    std::size_t hitCount = 0;
    ButterSegment affirmObject;
    int affirmScore = 0;
    int opposeScore = 0;
    int frames = 0;
    int lastSpeech = -1;
    bool affirmative = false;
    bool opposing = false;
    bool complete = false;
};

ButterEvidence measureAcousticButter(std::span<const int16_t> pcm);
bool acceptsAcousticButter(const ButterEvidence& evidence, const ButterGate& gate);
bool recognizesAcousticButter(std::span<const int16_t> pcm);

}
