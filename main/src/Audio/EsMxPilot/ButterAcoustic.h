#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace EsMx::Pilot {

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
    ButterSegment affirmArticle;
    int affirmScore = 0;
    int opposeScore = 0;
    int frames = 0;
    int lastSpeech = -1;
    bool affirmative = false;
    bool opposing = false;
    bool complete = false;
    bool ranFsg = false;
};

ButterEvidence measureAcousticButter(std::span<const int16_t> pcm);
bool acceptsAcousticButter(const ButterEvidence& evidence);
bool recognizesAcousticButter(std::span<const int16_t> pcm);

}
