#ifdef BB_ESMX_DICE_ACOUSTIC
#include "DiceAcoustic.h"

#include "DiceAcousticAssets.h"
#include <algorithm>
#include <array>
#include <optional>
#include <string_view>

namespace EsMx::Pilot {
namespace {

constexpr char SpotSearch[] = "dice_spot";
constexpr char AffirmativeSearch[] = "dice_affirmative";
constexpr char NegativeSearch[] = "dice_negative";
constexpr int PositiveFloor = -2350;
constexpr int NegativeFloor = -2250;
constexpr int TailFloor = -2200;
constexpr int OuterFrames = 48;

enum class Kind { Unknown, Order, PoliteTail, Denial, EarlierDenial, LaterDenial, OtherDie };

template<size_t N>
bool has(std::string_view word, const char* const (&list)[N]) {
    return std::find(std::begin(list), std::end(list), word) != std::end(list);
}

Kind classifyPhrase(std::string_view word) {
    using namespace DiceAcousticAssets;
    if (has(word, Orders)) return Kind::Order;
    if (has(word, Tails)) return Kind::PoliteTail;
    if (has(word, Denials)) return Kind::Denial;
    if (has(word, EarlierDenials)) return Kind::EarlierDenial;
    if (has(word, LaterDenials)) return Kind::LaterDenial;
    if (has(word, OtherDice)) return Kind::OtherDie;
    return Kind::Unknown;
}

struct Hit {
    Kind kind;
    int start;
    int end;
    int score;
};

struct Speech {
    int first;
    int last;
};

std::optional<Speech> speechSpan(std::span<const int16_t> pcm) {
    constexpr size_t Frame = 160;
    constexpr uint64_t EnergyFloor = Frame * uint64_t{180} * 180;
    std::optional<Speech> result;
    for (size_t offset = 0; offset < pcm.size(); offset += Frame) {
        const auto samples = pcm.subspan(offset, std::min(Frame, pcm.size() - offset));
        uint64_t energy = 0;
        for (const int32_t sample : samples) energy += uint64_t(sample * sample);
        if (energy >= EnergyFloor) {
            const int frame = static_cast<int>(offset / Frame);
            if (!result) result = Speech{frame, frame};
            result->last = frame;
        }
    }
    return result;
}

bool covers(Speech span, int first, int last) {
    return first - span.first <= OuterFrames && span.last - last <= OuterFrames;
}

bool confirmsOrder(std::span<const Hit> hits, Speech speech) {
    for (const auto& order : hits) {
        if (order.kind != Kind::Order || order.score < PositiveFloor) continue;
        int end = order.end;
        bool opposed = false;
        for (const auto& hit : hits) {
            if (hit.kind == Kind::PoliteTail && hit.score >= TailFloor &&
                hit.start >= order.end - 45 && hit.start <= order.end + 16 && hit.end <= order.end + 82) {
                end = std::max(end, hit.end);
            } else if (hit.score >= NegativeFloor &&
                       ((hit.kind == Kind::Denial && covers(speech, hit.start, hit.end)) ||
                        (hit.kind == Kind::EarlierDenial && hit.start <= order.start + 12) ||
                        (hit.kind == Kind::LaterDenial && hit.start >= order.end - 6) ||
                        hit.kind == Kind::OtherDie)) {
                opposed = true;
            }
        }
        if (!opposed && covers(speech, order.start, end)) return true;
    }
    return false;
}

bool hasWholeHypothesis(ps_decoder_t* decoder, int32& score) {
    const char* result = ps_get_hyp(decoder, &score);
    if (!result) return false;
    const std::string_view text(result);
    constexpr std::string_view marker = " de seis";
    const auto unit = text.find(marker);
    if (unit == std::string_view::npos) return false;
    auto rest = text.substr(unit + marker.size());
    if (rest.starts_with(" caras") || rest.starts_with(" lados")) rest.remove_prefix(6);
    return rest.empty() || rest == " por favor" || rest == " porfa";
}

}

bool runAcousticSearch(ps_decoder_t* decoder, const char* search, std::span<const int16_t> pcm) {
    if (ps_activate_search(decoder, search) < 0 || ps_start_utt(decoder) < 0) return false;
    const auto processed = ps_process_raw(decoder, pcm.data(), pcm.size(), 0, 1);
    const auto ended = ps_end_utt(decoder);
    return processed >= 0 && ended >= 0;
}

bool addAcousticDice(ps_decoder_t* decoder, const char* keywords) {
    using namespace DiceAcousticAssets;
    return ps_add_kws(decoder, SpotSearch, keywords) == 0 &&
           ps_add_jsgf_string(decoder, AffirmativeSearch, Affirmative) == 0 &&
           ps_add_jsgf_string(decoder, NegativeSearch, Negative) == 0;
}

bool confirmsAcousticDice(ps_decoder_t* decoder, std::span<const int16_t> pcm) {
    const auto speech = speechSpan(pcm);
    if (!speech || !runAcousticSearch(decoder, SpotSearch, pcm)) return false;
    std::array<Hit, 80> hits;
    size_t count = 0;
    for (auto* segment = ps_seg_iter(decoder); segment; segment = ps_seg_next(segment)) {
        const char* phrase = ps_seg_word(segment);
        if (!phrase) continue;
        std::string_view word(phrase);
        if (word.ends_with(' ')) word.remove_suffix(1);
        const auto kind = classifyPhrase(word);
        if (kind == Kind::Unknown) continue;
        if (count == hits.size()) {
            ps_seg_free(segment);
            return false;
        }
        auto& hit = hits[count++];
        hit.kind = kind;
        ps_seg_frames(segment, &hit.start, &hit.end);
        hit.score = ps_seg_prob(segment, nullptr, nullptr, nullptr);
    }
    if (!confirmsOrder(std::span(hits).first(count), *speech) ||
        !runAcousticSearch(decoder, AffirmativeSearch, pcm)) return false;

    int32 positiveScore = 0;
    int32 negativeScore = 0;
    if (!hasWholeHypothesis(decoder, positiveScore) ||
        !runAcousticSearch(decoder, NegativeSearch, pcm) ||
        !hasWholeHypothesis(decoder, negativeScore)) return false;

    // The negative grammar includes an explicit prefix. Ignore incomplete
    // paths, and preserve the reference terminal veto before reaching here.
    constexpr int MaximumNegativeDeficitPerFrame = 2;
    return int64_t(positiveScore) - negativeScore >
            MaximumNegativeDeficitPerFrame * ps_get_n_frames(decoder);
}

}
#endif
