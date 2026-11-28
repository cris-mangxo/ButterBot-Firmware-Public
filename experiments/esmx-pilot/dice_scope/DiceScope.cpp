#if (defined(BB_ESMX_D6_SCOPE_HOST) && !defined(ESP_PLATFORM)) || defined(BB_ESMX_D6_SCOPE_BENCH)
#include "DiceScope.h"
#include "DiceScopeAssets.h"
#include "Audio/EsMxPilot/AcousticSearch.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#ifdef BB_ESMX_D6_SCOPE_TRACE
#include <cstdio>
#endif
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>
#include <sys/stat.h>
#include <tuple>

namespace EsMx::Pilot {
namespace {
constexpr int Frame = 160;
constexpr int MaxFrames = 400;
constexpr char Spot[] = "dice_r38_spot";
constexpr int BodyFloor = -2400;
constexpr int MaxStartGap = 24;
constexpr int MaxEndGap = 56;
constexpr int MentionFloor = -1900;
constexpr double VetoMargin = -1.0;

struct Span {
    int first;
    int last;
    int count;
};

double frameRms(std::span<const int16_t> frame) {
    uint64_t sum = 0;
    for (const int32_t sample : frame) sum += uint64_t(sample * sample);
    return std::sqrt(static_cast<double>(sum) / Frame);
}

std::optional<Span> voicedSpan(std::span<const int16_t> pcm) {
    const auto count = static_cast<int>(pcm.size() / Frame);
    if (count <= 0 || count > MaxFrames) return std::nullopt;
    std::array<double, MaxFrames> ranking{};
    for (int k = 0; k < count; ++k) {
        ranking[k] = frameRms(pcm.subspan(k * Frame, Frame));
    }
    std::sort(ranking.begin(), ranking.begin() + count);
    const auto index = (count - 1) * .85;
    const auto below = static_cast<int>(index);
    const double q85 = ranking[below] + (ranking[std::min(below + 1, count - 1)] - ranking[below]) * (index - below);
    const auto floor = std::max(180.0, q85 * .08);
    int first = -1;
    int last = -1;
    for (int k = 0; k < count; ++k) {
        if (frameRms(pcm.subspan(k * Frame, Frame)) < floor) continue;
        if (first < 0) first = k;
        last = k;
    }
    if (first < 0) return std::nullopt;
    return Span{first, last, count};
}

std::string_view trimHit(const char* token) {
    if (!token) return {};
    std::string_view view(token);
    auto start = view.find_first_not_of(' ');
    if (start == std::string_view::npos) return {};
    return view.substr(start, view.find_last_not_of(' ') - start + 1);
}

bool containsWord(std::string_view value, std::string_view word) {
    for (auto i = value.find(word); i != std::string_view::npos; i = value.find(word, i + 1)) {
        const auto end = i + word.size();
        if ((i == 0 || value[i - 1] == ' ') && (end == value.size() || value[end] == ' ')) return true;
    }
    return false;
}

struct Cue {
    int start;
    int end;
    int score;
    const DiceScopeAssets::Phrase* phrase;
};

int gapBefore(const Cue& cue, Span speech) { return std::max(0, cue.start - speech.first); }
int gapAfter(const Cue& cue, Span speech) { return std::max(0, speech.last - cue.end); }

auto rank(const Cue& cue, Span speech) {
    return std::tuple(cue.score - 4 * (gapBefore(cue, speech) + gapAfter(cue, speech)),
                      cue.end - cue.start, cue.phrase->text);
}

std::optional<Cue> pickBody(ps_decoder_t* decoder, Span speech) {
    std::optional<Cue> best;
    for (auto* seg = ps_seg_iter(decoder); seg; seg = ps_seg_next(seg)) {
        const auto word = trimHit(ps_seg_word(seg));
        auto order = std::ranges::find_if(DiceScopeAssets::Orders, [word](const auto& p) { return p.text == word; });
        if (order == std::end(DiceScopeAssets::Orders)) continue;
        int start;
        int end;
        ps_seg_frames(seg, &start, &end);
        const Cue item{start, end, ps_seg_prob(seg, nullptr, nullptr, nullptr), order};
        if (!best || rank(item, speech) > rank(*best, speech)) best = item;
    }
    if (best) {
        best->start = std::clamp(best->start, 0, speech.count - 1);
        best->end = std::clamp(best->end, best->start, speech.count - 1);
    }
    return best;
}

bool conflictingMention(ps_decoder_t* decoder, const Cue& cue) {
    for (auto* seg = ps_seg_iter(decoder); seg; seg = ps_seg_next(seg)) {
        const auto word = trimHit(ps_seg_word(seg));
        if (std::ranges::find(DiceScopeAssets::Mentions, word) == std::end(DiceScopeAssets::Mentions)) continue;
        int start;
        int end;
        ps_seg_frames(seg, &start, &end);
        if (start <= cue.end + 30 && end >= cue.start - 25 &&
            ps_seg_prob(seg, nullptr, nullptr, nullptr) >= MentionFloor) {
            ps_seg_free(seg);
            return true;
        }
    }
    return false;
}

std::optional<double> score(ps_decoder_t* decoder, const char* name, std::span<const int16_t> pcm,
                            bool negative, bool prefix) {
    if (!decodeAcoustic(decoder, name, pcm)) return std::nullopt;
    int32 logScore = 0;
    const auto* heard = ps_get_hyp(decoder, &logScore);
    const auto count = ps_get_n_frames(decoder);
    if (!heard || count <= 0) return std::nullopt;
    if (negative) {
        using namespace std::literals;
        constexpr std::array start{"no"sv, "ni"sv};
        constexpr std::array end{"no"sv, "cancela"sv, "cancélalo"sv, "olvídalo"sv, "déjalo"sv};
        const auto negativeTerms = prefix ? std::span<const std::string_view>(start) : std::span<const std::string_view>(end);
        if (!std::ranges::any_of(negativeTerms, [heard](std::string_view marker) {
                return containsWord(heard, marker);
            })) return std::nullopt;
    }
    return static_cast<double>(logScore) / count;
}

bool veto(ps_decoder_t* decoder, std::span<const int16_t> pcm, bool active, bool prefix) {
    if (!active || pcm.size() < 800) return false;
    const auto affirm = score(decoder, prefix ? "r38_prefix_aff" : "r38_suffix_aff", pcm, false, prefix);
    const auto deny = score(decoder, prefix ? "r38_prefix_neg" : "r38_suffix_neg", pcm, true, prefix);
    const bool blocked = deny && (!affirm || *deny - *affirm >= VetoMargin);
#ifdef BB_ESMX_D6_SCOPE_TRACE
    std::fprintf(stderr, "BB_D6_TRACE %s %u %.8f %u %.8f %u\n",
                 prefix ? "prefix" : "suffix", affirm.has_value(), affirm.value_or(0.0),
                 deny.has_value(), deny.value_or(0.0), blocked);
#endif
    return blocked;
}

int roundEven(int numerator, int denominator) {
    const auto quotient = numerator / denominator;
    const auto twiceRemainder = 2 * (numerator % denominator);
    return quotient + (twiceRemainder > denominator || (twiceRemainder == denominator && quotient % 2 != 0));
}
}

bool installDiceScopeSearches(ps_decoder_t* decoder, const char* keywords) {
    struct stat info{};
    if (stat(keywords, &info) != 0 || !S_ISREG(info.st_mode)) return false;
    for (const auto& term : DiceScopeAssets::Lexicon) {
        const std::unique_ptr<char, decltype(&std::free)> known(ps_lookup_word(decoder, term.term), &std::free);
        if (known) {
            // Preserve the existing Clock and Butter pronunciation. Decline
            // this experiment when one of its pinned words would differ.
            if (std::string_view(known.get()) != term.phones) return false;
            continue;
        }
        if (ps_add_word(decoder, term.term, term.phones, 0) < 0) return false;
    }
    using namespace DiceScopeAssets;
    const auto grammars = std::array{
        std::pair{"r38_prefix_aff", prefix_aff}, std::pair{"r38_prefix_neg", prefix_neg},
        std::pair{"r38_suffix_aff", suffix_aff}, std::pair{"r38_suffix_neg", suffix_neg},
    };
    for (const auto& [name, grammar] : grammars) {
        if (ps_add_jsgf_string(decoder, name, grammar) < 0) return false;
    }
    // Register the entry search last so partial initialization never admits D6.
    return ps_add_kws(decoder, Spot, keywords) == 0;
}

bool confirmsDiceScope(std::span<const int16_t> pcm, bool requireFiniteVerb) {
    auto* decoder = sharedAcousticDecoder();
    const auto speech = voicedSpan(pcm);
    if (!speech || !decodeAcoustic(decoder, Spot, pcm)) return false;
    const auto body = pickBody(decoder, *speech);
#ifdef BB_ESMX_D6_SCOPE_TRACE
    std::fprintf(stderr, "BB_D6_TRACE span %d %d %d\n", speech->first, speech->last, speech->count);
    if (body) {
        std::fprintf(stderr, "BB_D6_TRACE body %d %d %d %d %d %s\n", body->start, body->end, body->score,
                     gapBefore(*body, *speech), gapAfter(*body, *speech), body->phrase->text.data());
    }
#endif
    if (!body || body->score < BodyFloor || gapBefore(*body, *speech) > MaxStartGap ||
        gapAfter(*body, *speech) > MaxEndGap) return false;
    if (requireFiniteVerb) {
        // Infinitives in the closed order list can occur in explanations.
        const auto text = body->phrase->text;
        const auto verb = text.substr(0, text.find(' '));
        if (verb.ends_with("ar") || verb.ends_with("er") || verb.ends_with("ir")) return false;
    }
    const bool mentioned = conflictingMention(decoder, *body);
#ifdef BB_ESMX_D6_SCOPE_TRACE
    std::fprintf(stderr, "BB_D6_TRACE mention %u\n", mentioned);
#endif
    if (mentioned) return false;

    const int relative = roundEven((body->end - body->start + 1) * body->phrase->verbPhones, body->phrase->phones);
    const int stop = std::min(speech->count, std::max(body->start + 8, body->start + relative + 12));
    const int begin = std::max(body->start, body->end - 10);
    return !veto(decoder, pcm.first(stop * Frame), speech->first < stop, true) &&
           !veto(decoder, pcm.subspan(begin * Frame), speech->last > body->end + 4, false);
}
}
#endif
