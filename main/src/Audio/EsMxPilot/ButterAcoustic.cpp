#ifdef BB_ESMX_BUTTER_ACOUSTIC
#include "ButterAcoustic.h"

#include "AcousticSearch.h"
#include "Audio/EsMxKws/Compact.h"
#include "ButterModelAssets.h"
#ifdef BB_ESMX_BUTTER_CEPSTRA
#include "cepstra_bridge.h"
#include <cstdio>
#endif
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#endif

namespace EsMx::Pilot {
#if defined(ESP_PLATFORM) && !defined(BB_ESMX_BUTTER_BENCHMARK)
static_assert(!ButterModelAssets::ValidationEnabled, "Butter validation is reserved for the emulator benchmark");
#endif
#ifdef BB_ESMX_BUTTER_WEAK_CONFLICT
static_assert(ButterModelAssets::ValidationEnabled, "Weak conflict requires the sealed experimental validation profile");
#endif
namespace {

using namespace std::literals;
constexpr char KeywordsSearch[] = "bb-butter-keywords";
constexpr char AffirmSearch[] = "bb-butter-affirm";
constexpr char OpposeSearch[] = "bb-butter-oppose";
constexpr std::string_view Object = "mantequillabb";
constexpr std::string_view ArticleObject = "la mantequillabb";
constexpr auto Courtesy = std::array{"por favor"sv, "porfa"sv};
constexpr auto Address = std::array{"oye"sv, "oiga"sv, "oigan"sv, "robot"sv, "amigo"sv, "amiga"sv,
                                    "señor"sv, "señora"sv, "niño"sv, "niña"sv, "juana"sv};
constexpr auto Negative = std::array{"no"sv, "ni"sv, "sin"sv, "mejor no"sv, "ya no"sv, "cancela"sv};
constexpr auto Recipe = std::array{"usa"sv, "uses"sv, "use"sv, "pon"sv, "agrega"sv, "agregue"sv, "derrite"sv,
                                    "derrita"sv, "mezcla"sv, "corta"sv, "corte"sv, "es"sv, "vende"sv,
                                    "preparo"sv, "preparé"sv, "untar"sv, "compra"sv, "compré"sv,
                                    "con"sv, "quiero"sv, "cambiamos"sv};
constexpr auto Query = std::array{"qué es"sv, "dónde está"sv, "cuánto cuesta"sv, "qué tiene"sv, "no hay"sv,
                                   "hay"sv, "necesitamos"sv};
constexpr auto Demonstrative = std::array{"la"sv, "esta"sv, "esa"sv};
constexpr auto Description = std::array{"es"sv, "está"sv, "cuesta"sv, "tiene"sv, "sirve"sv};
constexpr auto Substitutes = std::array{"margarina"sv, "mermelada"sv, "leche"sv, "harina"sv, "crema"sv};
constexpr auto Location = std::array{"aquí"sv, "ahí"sv, "allí"sv, "acá"sv};
constexpr auto LocationVerb = std::array{"está"sv, "estaba"sv, "hay"sv, "había"sv, "tienes"sv,
                                          "tiene"sv, "tengo"sv, "puse"sv, "queda"sv, "quedó"sv};
constexpr auto OpposingSuffix = std::array{"no"sv, "mejor no"sv, "olvídalo"sv, "ni se te ocurra"sv, "cancela"sv,
                                            "de maní"sv, "de cacahuate"sv, "de almendra"sv};
constexpr auto OpposingKeywords = std::array{"de cacahuate"sv, "de maní"sv, "mejor no"sv, "olvídalo"sv,
                                              "no gracias"sv, "no quiero que"sv, "cancela"sv};
constexpr size_t FrameSamples = bb::asr::kSampleRate / 100;
constexpr size_t TailSamples = 4 * FrameSamples;
// Frozen R37 validation settings, calibrated on its development split.
constexpr int ObjectMinimum = -1800;
constexpr int ObjectMinFrames = 24;
constexpr int ObjectMaxFrames = 128;
constexpr int BoundarySlack = 12;
constexpr int ArticleGap = 20;
constexpr int UtteranceTail = 24;
constexpr int ContrastPerFrame = 3;
constexpr int CancellationMinimum = -1900;
constexpr int CancellationLead = 2;
constexpr uint64_t FloorEnergy = uint64_t(FrameSamples) * 180 * 180;

bool startsPhrase(std::string_view text, std::string_view prefix){
    return text.starts_with(prefix) && (text.size() == prefix.size() || text[prefix.size()] == ' ');
}

bool endsPhrase(std::string_view text, std::string_view suffix){
    return text.ends_with(suffix) && (text.size() == suffix.size() || text[text.size() - suffix.size() - 1] == ' ');
}

template<size_t N>
bool startsOne(std::string_view text, const std::array<std::string_view, N>& words){
    return std::any_of(words.begin(), words.end(), [text](auto word){ return startsPhrase(text, word); });
}

template<size_t N>
std::string_view stripOne(std::string_view text, const std::array<std::string_view, N>& words){
    for(const auto word : words){
        if(startsPhrase(text, word)){
            return text.substr(std::min(text.size(), word.size() + 1));
        }
    }
    return text;
}

bool wholeAffirmative(std::string_view text){
    for(const auto suffix : Courtesy){
        if(endsPhrase(text, suffix)){
            text.remove_suffix(suffix.size() + (text.size() > suffix.size()));
            break;
        }
    }
    return endsPhrase(text, Object) && text.size() > Object.size();
}

}

bool butterOpposesRequest(std::string_view text){
    text = stripOne(stripOne(text, Address), Courtesy);
    if(std::any_of(Substitutes.begin(), Substitutes.end(), [text](auto word){ return endsPhrase(text, word); })){
        return true;
    }

    const auto noun = text.find(Object);
    if(noun == std::string_view::npos || (noun && text[noun - 1] != ' ') || !startsPhrase(text.substr(noun), Object)){
        return false;
    }
    const auto before = noun ? text.substr(0, noun - 1) : std::string_view{};
    const auto after = text.substr(std::min(text.size(), noun + Object.size() + 1));
    const auto locative = stripOne(before, Location);
    return (locative.size() != before.size() && startsOne(locative, LocationVerb)) ||
           startsOne(before, Negative) || startsOne(before, Recipe) || startsOne(before, Query) ||
           (std::find(Demonstrative.begin(), Demonstrative.end(), before) != Demonstrative.end() &&
              startsOne(after, Description)) || startsOne(after, OpposingSuffix);
}

bool admitsTerminalNoSupportCheck(std::string_view hypothesis){
    constexpr std::string_view terminalNo = " no";
    if(!endsPhrase(hypothesis, "mantequillabb no")){
        return false;
    }

    hypothesis.remove_suffix(terminalNo.size());
    return !butterOpposesRequest(hypothesis);
}

namespace {

std::optional<ButterTerm> term(std::string_view word){
    const auto begin = word.find_first_not_of(" \t\r\n");
    if(begin == std::string_view::npos){
        return std::nullopt;
    }
    word = word.substr(begin, word.find_last_not_of(" \t\r\n") - begin + 1);
    if(word == Object){
        return ButterTerm::Object;
    }
    if(word == ArticleObject){
        return ButterTerm::ArticleObject;
    }
    if(std::find(OpposingKeywords.begin(), OpposingKeywords.end(), word) != OpposingKeywords.end()){
        return ButterTerm::Opposition;
    }
    return std::nullopt;
}

int calibratedScore(ps_decoder_t* decoder, int score){
    // The sealed replay used the PocketSphinx Python probability round trip.
    auto* logmath = ps_get_logmath(decoder);
    return logmath_log(logmath, logmath_exp(logmath, score));
}

ButterSegment segmentInfo(ps_decoder_t* decoder, ps_seg_t* segment){
    ButterSegment info;
    ps_seg_frames(segment, &info.start, &info.end);
    info.score = calibratedScore(decoder, ps_seg_prob(segment, nullptr, nullptr, nullptr));
    return info;
}

bool recordHits(ps_decoder_t* decoder, ButterEvidence& evidence){
    for(auto* seg = ps_seg_iter(decoder); seg; seg = ps_seg_next(seg)){
        if(const auto kind = term(ps_seg_word(seg))){
            if(evidence.hitCount == evidence.hits.size()){
                ps_seg_free(seg);
                return false;
            }
            evidence.hits[evidence.hitCount++] = {*kind, segmentInfo(decoder, seg)};
        }
    }
    return true;
}

bool voicedFrame(std::span<const int16_t> pcm, size_t frame){
    const auto begin = frame * FrameSamples;
    if(begin >= pcm.size()){
        return false;
    }
    const auto samples = pcm.subspan(begin, std::min(FrameSamples, pcm.size() - begin));
    uint64_t energy = 0;
    for(const int32_t value : samples){
        energy += static_cast<uint64_t>(value * value);
    }
    return energy >= FloorEnergy;
}

int lastSpeechFrame(std::span<const int16_t> pcm){
    for(size_t end = (pcm.size() + FrameSamples - 1) / FrameSamples; end > 0; --end){
        if(voicedFrame(pcm, end - 1)){
            return static_cast<int>(end - 1);
        }
    }
    return -1;
}

void validateTerminalNo(ps_decoder_t* decoder, std::string_view hypothesis,
                        std::span<const int16_t> pcm, ButterEvidence& evidence){
    if(!evidence.opposing || !admitsTerminalNoSupportCheck(hypothesis)){
        return;
    }

    int count = 0;
    ButterSegment cue;
    for(auto* segment = ps_seg_iter(decoder); segment; segment = ps_seg_next(segment)){
        if(std::string_view(ps_seg_word(segment)) == "no"){
            ++count;
            cue = segmentInfo(decoder, segment);
        }
    }
    if(count != 1 || cue.start > cue.end){
        return;
    }

    evidence.terminalNo = cue;
    const auto frameCount = static_cast<int>((pcm.size() + FrameSamples - 1) / FrameSamples);
    const auto lastAvailable = std::min(cue.end, frameCount - 1);
    int supported = 0;
    for(int frame = std::max(cue.start, 0); frame <= lastAvailable; ++frame){
        supported += voicedFrame(pcm, static_cast<size_t>(frame));
    }

    // Use the original PCM only. A partially or entirely padded segment cannot
    // count those frames as spoken support for the single terminal "no".
    if(2 * supported < cue.end - cue.start + 1){
        evidence.opposing = false;
        evidence.unsupportedTerminalNo = true;
    }
}

void* allocSamples(size_t size){
#ifdef ESP_PLATFORM
    return heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
    return std::malloc(size);
#endif
}

void freeSamples(int16_t* data){
#ifdef ESP_PLATFORM
    heap_caps_free(data);
#else
    std::free(data);
#endif
}

bool qualifiesAsObject(const ButterHit& hit){
    const auto length = hit.segment.end - hit.segment.start + 1;
    return hit.type != ButterTerm::Opposition && hit.segment.score >= ObjectMinimum &&
           hit.segment.start >= 0 && length >= ObjectMinFrames && length <= ObjectMaxFrames;
}

bool matchesObject(const ButterHit& hit, const ButterEvidence& evidence){
    if(!qualifiesAsObject(hit)){
        return false;
    }
    const int start = hit.type == ButterTerm::ArticleObject ? evidence.affirmArticle.start
                                                           : evidence.affirmObject.start;
    const int spokenEnd = std::min(evidence.affirmObject.end, evidence.lastSpeech);
    return start >= 0 && std::abs(start - hit.segment.start) <= BoundarySlack &&
           std::abs(spokenEnd - hit.segment.end) <= BoundarySlack;
}

bool hasWitness(const ButterEvidence& evidence){
    const auto hits = std::span(evidence.hits).first(evidence.hitCount);
    return std::any_of(hits.begin(), hits.end(), qualifiesAsObject);
}

void recordAffirmative(ps_decoder_t* decoder, ButterEvidence& evidence){
    int score = 0;
    const auto* hypothesis = ps_get_hyp(decoder, &score);
    if(!hypothesis || !wholeAffirmative(hypothesis)){
        return;
    }

    ButterSegment lastArticle;
    int objects = 0;
    for(auto* segment = ps_seg_iter(decoder); segment; segment = ps_seg_next(segment)){
        const std::string_view word = ps_seg_word(segment);
        if(word == "la"){
            lastArticle = segmentInfo(decoder, segment);
        }else if(word == Object){
            ++objects;
            evidence.affirmObject = segmentInfo(decoder, segment);
            const int gap = evidence.affirmObject.start - lastArticle.end;
            if(lastArticle.start >= 0 && gap >= 0 && gap <= ArticleGap){
                evidence.affirmArticle = lastArticle;
            }
        }
    }
    evidence.affirmative = objects == 1;
    evidence.affirmScore = calibratedScore(decoder, score);
}

}

bool installButterAcousticSearches(ps_decoder_t* decoder, const char* keywords){
    return ps_add_kws(decoder, KeywordsSearch, keywords) >= 0 &&
           ps_add_jsgf_string(decoder, AffirmSearch, ButterModelAssets::Affirmative) >= 0 &&
           ps_add_jsgf_string(decoder, OpposeSearch, ButterModelAssets::Opposing) >= 0;
}

ButterEvidence measureAcousticButter(std::span<const int16_t> pcm){
    ButterEvidence result;
    if(pcm.empty() || pcm.size() > Kws::MaxSamples){
        return result;
    }
    result.lastSpeech = lastSpeechFrame(pcm);
    auto* decoder = sharedAcousticDecoder();
    if(result.lastSpeech < 0 || !decodeAcoustic(decoder, KeywordsSearch, pcm) || !recordHits(decoder, result)){
        return result;
    }

    if(!hasWitness(result)){
        result.complete = true;
        return result;
    }

    const std::unique_ptr<int16_t, decltype(&freeSamples)> buffer(
        static_cast<int16_t*>(allocSamples((pcm.size() + TailSamples) * sizeof(int16_t))), freeSamples);
    if(!buffer){
        return result;
    }
    std::copy(pcm.begin(), pcm.end(), buffer.get());
    std::fill_n(buffer.get() + pcm.size(), TailSamples, 0);
    const std::span<const int16_t> padded(buffer.get(), pcm.size() + TailSamples);
#ifdef BB_ESMX_BUTTER_CEPSTRA
    const std::unique_ptr<bb_ps_cepstra, decltype(&bb_ps_free_cepstra)> cepstra(
        bb_ps_capture_cepstra(decoder, padded.data(), padded.size()), bb_ps_free_cepstra);
    std::printf("BB_BUTTER_CEPSTRA used=%u\n", static_cast<unsigned>(static_cast<bool>(cepstra)));
#endif
    const auto decodeFsg = [&](const char* search){
#ifdef BB_ESMX_BUTTER_CEPSTRA
        if(cepstra){
            if(ps_activate_search(decoder, search) < 0 || ps_start_utt(decoder) < 0){
                return false;
            }
            const auto processed = bb_ps_replay_cepstra(decoder, cepstra.get());
            const auto ended = ps_end_utt(decoder);
            return processed >= 0 && ended >= 0;
        }
#endif
        return decodeAcoustic(decoder, search, padded);
    };
    if(!decodeFsg(AffirmSearch)){
        return result;
    }
    result.ranFsg = true;
    recordAffirmative(decoder, result);
    if(!decodeFsg(OpposeSearch)){
        return result;
    }
    int score = 0;
    if(const auto* hyp = ps_get_hyp(decoder, &score); hyp){
        result.opposing = butterOpposesRequest(hyp);
        result.opposeScore = calibratedScore(decoder, score);
        validateTerminalNo(decoder, hyp, pcm, result);
    }
    result.frames = ps_get_n_frames(decoder);
    result.complete = true;
    return result;
}

bool acceptsAcousticButter(const ButterEvidence& evidence){
    if(!evidence.complete || !evidence.affirmative || !evidence.ranFsg || evidence.frames <= 0 ||
       evidence.hitCount > evidence.hits.size() || evidence.affirmObject.start < 0){
        return false;
    }
    const int spokenEnd = std::min(evidence.affirmObject.end, evidence.lastSpeech);
    if(spokenEnd < evidence.affirmObject.start || evidence.lastSpeech - spokenEnd > UtteranceTail){
        return false;
    }
    if(evidence.opposing &&
       int64_t(evidence.affirmScore) - evidence.opposeScore <= int64_t(ContrastPerFrame) * evidence.frames){
        return false;
    }
    const auto hits = std::span(evidence.hits).first(evidence.hitCount);
    return std::none_of(hits.begin(), hits.end(), [&](const ButterHit& hit){
               return hit.type == ButterTerm::Opposition && hit.segment.score >= CancellationMinimum &&
                      hit.segment.end >= spokenEnd - CancellationLead;
           }) &&
           std::any_of(hits.begin(), hits.end(), [&](const ButterHit& hit){ return matchesObject(hit, evidence); });
}

bool recognizesAcousticButter(std::span<const int16_t> pcm){
    return ButterModelAssets::ValidationEnabled && acceptsAcousticButter(measureAcousticButter(pcm));
}

}
#endif
