#ifdef BB_ESMX_BUTTER_ACOUSTIC
#include "ButterAcoustic.h"

#include "AcousticSearch.h"
#include "Audio/EsMxKws/Compact.h"
#include "ButterModelAssets.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string_view>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#endif

namespace EsMx::Pilot {
#ifdef ESP_PLATFORM
static_assert(!ButterModelAssets::Gate.enabled, "The experimental Mantequilla gate is desktop-only");
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
constexpr auto Negative = std::array{"no"sv, "ni"sv, "mejor no"sv, "ya no"sv};
constexpr auto Recipe = std::array{"usa"sv, "uses"sv, "use"sv, "pon"sv, "agrega"sv, "agregue"sv, "derrite"sv,
                                    "derrita"sv, "mezcla"sv, "corta"sv, "corte"sv, "es"sv, "vende"sv,
                                    "preparo"sv, "preparé"sv, "untar"sv, "compra"sv, "compré"sv,
                                    "con"sv, "quiero"sv, "cambiamos"sv};
constexpr auto Query = std::array{"qué es"sv, "dónde está"sv, "cuánto cuesta"sv, "qué tiene"sv, "no hay"sv,
                                   "hay"sv, "necesitamos"sv};
constexpr auto Demonstrative = std::array{"la"sv, "esta"sv, "esa"sv};
constexpr auto Description = std::array{"es"sv, "está"sv, "cuesta"sv, "tiene"sv, "sirve"sv};
constexpr auto Substitutes = std::array{"margarina"sv, "mermelada"sv, "leche"sv, "harina"sv, "crema"sv};
constexpr auto OpposingSuffix = std::array{"no"sv, "mejor no"sv, "olvídalo"sv, "ni se te ocurra"sv, "cancela"sv,
                                            "de maní"sv, "de cacahuate"sv, "de almendra"sv};
constexpr auto OpposingKeywords = std::array{"de cacahuate"sv, "de maní"sv, "mejor no"sv, "olvídalo"sv,
                                              "no gracias"sv, "no quiero que"sv, "cancela"sv};
constexpr size_t FrameSamples = bb::asr::kSampleRate / 100;
constexpr size_t TailSamples = 4 * FrameSamples;
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

bool semanticOpposition(std::string_view text){
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
    return startsOne(before, Negative) || startsOne(before, Recipe) || startsOne(before, Query) ||
           (std::find(Demonstrative.begin(), Demonstrative.end(), before) != Demonstrative.end() &&
              startsOne(after, Description)) || startsOne(after, OpposingSuffix);
}

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

ButterSegment segmentInfo(ps_seg_t* segment){
    ButterSegment info;
    ps_seg_frames(segment, &info.start, &info.end);
    info.score = ps_seg_prob(segment, nullptr, nullptr, nullptr);
    return info;
}

bool recordHits(ps_decoder_t* decoder, ButterEvidence& evidence){
    for(auto* seg = ps_seg_iter(decoder); seg; seg = ps_seg_next(seg)){
        if(const auto kind = term(ps_seg_word(seg))){
            if(evidence.hitCount == evidence.hits.size()){
                ps_seg_free(seg);
                return false;
            }
            evidence.hits[evidence.hitCount++] = {*kind, segmentInfo(seg)};
        }
    }
    return true;
}

int lastSpeechFrame(std::span<const int16_t> pcm){
    for(size_t end = (pcm.size() + FrameSamples - 1) / FrameSamples; end > 0; --end){
        const auto begin = (end - 1) * FrameSamples;
        const auto samples = pcm.subspan(begin, std::min(FrameSamples, pcm.size() - begin));
        uint64_t energy = 0;
        for(const int32_t value : samples){
            energy += static_cast<uint64_t>(value * value);
        }
        if(energy >= FloorEnergy){
            return static_cast<int>(end - 1);
        }
    }
    return -1;
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

bool matchesObject(const ButterHit& hit, const ButterEvidence& evidence, const ButterGate& gate){
    if(hit.type == ButterTerm::Opposition || hit.segment.score < gate.keywordMinimum ||
       hit.segment.start < 0 || hit.segment.end < hit.segment.start){
        return false;
    }
    const int deltaStart = evidence.affirmObject.start - hit.segment.start;
    const int lead = hit.type == ButterTerm::ArticleObject ? gate.articleLead : gate.onsetSlack;
    const int spokenEnd = std::min(evidence.affirmObject.end, evidence.lastSpeech);
    return deltaStart >= -gate.onsetSlack && deltaStart <= lead &&
           std::abs(spokenEnd - hit.segment.end) <= gate.endSlack &&
           evidence.lastSpeech - hit.segment.end <= gate.tailSlack;
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

    const std::unique_ptr<int16_t, decltype(&freeSamples)> buffer(
        static_cast<int16_t*>(allocSamples((pcm.size() + TailSamples) * sizeof(int16_t))), freeSamples);
    if(!buffer){
        return result;
    }
    std::copy(pcm.begin(), pcm.end(), buffer.get());
    std::fill_n(buffer.get() + pcm.size(), TailSamples, 0);
    const std::span<const int16_t> padded(buffer.get(), pcm.size() + TailSamples);
    if(!decodeAcoustic(decoder, AffirmSearch, padded)){
        return result;
    }
    if(const auto* hyp = ps_get_hyp(decoder, &result.affirmScore); hyp && wholeAffirmative(hyp)){
        result.affirmative = true;
        for(auto* seg = ps_seg_iter(decoder); seg; seg = ps_seg_next(seg)){
            if(Object == ps_seg_word(seg)){
                result.affirmObject = segmentInfo(seg);
            }
        }
    }
    if(!decodeAcoustic(decoder, OpposeSearch, padded)){
        return result;
    }
    if(const auto* hyp = ps_get_hyp(decoder, &result.opposeScore); hyp){
        result.opposing = semanticOpposition(hyp);
    }
    result.frames = ps_get_n_frames(decoder);
    result.complete = true;
    return result;
}

bool acceptsAcousticButter(const ButterEvidence& evidence, const ButterGate& gate){
    if(!gate.enabled || !evidence.complete || !evidence.affirmative || evidence.frames <= 0 ||
       evidence.hitCount > evidence.hits.size() ||
       evidence.lastSpeech < evidence.affirmObject.start || evidence.affirmObject.start < 0 ||
       evidence.affirmObject.end < evidence.affirmObject.start ||
       evidence.affirmScore < int64_t(gate.floorPerFrame) * evidence.frames){
        return false;
    }
    if(evidence.opposing &&
       int64_t(evidence.affirmScore) - evidence.opposeScore <= int64_t(gate.contrastPerFrame) * evidence.frames){
        return false;
    }
    const auto hits = std::span(evidence.hits).first(evidence.hitCount);
    const int noObject = evidence.frames + 1;
    int earliest = noObject;
    for(const auto& hit : hits){
        if(matchesObject(hit, evidence, gate)){
            earliest = std::min(earliest, hit.segment.start);
        }
    }
    return earliest != noObject && std::none_of(hits.begin(), hits.end(), [&](const ButterHit& opposition){
        return opposition.type == ButterTerm::Opposition &&
               opposition.segment.score >= gate.oppositionKeywordMinimum &&
               opposition.segment.end >= earliest;
    });
}

bool recognizesAcousticButter(std::span<const int16_t> pcm){
    return ButterModelAssets::Gate.enabled && acceptsAcousticButter(measureAcousticButter(pcm), ButterModelAssets::Gate);
}

}
#endif
