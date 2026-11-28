#include "Audio/EsMxPilot/Recognition.h"
#include "DiceConsensus.h"
#include "acoustic_dice.h"
#include "backup_acoustic_dice.h"
#include "consensus_acoustic_dice.h"
#include "phrase_bank.h"
#include "pilot_model.h"
#include "word_adapted.h"
#include "word_bank.h"
#include "word_extra.h"

#include <array>
#include <iostream>
#include <string_view>

namespace {
using namespace EsMx::Kws;

void reportAcoustic(std::string_view name, const Signature& signature, const AcousticDice& bank){
    const auto scores = acousticDiceScores(signature, bank);
    std::cout << '\"' << name << "\":{\"would_propose_d6\":" << proposesDice6(scores, bank)
              << ",\"would_compete_as_negation\":" << hasCompetingNegation(scores, bank)
              << ",\"would_prefer_negation\":" << prefersNegation(scores, bank)
              << ",\"would_meet_negation_floor\":" << (scores[DiceNegation] >= bank.minimumScore) << '}';
}
}

int main(){
    using namespace EsMx;
    static std::array<int16_t, Kws::MaxSamples + 1> pcm;
    std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
    const auto bytes = static_cast<size_t>(std::cin.gcount());
    if(bytes % sizeof(int16_t) != 0 || bytes > Kws::MaxSamples * sizeof(int16_t) || std::cin.bad()){
        return 2;
    }
    const std::span<const int16_t> samples(pcm.data(), bytes / sizeof(int16_t));
    const auto final = Pilot::recognize(samples);
    bb::asr::LogMel frontend;
    Signature signature;
    const bool fullTurn = bb::asr::logmel_init(&frontend) && extract(frontend, samples, signature);
    std::cout << std::boolalpha << "{\"final\":" << static_cast<int>(final)
              << ",\"signature_available\":" << fullTurn;
    if(fullTurn){
        const auto baseline = classify(signature, PilotModel, PilotLimits).intent;
        const auto phrase = verifyPhrase(signature, baseline, PilotPhrases);
        static WordFeatures words;
        constexpr const WordBank* supplements[]{&PilotWordExtra, &PilotWordAdapted};
        const auto lexical = completeWordDice(frontend, samples, baseline, phrase, PilotWordDice,
                                              words, supplements);
        const auto consensus = completeConsensusDice(signature, baseline, lexical, PilotAcousticDice,
                                                      PilotConsensusDice, words, PilotWordDice,
                                                      PilotWordExtra, PilotWordAdapted,
                                                      ConsensusMode::OrderedDie, &PilotBackupDice);
        std::cout << ",\"baseline\":" << static_cast<int>(baseline)
                  << ",\"phrase\":" << static_cast<int>(phrase)
                  << ",\"lexical\":" << static_cast<int>(lexical)
                  << ",\"consensus\":" << static_cast<int>(consensus);
        // Independent facts on the computed features, even when a guard
        // prevents the runtime from reaching the corresponding bank.
        if(baseline == Intent::Unknown){
            std::cout << ",\"hypothetical_acoustic_checks\":{";
            reportAcoustic("reference", signature, PilotAcousticDice);
            std::cout << ',';
            reportAcoustic("candidate", signature, PilotConsensusDice);
            std::cout << ',';
            reportAcoustic("backup", signature, PilotBackupDice);
            std::cout << '}';
        }
        std::cout << ",\"lexical_frames\":" << words.count;
        if(words.count){
            std::cout << ",\"primary_words\":" << acceptsWordDice(words, PilotWordDice)
                      << ",\"additional_words\":" << acceptsWordDice(words, PilotWordExtra)
                      << ",\"adapted_words\":" << acceptsWordDice(words, PilotWordAdapted)
                      << ",\"shared_six\":" << confirmsSixAcrossBanks(words, PilotWordDice,
                                                                     PilotWordExtra, PilotWordAdapted)
                      << ",\"ordered_die\":" << confirmsOrderedDieSix(words, PilotWordDice,
                                                                      PilotWordExtra, PilotWordAdapted)
                      << ",\"would_confirm_words_if_reference_proposed\":"
                      << (confirmContinuousDice(Intent::Unknown, Intent::Dice6, words, PilotWordAdapted,
                                                 DiceConfirmation::OrderedRescue) == Intent::Dice6);
        }
    }
    std::cout << "}\n";
    return std::cout ? 0 : 2;
}
