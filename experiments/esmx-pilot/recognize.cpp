#include "Audio/EsMxPilot/Recognition.h"
#ifdef BB_ESMX_BUTTER_ACOUSTIC
#include "Audio/EsMxPilot/JokeFallback.h"
#include "WordDice.h"
#include <iomanip>
#endif

#include <array>
#include <iostream>
#include <string_view>

namespace {
std::string_view dispositionName(EsMx::Pilot::TurnDisposition disposition){
	using enum EsMx::Pilot::TurnDisposition;
	switch(disposition){
		case Accepted: return "accepted";
		case NoMatch: return "no_match";
		case InvalidAudio: return "invalid_audio";
		case FactVeto: return "fact_veto";
		case UnconfirmedDice: return "unconfirmed_dice";
		case PhraseRejected: return "phrase_rejected";
		case DiceNegationVeto: return "dice_negation_veto";
		case DiceNegationConflict: return "dice_negation_conflict";
		case DiceFollowUp: return "dice_follow_up";
	}
	return "invalid_disposition";
}
}

int main(int argc, char** argv){
	using namespace EsMx;
	const bool reportDecision = argc == 2 && std::string_view(argv[1]) == "--decision";
#ifdef BB_ESMX_BUTTER_ACOUSTIC
	const bool reportButterPhonemes = argc == 2 && std::string_view(argv[1]) == "--butter-phonemes";
#else
	constexpr bool reportButterPhonemes = false;
#endif
	if(argc != 1 && !reportDecision && !reportButterPhonemes){
		return 2;
	}
	static std::array<int16_t, Kws::MaxSamples + 1> pcm;

	std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
	const auto bytes = static_cast<size_t>(std::cin.gcount());
	if(bytes % sizeof(int16_t) != 0 || bytes > Kws::MaxSamples * sizeof(int16_t) || std::cin.bad()){
		return 2;
	}

	const std::span<const int16_t> samples(pcm.data(), bytes / sizeof(int16_t));
#ifdef BB_ESMX_BUTTER_ACOUSTIC
	if(reportButterPhonemes){
		static bb::asr::LogMel frontend;
		static Kws::WordFeatures words;
		const auto evidence = bb::asr::logmel_init(&frontend) ? Pilot::measureButterPhonemes(frontend, words, samples)
		                                                     : Pilot::ButterPhonemeEvidence{};
		std::cout << std::setprecision(15) << std::boolalpha << "{\"valid\":" << evidence.valid
		          << ",\"posterior\":" << evidence.posterior << ",\"margin\":" << evidence.margin
		          << ",\"negation\":" << evidence.negation
		          << ",\"accepted\":" << Pilot::confirmsButterPhonemes(evidence) << "}\n";
		return std::cout ? 0 : 2;
	}
#endif
	if(reportDecision){
		const auto decision = Pilot::recognizeDetailed(samples);
		std::cout << "{\"intent\":" << static_cast<int>(decision.intent)
		          << ",\"disposition\":\"" << dispositionName(decision.disposition) << "\""
		          << ",\"allows_fallback\":" << std::boolalpha << decision.allowsFallback() << "}\n";
	}else{
		std::cout << static_cast<int>(Pilot::recognize(samples)) << '\n';
	}
	return std::cout ? 0 : 2;
}
