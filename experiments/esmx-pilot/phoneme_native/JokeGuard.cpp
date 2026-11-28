#include "JokeGuard.h"

#include <algorithm>
#include <cmath>
#include <tuple>

namespace EsMx::PhonemeNative {
namespace {

Alignment cheapest(const AlignmentFrames& scores, std::size_t frames, std::span<const TokenPath> paths){
	std::optional<Alignment> best;
	for(const auto path : paths){
		const auto match = scores.match(path, frames);
		if(match && (!best ||
		             std::tie(match->deficitPerPhone, match->beginFrame, match->endFrame) <
		             std::tie(best->deficitPerPhone, best->beginFrame, best->endFrame))){
			best = match;
		}
	}
	return best.value_or(Alignment{8.0, 0, 0});
}

}

std::optional<double> scoreJoke(const Output& output, std::size_t frames, const JokeRule& rule,
                                std::span<float> peakWorkspace){
	constexpr std::size_t JokeIndex = 2;  // Intent output order is 0, 1, 3, 5, 6.
	if(frames < 16 || frames > MaxFrames || !std::isfinite(output.negation) || output.negation >= 0 ||
	   !std::all_of(output.intents.begin(), output.intents.end(), [](float value){ return std::isfinite(value); })){
		return std::nullopt;
	}
	const auto scores = AlignmentFrames::prepare(std::span(output.phones).first(frames * PhoneCount), peakWorkspace);
	if(!scores){
		return std::nullopt;
	}
	const auto noun = cheapest(*scores, frames, rule.noun);
	const auto verb = cheapest(*scores, noun.beginFrame, rule.request);
	const auto denial = cheapest(*scores, noun.beginFrame, rule.denial);
	const auto positive = cheapest(*scores, frames, rule.positive).deficitPerPhone;
	const auto negative = cheapest(*scores, frames, rule.negative).deficitPerPhone;
	const auto rival = std::max({output.intents[0], output.intents[1], output.intents[3], output.intents[4]});
	const auto count = static_cast<double>(frames);
	const std::array<double, 9> values{
		output.intents[JokeIndex] - rival, -output.negation, noun.deficitPerPhone, verb.deficitPerPhone,
		denial.deficitPerPhone - verb.deficitPerPhone,
		(noun.beginFrame - verb.endFrame) / count, (frames - noun.endFrame) / count,
		positive, negative - positive,
	};
	double score = rule.intercept;
	for(std::size_t index = 0; index < values.size(); ++index){
		score += values[index] * rule.coefficients[index];
	}
	return std::isfinite(score) ? std::optional(score) : std::nullopt;
}

bool confirmsJoke(const Output& output, std::size_t frames, const JokeRule& rule, std::span<float> peakWorkspace){
	const auto score = scoreJoke(output, frames, rule, peakWorkspace);
	return score && *score > rule.threshold;
}

}
