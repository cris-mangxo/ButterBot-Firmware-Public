#include "ClockGuard.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace EsMx::PhonemeNative {
namespace {

constexpr float Impossible = -std::numeric_limits<float>::infinity();
constexpr std::size_t States = 2 * MaxTemplatePhones + 1;
static_assert(std::tuple_size_v<decltype(Workspace::first)> >= MaxFrames + States);

float logAdd(float left, float right){
	const auto high = std::max(left, right);
	if(high == Impossible){
		return high;
	}
	return high + std::log1p(std::exp(std::min(left, right) - high));
}

float fullPath(std::span<const float> logits, std::span<const float> peaks,
               std::span<const std::uint8_t> path, std::span<float> states){
	if(path.empty() || path.size() > MaxTemplatePhones ||
	   std::any_of(path.begin(), path.end(), [](auto token){ return token == 0 || token >= PhoneCount; })){
		return Impossible;
	}
	const auto width = 2 * path.size() + 1;
	auto dp = states.first(width);
	std::ranges::fill(dp, Impossible);
	dp[0] = 0;
	for(std::size_t frame = 0; frame < peaks.size(); ++frame){
		const auto row = logits.subspan(frame * PhoneCount, PhoneCount);
		for(std::size_t remaining = width; remaining > 0; --remaining){
			const auto i = remaining - 1;
			const auto token = i % 2 ? path[i / 2] : 0;
			auto total = dp[i];
			if(i > 0){
				total = logAdd(total, dp[i - 1]);
			}
			if(token && i >= 3 && token != path[i / 2 - 1]){
				total = logAdd(total, dp[i - 2]);
			}
			dp[i] = total + (row[token] - peaks[frame]);
		}
	}
	return logAdd(dp[width - 1], dp[width - 2]);
}

float bestFull(std::span<const float> logits, std::span<const float> peaks,
               std::span<const std::span<const std::uint8_t>> patterns, std::span<float> states){
	float best = Impossible;
	for(const auto pattern : patterns){
		best = std::max(best, fullPath(logits, peaks, pattern, states));
	}
	return best;
}

Alignment bestLocal(const AlignmentFrames& view, std::size_t frames,
                    std::span<const std::span<const std::uint8_t>> paths){
	std::optional<Alignment> best;
	for(const auto path : paths){
		const auto hit = view.match(path, frames);
		if(hit && (!best || std::tie(hit->deficitPerPhone, hit->beginFrame, hit->endFrame) <
		                    std::tie(best->deficitPerPhone, best->beginFrame, best->endFrame))){
			best = hit;
		}
	}
	return best.value_or(Alignment{8.0, 0, 0});
}

}

std::optional<double> scoreClock(const Output& output, std::size_t frames,
                                const ClockRule& rule, std::span<float> scratch){
	constexpr std::size_t ClockIndex = 3; // Intent outputs are 0, 1, 3, 5, 6.
	if(frames < 12 || frames > MaxFrames || scratch.size() < frames + States ||
	   !std::isfinite(output.negation) || output.negation >= 0 ||
	   !std::ranges::all_of(output.intents, [](auto value){ return std::isfinite(value); })){
		return std::nullopt;
	}
	const auto logits = std::span(output.phones).first(frames * PhoneCount);
	const auto peaks = scratch.first(frames);
	const auto view = AlignmentFrames::prepare(logits, peaks);
	if(!view){
		return std::nullopt;
	}
	auto states = scratch.subspan(frames, States);
	const auto now = bestFull(logits, peaks, rule.now, states);
	const auto event = bestFull(logits, peaks, rule.event, states);
	const auto other = bestFull(logits, peaks, rule.other, states);
	if(!std::isfinite(now) || !std::isfinite(event) || !std::isfinite(other)){
		return std::nullopt;
	}

	// All paths share the normalization of each row. Apply it once to the
	// absolute cost. It cancels out of the two relative costs.
	double normalization = 0;
	for(std::size_t frame = 0; frame < frames; ++frame){
		float sum = 0;
		for(const auto logit : logits.subspan(frame * PhoneCount, PhoneCount)){
			sum += std::exp(logit - peaks[frame]);
		}
		normalization += std::log(sum);
	}
	const auto noun = bestLocal(*view, frames, rule.noun);
	const auto direct = bestLocal(*view, frames, rule.localNow);
	const auto eventPrefix = view->match(rule.eventPrefix, noun.endFrame).value_or(Alignment{8., 0, 0});
	const auto nowPrefix = view->match(rule.nowPrefix, noun.endFrame).value_or(Alignment{8., 0, 0});
	const auto rival = std::max({output.intents[0], output.intents[1], output.intents[2], output.intents[4]});
	const auto count = static_cast<double>(frames);
	const std::array<double, 9> values{
		noun.deficitPerPhone, output.intents[ClockIndex] - rival, (normalization - now) / count,
		-output.negation, (static_cast<double>(now) - event) / count, (static_cast<double>(now) - other) / count,
		eventPrefix.deficitPerPhone - nowPrefix.deficitPerPhone, (direct.endFrame - direct.beginFrame) / count,
		direct.deficitPerPhone,
	};
	double score = rule.intercept;
	for(std::size_t i = 0; i < values.size(); ++i){
		score += rule.coefficients[i] * values[i];
	}
	return std::isfinite(score) ? std::optional(score) : std::nullopt;
}

}
