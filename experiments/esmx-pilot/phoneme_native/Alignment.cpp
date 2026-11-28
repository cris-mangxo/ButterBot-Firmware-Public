#include "Alignment.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace EsMx::PhonemeNative {
namespace {

struct Path {
	double deficit = std::numeric_limits<double>::infinity();
	std::uint16_t begin = 0;
};

Path preferred(Path left, Path right){
	return right.deficit < left.deficit ||
	       (right.deficit == left.deficit && right.begin < left.begin) ? right : left;
}

}

std::optional<AlignmentFrames> AlignmentFrames::prepare(std::span<const float> logits,
                                                       std::span<float> peakWorkspace){
	const auto frames = logits.size() / PhoneCount;
	if(!frames || logits.size() % PhoneCount || frames > MaxFrames || frames > peakWorkspace.size()){
		return std::nullopt;
	}
	for(std::size_t frame = 0; frame < frames; ++frame){
		const auto row = logits.subspan(frame * PhoneCount, PhoneCount);
		float peak = row.front();
		for(const float value : row){
			if(!std::isfinite(value)){
				return std::nullopt;
			}
			peak = std::max(peak, value);
		}
		peakWorkspace[frame] = peak;
	}
	return AlignmentFrames(logits, peakWorkspace.first(frames));
}

std::optional<Alignment> AlignmentFrames::match(std::span<const std::uint8_t> phones,
                                                std::size_t frameCount) const{
	if(phones.empty() || phones.size() > MaxTemplatePhones || !frameCount || frameCount > rowPeaks.size() ||
	   std::any_of(phones.begin(), phones.end(), [](auto phone){ return phone == 0 || phone >= PhoneCount; })){
		return std::nullopt;
	}
	std::array<Path, 2 * MaxTemplatePhones - 1> previous;
	const auto states = 2 * phones.size() - 1;
	Path best;
	std::uint16_t end = 0;
	for(std::uint16_t frame = 0; frame < frameCount; ++frame){
		const auto row = logits.subspan(frame * PhoneCount, PhoneCount);
		const auto peak = rowPeaks[frame];
		// Visit states in reverse so all predecessors still belong to the previous frame.
		for(auto remaining = states; remaining > 0; --remaining){
			const auto state = remaining - 1;
			const bool isPhone = state % 2 == 0;
			const auto phone = isPhone ? phones[state / 2] : 0;
			const auto advance = state ? previous[state - 1] : Path{0.0, frame};
			auto selected = preferred(previous[state], advance);
			if(isPhone && state >= 2 && phone != phones[state / 2 - 1]){
				selected = preferred(selected, previous[state - 2]);
			}
			// Match float32 subtraction and float64 accumulation in the reference.
			selected.deficit += static_cast<double>(peak - row[phone]);
			previous[state] = selected;
		}
		if(previous[states - 1].deficit < best.deficit){
			best = previous[states - 1];
			end = frame + 1;
		}
	}
	if(!std::isfinite(best.deficit)){
		return std::nullopt;
	}
	return Alignment{best.deficit / phones.size(), best.begin, end};
}

std::optional<Alignment> align(std::span<const float> logits, std::span<const std::uint8_t> phones){
	std::array<float, MaxFrames> peaks;
	const auto frames = AlignmentFrames::prepare(logits, peaks);
	return frames ? frames->match(phones, logits.size() / PhoneCount) : std::nullopt;
}

}
