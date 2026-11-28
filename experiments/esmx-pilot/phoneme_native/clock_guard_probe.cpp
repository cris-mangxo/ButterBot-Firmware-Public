#include "ClockGuard.h"
#include "clock_rule.h"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

int main(int argc, char** argv){
	using namespace EsMx::PhonemeNative;
	if(argc != 2){
		return 2;
	}
	const std::string_view argument(argv[1]);
	const auto end = argument.data() + argument.size();
	std::size_t frames = 0;
	const auto parsed = std::from_chars(argument.data(), end, frames);
	if(parsed.ec != std::errc{} || parsed.ptr != end || frames > MaxFrames){
		return 2;
	}
	static Output output;
	static Workspace workspace;
	std::cin.read(reinterpret_cast<char*>(output.phones.data()), frames * PhoneCount * sizeof(float));
	std::cin.read(reinterpret_cast<char*>(output.intents.data()), sizeof(output.intents));
	std::cin.read(reinterpret_cast<char*>(&output.negation), sizeof(output.negation));
	if(!std::cin || std::cin.peek() != std::char_traits<char>::eof() || std::cin.bad()){
		return 2;
	}
	const auto score = scoreClock(output, frames, FrozenClock, workspace.first);
	std::cout << "{\"score\":";
	if(score){
		std::cout << std::setprecision(std::numeric_limits<double>::max_digits10) << *score;
	}else{
		std::cout << "null";
	}
	std::cout << ",\"accepted\":" << std::boolalpha << (score && *score > FrozenClock.threshold) << "}\n";
	return std::cout ? 0 : 2;
}
