#include "DiceGrammar.h"

#include <array>
#include <charconv>
#include <iostream>
#include <string_view>

using namespace EsMx::Kws;

bool intentCode(std::string_view text, Intent& intent){
	unsigned numeric;
	const auto [end, err] = std::from_chars(text.data(), text.data() + text.size(), numeric);
	if(err != std::errc{} || end != text.data() + text.size() || numeric >= static_cast<unsigned>(Intent::Count)){
		return false;
	}
	intent = static_cast<Intent>(numeric);
	return true;
}

int main(int argc, char** argv){
	Intent base, verified;
	if(argc != 3 || !intentCode(argv[1], base) || !intentCode(argv[2], verified)){
		return 2;
	}
	std::array<char, MaxTranscriptBytes + 1> input;
	std::cin.read(input.data(), input.size());
	if(!std::cin.eof() || std::cin.bad()){
		return 2;
	}
	const std::string_view transcript(input.data(), static_cast<size_t>(std::cin.gcount()));
	std::cout << static_cast<int>(completeDice6(base, verified, transcript)) << '\n';
}
