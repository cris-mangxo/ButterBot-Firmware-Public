#include "Audio/EsMxPilot/Recognition.h"

#include <array>
#include <iostream>

int main(){
	using namespace EsMx;
	static std::array<int16_t, Kws::MaxSamples + 1> pcm;

	std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
	const auto bytes = static_cast<size_t>(std::cin.gcount());
	if(bytes % sizeof(int16_t) != 0 || bytes > Kws::MaxSamples * sizeof(int16_t) || std::cin.bad()){
		return 2;
	}

	const auto intent = Pilot::recognize(std::span(pcm.data(), bytes / sizeof(int16_t)));
	std::cout << static_cast<int>(intent) << '\n';
	return std::cout ? 0 : 2;
}
