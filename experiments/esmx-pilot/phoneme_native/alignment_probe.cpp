#include "Alignment.h"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

int main(int argc, char** argv){
	using namespace EsMx::PhonemeNative;
	if(argc < 2 || argc > 1 + static_cast<int>(MaxTemplatePhones)){
		return 2;
	}
	std::array<std::uint8_t, MaxTemplatePhones> phones{};
	for(int index = 1; index < argc; ++index){
		const std::string_view token(argv[index]);
		unsigned value = 0;
		const auto end = token.data() + token.size();
		const auto result = std::from_chars(token.data(), end, value);
		if(result.ec != std::errc{} || result.ptr != end || value >= PhoneCount){
			return 2;
		}
		phones[index - 1] = static_cast<std::uint8_t>(value);
	}
	static std::array<float, MaxFrames * PhoneCount + 1> logits;
	std::cin.read(reinterpret_cast<char*>(logits.data()), sizeof(logits));
	const auto bytes = static_cast<std::size_t>(std::cin.gcount());
	if(std::cin.bad() || bytes % sizeof(float)){
		return 2;
	}
	const auto result = align(std::span(logits).first(bytes / sizeof(float)), std::span(phones).first(argc - 1));
	if(result){
		std::cout << '[' << std::setprecision(std::numeric_limits<double>::max_digits10)
		          << result->deficitPerPhone << ',' << result->beginFrame << ',' << result->endFrame << "]\n";
	}else{
		std::cout << "null\n";
	}
	return std::cout ? 0 : 2;
}
