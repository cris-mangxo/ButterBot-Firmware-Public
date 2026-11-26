#include "Audio/EsMxCommand.h"

#include <array>
#include <iostream>
#include <string_view>

int main(){
	std::array<char, 129> text;
	std::cin.read(text.data(), text.size());
	const auto transcript = std::string_view(text.data(), static_cast<std::size_t>(std::cin.gcount()));
	const auto command = EsMx::matchCommand(transcript);
	if(!command){
		std::cout << "{\"status\":\"rejected\"}\n";
		return 0;
	}
	const auto& [scenario, data] = command->scenario;
	std::cout << "{\"status\":\"matched\",\"intent\":\"" << command->name
		<< "\",\"scenario\":" << static_cast<int>(scenario)
		<< ",\"data\":" << static_cast<int>(data.raw) << "}\n";
}
