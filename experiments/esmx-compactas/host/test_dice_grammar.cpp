#include "DiceGrammar.h"

#include <cstdlib>
#include <string>
#include <string_view>

using namespace EsMx::Kws;
using namespace std::literals;

void require(bool condition){
	if(!condition){
		std::abort();
	}
}

int main(){
	for(auto phrase : {"  ¿PODRÍAS lanzarme ese dado de 6 caras?  "sv,
	                   "Oye, echa por fa tu dado de seis lados."sv,
	                   "Haz rodar este dado de seis."sv, "Tira un D6."sv,
	                   "A ver, ¿puedes tirar el dado de seis?"sv}){
		require(isDice6Request(phrase));
	}
	for(auto phrase : {"No tires un dado de seis."sv, "Tira un dado de seis. No, espera."sv,
	                   "Tira un dado de seis pero nunca ahorita."sv, "Tira un dado de dieciséis."sv,
	                   "Lanza un D60."sv, "Rueda seis dados."sv, "Tira un dado de seis y otro de veinte."sv,
	                   "Ese dado de seis lados es rojo."sv, "Si puedes, tira un dado de seis."sv,
	                   "Él dijo: tira un dado de seis."sv, "\"Tira un dado de seis.\""sv,
	                   "¿Cuántos lados tiene un dado de seis?"sv, "tira un dado de seis\ny después"sv,
	                   "Tira un dado de seis. Tira otro."sv, "Tira un dado de seis\xc3"sv, ""sv}){
		require(!isDice6Request(phrase));
	}
	const auto text = "Tira un dado de seis"sv;
	require(!isDice6Request(std::string(MaxTranscriptBytes + 1, ' ') + std::string(text)));
	require(completeDice6(Intent::Unknown, Intent::Unknown, text) == Intent::Dice6);
	require(completeDice6(Intent::Fact, Intent::Unknown, text) == Intent::Unknown);
	require(completeDice6(Intent::Dice6, Intent::Unknown, text) == Intent::Unknown);
	require(completeDice6(Intent::Unknown, Intent::Dice6, "no") == Intent::Dice6);
	for(auto intent : {Intent::Butter, Intent::YouButter, Intent::Joke, Intent::Fact, Intent::Time}){
		require(completeDice6(intent, intent, text) == intent);
	}
}
