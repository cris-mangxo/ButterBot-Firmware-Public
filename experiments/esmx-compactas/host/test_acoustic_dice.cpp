#include "AcousticDice.h"

#include <cstdlib>

using namespace EsMx::Kws;
namespace {

void require(bool condition){
	if(!condition){
		std::abort();
	}
}
}

int main(){
	AcousticDice model{};
	model.first.shift = model.second.shift = model.third.shift = 1;
	model.biases[1] = 24;
	model.minimumScore = 16;
	model.minimumMargin = 8;
	require(validAcousticDice(model));

	Signature frame{};
	require(completeAcousticDice(frame, Intent::Unknown, Intent::Unknown, model) == Intent::Dice6);
	require(completeAcousticDice(frame, Intent::Fact, Intent::Unknown, model) == Intent::Unknown);
	for(auto intent : {Intent::Butter, Intent::YouButter, Intent::Joke, Intent::Fact, Intent::Time, Intent::Dice6}){
		require(completeAcousticDice(frame, Intent::Unknown, intent, model) == intent);
	}
}
