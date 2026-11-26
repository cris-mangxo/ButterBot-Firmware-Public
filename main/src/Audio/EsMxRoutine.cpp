#include "EsMxRoutine.h"
#include "EsMxCommand.h"

RoutineFactory EsMx::routineForTranscript(std::string_view transcript){
	const auto command = matchCommand(transcript);
	return command ? routineForScenario(command->scenario) : nullptr;
}
