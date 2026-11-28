#include "Routines/SettingsActivityRoutine.h"
#include <Phrases.h>
#include <Util/ServiceLocator.h>

PhraseListeningRoutine::PhraseListeningRoutine(BBStateMachine* sm) : Routine(sm){
    af = ApplicationStatics::getApplication()->getService<AudioFrontend>();
}

PhraseListeningRoutine::~PhraseListeningRoutine() = default;

void PhraseListeningRoutine::listen(std::span<const AudioFrontend::Phrase> phrases){
    assert(phrases.size() == 5);
    ActivityHost::steps.push_back(ActivityHost::Step::Listen);
}

void testActivity(){
    using namespace ActivityHost;
    ServiceLocator::SpeechAudioGenInstance = std::make_unique<SpeechAudioGen>();
    ServiceLocator::SettingsInstance = std::make_unique<Settings>();

    SettingsActivityHyperRoutine hyper(nullptr);
    assert(hyper.tick(0) == Routine::TickingState::Done);
    assert((steps == std::vector{Step::Set, Step::Store, Step::Sync, Step::Change, Step::Play, Step::Wait}));
    assert(ServiceLocator::SettingsInstance->get().activityMode == ActivityMode::Hyper);
    assert(sync.activityMode == ActivityMode::Hyper && sync.frontSensor == FrontSensorMode::Off &&
           sync.bottomSensor == BottomSensorMode::Reduced);
    assert(change.phrase == Phrase::SettingsActivityHyper && change.id == 0);
    assert(pcmSamples > 0);

    steps.clear();
    pcmSamples = 0;
    SettingsActivityRoutine ask(nullptr);
    assert(ask.tick(0) == Routine::TickingState::Block);
    assert((steps == std::vector{Step::Change, Step::Play, Step::Wait, Step::Listen}));
    assert(change.phrase == Phrase::SettingsAskActivity && change.id == 0);
    assert(pcmSamples > 0);
}
