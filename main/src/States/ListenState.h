#ifndef BUTTERBOT_FIRMWARE_LISTENSTATE_H
#define BUTTERBOT_FIRMWARE_LISTENSTATE_H

#include "State.h"
#include "CtrlData.h"
#include "Audio/AudioFrontend.h"
#include "Services/Com.h"
#include "Services/ISRButtonInput.h"

class ListenState : public State {
public:
    explicit ListenState(BBStateMachine* sm);
    virtual ~ListenState() override;

    static void preallocate();
#ifdef BB_ESMX_PILOT
    void tick(float deltaTime) override;
#endif

private:
    Com* com = nullptr;

    bool timedOut = false;
#ifdef BB_ESMX_PILOT
    bool pilotStartupFailed = true;
    void onPilotIntent(EsMx::Kws::Intent intent);
#endif

    void onRC();
    void onPhrase(bool recognized, int index, const std::string& transcript, float confidence);
    void onButtonEvent(int button, ISRButtonInput::Action action);
    void onCommand(Ctrl::Command cmd);

    void aborted();

	static std::vector<AudioFrontend::Phrase> phrases;
};

#endif //BUTTERBOT_FIRMWARE_LISTENSTATE_H
