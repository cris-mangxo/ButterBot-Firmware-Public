#ifndef BUTTERBOT_ESMX_PILOT_VOICE_H
#define BUTTERBOT_ESMX_PILOT_VOICE_H

#ifdef BB_ESMX_PILOT

#include <cstdint>

class Audio;

namespace EsMx::Pilot {

// These calls start playback and wait for it to end. Use them outside the AFE tasks.
int16_t butterVoiceId(int16_t stockId);
int16_t jokeVoiceId(int16_t stockId);
bool playButter(Audio& audio, int16_t id);
bool playJoke(Audio& audio, int16_t id);
bool playClockUnset(Audio& audio, int16_t id);
bool playClockTime(Audio& audio, unsigned hour24, unsigned minute);
bool playDiceIntro(Audio& audio);
bool playDiceResult(Audio& audio, unsigned face);
bool playListening(Audio& audio);
bool playUnrecognized(Audio& audio);
bool playCancelled(Audio& audio);

}

#endif

#endif
