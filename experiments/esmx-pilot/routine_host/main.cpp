#include "stubs.h"

#ifdef NDEBUG
#error "The routine host harness requires assertions"
#endif

#include "Audio/EsMxPilot/Dispatch.h"
#ifdef ROUTINE_HOST_RECOGNITION
#include "Audio/EsMxPilot/Recognition.h"
#include <iostream>
#include <string_view>
#endif
#include "Routines/CurrentTimeRoutine.h"
#include "Routines/DiceRollRoutine.h"
#include "Routines/JokeRoutine.h"
#include "Routines/PassTheButterRoutine.h"
#include "States/ScenarioRoutineMappings.h"
#include "voice_data.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <span>

using EsMx::Kws::Intent;
using namespace RoutineHost;

// The OEM table is exercised independently via the firmware build and ELF checks.
// This host seam forwards the same three scenario keys to the actual routine types.
RoutineFactory routineForScenario(const std::pair<Scenario, ScenarioData>& scenario){
    assert(scenario.second.raw == 0);
    switch(scenario.first){
        case Scenario::PassTheButter: return &makeRoutine<PassTheButterRoutine>;
        case Scenario::Joke: return &makeRoutine<JokeRoutine>;
        case Scenario::CurrentTime: return &makeRoutine<CurrentTimeRoutine>;
        default: return nullptr;
    }
}

// Keep the legacy listening dependency absent from the host. Its dialog can be spotted
// via the constructor and listen hook without using a recognizer in that legacy path.
PhraseListeningRoutine::PhraseListeningRoutine(BBStateMachine* sm) : Routine(sm){
    af = ApplicationStatics::getApplication()->getService<AudioFrontend>();
    ++legacyModelCalls;
}

PhraseListeningRoutine::~PhraseListeningRoutine() = default;

void PhraseListeningRoutine::listen(std::span<const AudioFrontend::Phrase> phrases){
    assert(!phrases.empty());
    ++legacyListening;
}

namespace {

using Clip = std::span<const uint8_t>;
using namespace EsMx::Pilot::VoiceData;

void clearTrace(){
    events.clear();
    randomReads = 0;
    legacyListening = 0;
    legacyModelCalls = 0;
}

void run(Intent intent){
    clearTrace();
    const RoutineFactory factory = EsMx::Pilot::routineForIntent(intent);
    assert(factory != nullptr);
    alignas(std::max_align_t) std::byte storage[RoutineBufSize];
    auto* routine = factory(storage, nullptr);
    assert(routine->tick(0.0f) == Routine::TickingState::Done);
    routine->~Routine();
    assert(legacyListening == 0 && legacyModelCalls == 0);
}

template<class Data>
const Data& expectPacket(size_t index, Scenario scenario){
    const auto& packet = std::get<Packet>(events.at(index));
    assert(packet.state == BB::State::Scenario && packet.scenario == scenario);
    return std::get<Data>(packet.payload);
}

const Playback& expectPlayback(size_t index, Source kind){
    const auto& playback = std::get<Playback>(events.at(index));
    assert(playback.source == kind);
    assert(std::holds_alternative<Wait>(events.at(index + 1)));
    return playback;
}

void expectClips(size_t index, std::initializer_list<Clip> clips){
    const auto& playback = expectPlayback(index, Source::Aac);
    std::vector<uint8_t> expected;
    for(const auto clip : clips){
        assert(!clip.empty());
        expected.insert(expected.end(), clip.begin(), clip.end());
    }
    assert(playback.bytes == expected);
    assert(expected.size() > 7 && expected[0] == 0xff && (expected[1] & 0xf0) == 0xf0);
}

void testButter(){
    run(Intent::Butter);
    assert(events.size() == 3);
    const auto id = expectPacket<PassTheButterData>(0, Scenario::PassTheButter).id;
    assert(id < std::size(Butter) && !Phrases::map(Phrase::PassTheButter, id).empty());
    expectClips(1, {Butter[id]});
}

void testJoke(){
    run(Intent::Joke);
    assert(events.size() == 3);
    const auto id = expectPacket<JokeData>(0, Scenario::Joke).id;
    assert(id < std::size(Joke) && !Phrases::map(Phrase::Joke, id).empty());
    expectClips(1, {Joke[id]});
}

void testClockAt(int hour, int minute, std::initializer_list<Clip> clips){
    auto& rtc = RoutineHost::clock;
    rtc.configured = true;
    rtc.value = std::tm{.tm_min = minute, .tm_hour = hour, .tm_mday = 27, .tm_mon = 10, .tm_year = 126};
    run(Intent::Time);
    assert(events.size() == 3);
    const auto& data = expectPacket<CurrentTimeData>(0, Scenario::CurrentTime);
    assert(data.status == CurrentTimeData::Status::ShowTime);
    assert(data.currentTime.tm_hour == hour && data.currentTime.tm_min == minute);
    assert(data.currentTime.tm_mday == 27 && data.currentTime.tm_mon == 10 && data.currentTime.tm_year == 126);
    expectClips(1, clips);
}

void testClockUnset(){
    RoutineHost::clock.configured = false;
    run(Intent::Time);
    assert(events.size() == 3);
    const auto& data = expectPacket<CurrentTimeData>(0, Scenario::CurrentTime);
    assert(data.status == CurrentTimeData::Status::NoTime && data.currentTime.tm_year == 0);
    const auto& playback = expectPlayback(1, Source::Aac);
    const auto matchingClip = std::ranges::find_if(ClockUnset, [&](Clip clip) {
        return std::ranges::equal(clip, playback.bytes);
    });
    assert(matchingClip != std::end(ClockUnset));
    expectClips(1, {*matchingClip});
}

void testDice6(uint16_t face){
    assert(face >= 1 && face <= 6);
    randomValues = {static_cast<uint32_t>(face - 1), 2};
    run(Intent::Dice6);
    assert(randomReads == 2);
    assert(events.size() == 8);
    expectClips(0, {DiceIntro});
    const auto& animation = expectPacket<DiceRollData>(2, Scenario::DiceRoll);
    assert(animation.phase == DiceRollData::Phase::RollAnim && animation.diceType == DiceRollData::DiceType::D6);
    assert(animation.diceCount == 1 && animation.diceResult == 0);
    assert(expectPlayback(3, Source::File).argument == "/spiffs/dice/roll_3.aac");
    const auto& result = expectPacket<DiceRollData>(5, Scenario::DiceRoll);
    assert(result.phase == DiceRollData::Phase::Result && result.diceType == DiceRollData::DiceType::D6);
    assert(result.diceCount == 1 && result.diceResult == face);
    expectClips(6, {DiceFaces[face - 1]});
}

void testFactoryD6KeepsCountPrompt(){
    clearTrace();
    DiceRollD6Routine routine(nullptr);
    assert(routine.tick(0.0f) == Routine::TickingState::Block);
    assert(legacyListening == 1 && legacyModelCalls == 1);
    assert(events.size() == 3 && randomReads == 0);
    const auto& count = expectPacket<DiceRollData>(0, Scenario::DiceRoll);
    assert(count.phase == DiceRollData::Phase::RollCount && count.diceType == DiceRollData::DiceType::None);
    assert(count.diceCount == 0 && count.diceResult == 0);
    assert(!expectPlayback(1, Source::Speech).argument.empty());
}

void testEnglishFallbackWhenAacUnavailable(){
    aacAvailable = false;
    run(Intent::Butter);
    aacAvailable = true;
    assert(events.size() == 3);
    const auto id = expectPacket<PassTheButterData>(0, Scenario::PassTheButter).id;
    assert(expectPlayback(1, Source::Speech).argument == Phrases::map(Phrase::PassTheButter, id));
}

#ifdef ROUTINE_HOST_RECOGNITION
int testPcmEpisode(std::string_view requested){
    using namespace std::literals;
    constexpr std::array labels{
        std::pair{"butter"sv, Intent::Butter},
        std::pair{"joke"sv, Intent::Joke},
        std::pair{"time"sv, Intent::Time},
        std::pair{"dice6"sv, Intent::Dice6},
        std::pair{"none"sv, Intent::Unknown},
    };
    const auto expected = std::ranges::find_if(labels, [&](auto entry){ return entry.first == requested; });
    if(expected == labels.end()){
        std::fputs("Expected intent: butter, joke, time, dice6 or none\n", stderr);
        return 2;
    }

    std::array<int16_t, EsMx::Kws::MaxSamples + 1> pcm;
    std::cin.read(reinterpret_cast<char*>(pcm.data()), sizeof(pcm));
    const auto bytes = std::cin.gcount();
    if(std::cin.bad() || bytes <= 0 || bytes % 2 || bytes > EsMx::Kws::MaxSamples * 2){
        std::fputs("Expected mono PCM s16le at 16 kHz, at most four seconds\n", stderr);
        return 2;
    }

    const auto observed = EsMx::Pilot::recognize(std::span(pcm.data(), bytes / 2));
    if(observed != expected->second){
        std::printf("{\"recognized_label\":%u,\"expected_label\":%u,\"real_ticks\":0}\n",
                    static_cast<unsigned>(observed), static_cast<unsigned>(expected->second));
        return 1;
    }

    switch(observed){
        case Intent::Butter: testButter(); break;
        case Intent::Joke: testJoke(); break;
        case Intent::Time:
            testClockAt(19, 31, {ClockPlural, Numbers[7], ClockWith, CompoundDecades[0], Numbers[1], ClockNight});
            break;
        case Intent::Dice6: testDice6(5); break;
        case Intent::Unknown:
            assert(EsMx::Pilot::routineForIntent(observed) == nullptr && events.empty());
            std::puts("{\"recognized_label\":0,\"real_ticks\":0,\"abstained\":true}");
            return 0;
        default: return 2;
    }
    std::printf("{\"recognized_label\":%u,\"real_ticks\":1,\"stock_payload_verified\":true,"
                "\"matching_aac_clips\":true}\n", static_cast<unsigned>(observed));
    return 0;
}
#endif

} // namespace

int main(int argc, char** argv){
    Phrases::preallocate();
#ifdef ROUTINE_HOST_RECOGNITION
    if(argc == 2){
        return testPcmEpisode(argv[1]);
    }
#endif
    if(argc != 1){
        return 2;
    }
    testButter();
    testJoke();
    testClockUnset();
    testClockAt(0, 0, {ClockPlural, Numbers[12], ClockNight, ClockOnTime});
    testClockAt(13, 1, {ClockOne, ClockWith, ClockOneMinute, ClockAfternoon});
    testClockAt(19, 31, {ClockPlural, Numbers[7], ClockWith, CompoundDecades[0], Numbers[1], ClockNight});
    for(uint16_t face = 1; face <= 6; ++face){
        testDice6(face);
    }
    testFactoryD6KeepsCountPrompt();
    testEnglishFallbackWhenAacUnavailable();
    assert(EsMx::Pilot::routineForIntent(Intent::Fact) == nullptr);
    std::puts("{\"real_ticks\":14,\"stock_payloads\":true,\"matching_aac_clips\":true,"
              "\"d6_faces\":6,\"legacy_d6_keeps_count_prompt\":true,\"aac_failure_uses_english\":true}");
}
