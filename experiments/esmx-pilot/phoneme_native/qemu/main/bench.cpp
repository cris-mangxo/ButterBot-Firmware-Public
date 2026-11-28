#include "Kernel.h"
#include "student_bank.h"
#include "student_tests.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <inttypes.h>
#include <span>

#include <esp_timer.h>

namespace {

using namespace EsMx::PhonemeNative;
static_assert(Test::BankSha == FrozenSha256);
Workspace workspace;
Output output;

bool compare(std::span<const float> values, std::span<const float> expected, float& maximumError) {
    bool result = true;
    for(std::size_t index = 0; index < expected.size(); ++index) {
        const auto delta = std::abs(values[index] - expected[index]);
        maximumError = std::max(delta, maximumError);
        result &= delta <= Test::AbsoluteTolerance + Test::RelativeTolerance * std::abs(expected[index]);
    }
    return result;
}

}

extern "C" void app_main() {
    unsigned failures = 0;
    std::printf("PHONEME_BANK %s %u\n", FrozenSha256, static_cast<unsigned>(sizeof(Bank)));
    for(const auto& fixture : Test::Cases) {
        const auto start = esp_timer_get_time();
        const bool valid = infer(fixture.frames, Frozen, workspace, output);
        const auto elapsed = esp_timer_get_time() - start;
        const auto phonemeCount = fixture.frames.size() / Bands * PhoneCount;
        float maximumError = 0;
        const bool logits = compare(output.phones, fixture.logits.first(phonemeCount), maximumError)
                            & compare(output.intents, fixture.logits.subspan(phonemeCount, IntentCount), maximumError)
                            & compare(std::span(&output.negation, 1), fixture.logits.last(1), maximumError);
        const bool equal = valid && logits && output.intentId == fixture.intentId;
        failures += !equal;
        std::printf("PHONEME_CASE %.*s %u %u %u %.8g %" PRId64 "\n",
                    static_cast<int>(fixture.name.size()), fixture.name.data(),
                    static_cast<unsigned>(output.intentId), static_cast<unsigned>(fixture.intentId),
                    static_cast<unsigned>(logits), static_cast<double>(maximumError), elapsed);
    }
    std::printf("PHONEME_DONE %u %u\n", static_cast<unsigned>(std::size(Test::Cases)), failures);
    std::fflush(stdout);
}
