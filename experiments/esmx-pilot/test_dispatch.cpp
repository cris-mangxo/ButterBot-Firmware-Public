#include "dispatch_probe.h"

#include <array>
#include <cstdio>

int main(){
    using Intent = EsMx::Kws::Intent;
    const std::array successes{
        std::pair{Intent::Butter, "butter"}, std::pair{Intent::Joke, "joke"},
        std::pair{Intent::Time, "time"}, std::pair{Intent::Dice6, "dice6_factory"}
    };
    for(const auto& [intent, route] : successes){
        if(probeDispatch(intent) != route){
            return 1;
        }
    }
    for(const Intent rejected : {Intent::Unknown, Intent::YouButter, Intent::Fact, Intent::Count,
                                 static_cast<Intent>(255)}){
        if(probeDispatch(rejected) != "none"){
            return 1;
        }
    }
    std::puts("DISPATCH_HOST_OK stock_scenario_keys=3 d6_factory=1 rejected=5");
}
