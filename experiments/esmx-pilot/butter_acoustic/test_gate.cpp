#include "Audio/EsMxPilot/ButterAcoustic.h"
#include <cassert>
#include <iostream>

int main(){
    using namespace EsMx::Pilot;
    assert(butterOpposesRequest("no mantequillabb"));
    assert(butterOpposesRequest("sin mantequillabb"));
    assert(butterOpposesRequest("mantequillabb cancela"));
    assert(butterOpposesRequest("mantequillabb no"));
    assert(!butterOpposesRequest("pásame la mantequillabb"));
    assert(admitsTerminalNoSupportCheck("pásame la mantequillabb no"));
    assert(admitsTerminalNoSupportCheck("mantequillabb no"));
    assert(!admitsTerminalNoSupportCheck("mantequillabb mejor no"));
    assert(!admitsTerminalNoSupportCheck("aquí está la mantequillabb no"));
    assert(!admitsTerminalNoSupportCheck("no pásame la mantequillabb no"));
    assert(!admitsTerminalNoSupportCheck("quiero mantequillabb no"));
    assert(!admitsTerminalNoSupportCheck("dónde está la mantequillabb no"));

    ButterEvidence spoken{
        .hitCount = 1, .affirmObject = {50, 124, -50}, .affirmArticle = {}, .affirmScore = -2500,
        .frames = 125, .lastSpeech = 100, .affirmative = true, .complete = true, .ranFsg = true, .terminalNo = {},
    };
    spoken.hits[0] = {ButterTerm::Object, {48, 92, -1400}};
    assert(acceptsAcousticButter(spoken));

    auto extra = spoken;
    extra.hits[0].segment.start = 37;
    assert(!acceptsAcousticButter(extra));
    extra.hits[0].type = ButterTerm::ArticleObject;
    assert(!acceptsAcousticButter(extra));
    extra.affirmArticle = {39, 49, 0};
    assert(acceptsAcousticButter(extra));
    extra.hits[0].segment.end = 87;
    assert(!acceptsAcousticButter(extra));

    extra = spoken;
    extra.opposing = true;
    extra.opposeScore = -2600;
    assert(!acceptsAcousticButter(extra));
    extra.opposeScore = -3000;
    assert(acceptsAcousticButter(extra));

    extra = spoken;
    extra.hitCount = 3;
    extra.hits[1] = {ButterTerm::Object, {52, 92, -1600}};
    extra.hits[2] = {ButterTerm::Opposition, {80, 98, -1500}};
    assert(!acceptsAcousticButter(extra));
    extra.hits[2].segment.end = 97;
    assert(acceptsAcousticButter(extra));
    extra.hitCount = 0;
    assert(!acceptsAcousticButter(extra));

    extra = spoken;
    extra.complete = false;
    assert(!acceptsAcousticButter(extra));
    extra = spoken;
    extra.affirmative = false;
    assert(!acceptsAcousticButter(extra));
    extra = spoken;
    extra.affirmObject.end = 73;
    assert(!acceptsAcousticButter(extra));
    extra = spoken;
    extra.hits[0].segment.start = 75;
    assert(!acceptsAcousticButter(extra));
    extra = spoken;
    extra.hits[0].segment.score = -1801;
    assert(!acceptsAcousticButter(extra));
    std::cout << "BUTTER_GATE_HOST timing, opposition, cancellation and fail-closed coverage\n";
}
