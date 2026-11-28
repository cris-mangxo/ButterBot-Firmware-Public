#pragma once

namespace EsMx::Pilot {

inline constexpr char ClockGrammar[] = R"clock(#JSGF V1.0 UTF-8;
grammar robot;
public <utterance> = <clock> | <agenda> | <other>;
<clock> = [oye | disculpa | dime | me puedes decir] (qué hora es [ahorita | ahora] | qué horas son | qué hora tienes | dame la hora | dime la hora);
<agenda> = [oye | disculpa | sabes | dime] (a qué horas | a qué hora) [es | sale | llega | empieza | viene]
           [el avión | el vuelo | la cena | el examen | la reunión | el partido | la clase | la cita | el camión] |
           (qué hora es | qué horas son) (de la cita | en madrid | en valencia | de la reunión | mañana) |
           qué hora era [antes | ayer] | no me digas [qué hora es | la hora] | es hora de [comer | salir];
<other> = (tira | avienta | lanza) [un] dado de (seis | diez | veinte) [caras | lados] |
           (dime | échame | cuéntame) un chiste | (dame | pásame) [la] mantequilla |
           qué es eso | es temprano | no sé | [son | es] [la | las] (una | dos | tres | cuatro | cinco | seis | siete | ocho | nueve | diez | once | doce) [y media];)clock";

}
