# bot/states.py

from enum import Enum

class BotStates(Enum):
    WAITING_FOR_NEUROTYPE_DESCRIPTION = None
    DEFAULT = 0
    WAITING_FOR_STATEMENT = 1
    WAITING_FOR_CORRELATIONS_DECISION = 2
    WAITING_FOR_CORRELATIONS_INPUT = 3
    OPROSNIK = 4
    OPROSNIK_PROCESSING = 5
    ADD_STATEMENT = 6
    ADD_CORRELATIONS = 7