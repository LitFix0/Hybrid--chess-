import time

SEARCH_START = 0.0
TIME_LIMIT = 0.0


def start_search(time_limit):
    """
    Called before every minimax search.
    Resets the timer cleanly for each move.
    """
    global SEARCH_START, TIME_LIMIT
    SEARCH_START = time.perf_counter()
    TIME_LIMIT = time_limit


def should_stop():
    """
    Returns True when the allowed thinking time is exceeded.
    Safe for recursive minimax and repeated self-play games.
    """
    if TIME_LIMIT <= 0:
        return False

    return (time.perf_counter() - SEARCH_START) >= TIME_LIMIT