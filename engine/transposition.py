TT = {}

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

def store(key, depth, value, flag, best_move):
    TT[key] = (depth, value, flag, best_move)

def lookup(key):
    return TT.get(key, None)

def clear():
    TT.clear()
