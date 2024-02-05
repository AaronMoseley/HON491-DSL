import math

#Positive for player 1
#Negative for player 2

#0 - ocean, 1 - land
#2 - settlement, 3 - town, 4 - barracks
#5 - settler, 6 - worker, 7 - soldier

class bcolors:
    #HEADER = '\033[95m'
    PLAYER1 = '\033[94m'
    #OKCYAN = '\033[96m'
    EMPTY = '\033[92m'
    #WARNING = '\033[93m'
    PLAYER2 = '\033[91m'
    ENDC = '\033[0m'
    #BOLD = '\033[1m'
    #UNDERLINE = '\033[4m'

chars = [" ", ".", "♔", "♕", "♖", "♙", "♗", "♘"]

def displayLevel(levelState):
    rows = len(levelState)
    cols = len(levelState[0])

    endSpaces = ''.join([" " for _ in range(math.ceil(cols / 26))])
    if cols > 26:
        endSpaces += " "

    print("\t  ", end="")
    for i in range(cols):
        colChar = chr(((i) % 26) + 65)
        mult = math.ceil((i + 1) / 26)

        print(''.join([colChar for _ in range(mult)]), end=''.join([" " for _ in range(len(endSpaces) - mult + 1)]))

    print("\n\t  " + ''.join(["-" + endSpaces for _ in range(cols)]))

    for i, row in enumerate(levelState):
        print(str(i) + "\t" + "| ", end="")
        for tile in row:
            if tile > 1:
                color = bcolors.PLAYER1
            elif tile < 0:
                color = bcolors.PLAYER2
            else:
                color = bcolors.EMPTY
            
            print(color + chars[abs(tile)] + bcolors.ENDC, end=endSpaces)
        
        print("|")

    print("\t  " + ''.join(["-" + endSpaces for _ in range(cols)]))