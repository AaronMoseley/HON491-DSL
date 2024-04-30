import math
import os
import LevelManager

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

def displayLevel(levelState, turnCounter, clear):
    if clear:
        os.system("cls")

    print("Turn: " + str(turnCounter + 1))
    rows = len(levelState)
    cols = len(levelState[0])

    print("\t  ", end="")
    for i in range(cols):
        secondChar = " "
        if i < 26:
            firstChar = chr(((i) % 26) + 65)
        else:
            firstChar = chr(math.ceil((i + 1) / 26) + 65)
            secondChar = chr(((i) % 26) + 65)

        print(firstChar + secondChar + " ", end="")

    print("\n\t  " + ''.join(["-  " for _ in range(cols)]))

    for i, row in enumerate(levelState):
        print(str(i) + "\t" + "| ", end="")
        for tile in row:
            if tile > 1:
                color = bcolors.PLAYER1
            elif tile < 0:
                color = bcolors.PLAYER2
            else:
                color = bcolors.EMPTY
            
            print(color + chars[abs(tile)] + bcolors.ENDC, end="  ")
        
        print("|")

    print("\t  " + ''.join(["-  " for _ in range(cols)]))