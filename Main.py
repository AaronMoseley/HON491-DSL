import Display
import LevelManager
import os
import sys

def checkWinCond(levelState):
    hasNeg = False
    hasPos = False
    
    for row in levelState:
        for tile in row:
            if tile < 0 and not hasNeg:
                hasNeg = True
            elif tile > 1 and not hasPos:
                hasPos = True

            if hasNeg and hasPos:
                return 0
            
    if hasPos:
        return 1
    else:
        return -1

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
os.system("cls")

levelLoc = ""

testing = True
if testing:
    levelLoc = "Levels/Testing/"
else:
    levelLoc = "Levels/Final"

print(*os.listdir(levelLoc), sep=", ")
levelName = input("Which level do you want to play? ")
while(levelName not in os.listdir(levelLoc)):
    print(*os.listdir(levelLoc), sep=", ")
    levelName = input("Invalid level name. Which level do you want to play? ")

initMat = LevelManager.readLevel(levelLoc + levelName)
while checkWinCond(initMat) == 0:
    Display.displayLevel(initMat)
    player1Move = input("What move does player 1 make?")
    player2Move = input("What move does player 2 make?")

winner = checkWinCond(initMat)
if winner == 1:
    print("Congratulations player 1 for winning!")
else:
    print("Congratulations player 2 for winning!")