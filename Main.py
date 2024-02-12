import Display
import LevelManager
import MinMaxAI
import os
import sys

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

turnCounter = 0
initMat, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName)
while LevelManager.checkWinCond(initMat) == 0:
    Display.displayLevel(initMat, turnCounter, False)
    player1Move = input("What move does player 1 make? ")

    if player1Move.lower() != "skip":
        parsedP1Move = LevelManager.parseMove(initMat, player1Move, 1)

        while parsedP1Move == None:
            player1Move = input("Invalid move! Try again: ")
            parsedP1Move = LevelManager.parseMove(initMat, player1Move, 1) 

        initMat, buildingTurnCounter = LevelManager.makeMove(initMat, buildingTurnCounter, parsedP1Move)

    Display.displayLevel(initMat, turnCounter, True)
    player2Move = input("What move does player 2 make? ")

    if player2Move.lower() != "skip":
        parsedP2Move = LevelManager.parseMove(initMat, player2Move, -1)

        while parsedP2Move == None:
            player2Move = input("Invalid move! Try again: ")
            parsedP2Move = LevelManager.parseMove(initMat, player2Move, -1)

        initMat, buildingTurnCounter = LevelManager.makeMove(initMat, buildingTurnCounter, parsedP2Move)

    initMat, buildingTurnCounter = LevelManager.incrementTurn(initMat, buildingTurnCounter)
    turnCounter += 1

winner = LevelManager.checkWinCond(initMat)
if winner == 1:
    print("Congratulations player 1 for winning!")
else:
    print("Congratulations player 2 for winning!")