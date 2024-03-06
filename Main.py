import Display
import LevelManager
import MinMaxAI
import MenuManager
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

userIn = ""
while True:
    MenuManager.printMainMenu()
    userIn = input()

    if userIn.lower() == "quit":
        break

    while not MenuManager.validSelection(userIn):
        print("Invalid selection, please try again")
        userIn = input()

    if int(userIn) == 7:
        MenuManager.printHelpMenu()
        userIn = input()
        continue

    print("\nAvailable Levels: ")
    print(*os.listdir(levelLoc), sep=", ")
    levelName = input("Which level do you want to play? ")
    while(levelName not in os.listdir(levelLoc)):
        print(*os.listdir(levelLoc), sep=", ")
        levelName = input("Invalid level name. Which level do you want to play? ")

    turnCounter = 0
    initMat, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName)


    currPlayer = -1
    while LevelManager.checkWinCond(initMat) == 0:
        currPlayer *= -1
        initMat, buildingTurnCounter = LevelManager.incrementTurn(initMat, buildingTurnCounter)
        Display.displayLevel(initMat, turnCounter, True)

        if currPlayer > 0:
            print("Player 1 (blue):")
        else:
            print("Player 2 (red):")

        if len(MinMaxAI.getValidMoves(initMat, currPlayer)) == 0:
            playerMove = input("No valid moves can be made. Type anything to continue.\n")
            continue

        playerMove = input("What move do you make? Type \"skip\" to skip your turn or \"quit\" to exit the game.\n")

        if playerMove.lower() == "quit":
            break

        if playerMove.lower() == "skip":
            continue

        parsedMove = LevelManager.parseMove(initMat, playerMove, currPlayer)
        while parsedMove == None and playerMove.lower() != "skip" and playerMove.lower() != "quit":
            playerMove = input("Invalid move! Try again: ")
            parsedMove = LevelManager.parseMove(initMat, playerMove, currPlayer) 

        if playerMove.lower() == "quit":
            break

        if playerMove.lower() == "skip":
            continue

        initMat, buildingTurnCounter = LevelManager.makeMove(initMat, buildingTurnCounter, parsedMove)

    winner = LevelManager.checkWinCond(initMat)
    if winner == 1:
        print("Congratulations player 1 for winning!")
    elif winner == -1:
        print("Congratulations player 2 for winning!")