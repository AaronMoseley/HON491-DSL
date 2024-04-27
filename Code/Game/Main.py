import Display
import LevelManager
import MinMaxAI
import MenuManager
import os
import sys
import random
import time
import NewRLModel
import torch
import copy

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
os.system("cls")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

modelLoc = "Models/RLModel5000BudgetEpoch100"
model = NewRLModel.MinMaxWarGamesAI(device)
model.load_state_dict(torch.load(modelLoc))
model.eval()

levelLoc = ""

modelPlayer = 1

testing = False
if testing:
    levelLoc = "Levels/Testing/"
else:
    levelLoc = "Levels/Final/"

userIn = ""
while True:
    MenuManager.printMainMenu()
    userIn = input()

    if userIn.lower() == "quit":
        break

    while not MenuManager.validSelection(userIn):
        print("Invalid selection, please try again")
        userIn = input()

    #0: user, 1: min-max, 2: reinforcement learning, 3: random
    userTypes = []
    if userIn == "help":
        MenuManager.printHelpMenu()
        userIn = input()
        continue
    else:
        userTypes = [int(userIn.split()[0]), int(userIn.split()[1])]

    print("\nAvailable Levels: ")
    print(*os.listdir(levelLoc), sep=", ")
    levelName = input("Which level do you want to play? ")
    while(levelName not in os.listdir(levelLoc)):
        print(*os.listdir(levelLoc), sep=", ")
        levelName = input("Invalid level name. Which level do you want to play? ")

    turnCounter = 0
    levelState, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName)

    movesMade = 0
    currPlayer = -1
    while LevelManager.checkWinCond(levelState) == 0:
        currPlayer *= -1
        levelState, buildingTurnCounter = LevelManager.incrementTurn(levelState, buildingTurnCounter)

        if currPlayer == 1:
            turnCounter += 1
        
        Display.displayLevel(levelState, turnCounter, False)

        userType = userTypes[0 if currPlayer == 1 else 1]
        if userType == 0:
            if currPlayer > 0:
                print("Player 1 (blue):")
            else:
                print("Player 2 (red):")

            if len(MinMaxAI.getValidMoves(levelState, currPlayer)) == 0:
                playerMove = input("No valid moves can be made. Type anything to continue.\n")
                continue

            playerMove = input("What move do you make? Type \"skip\" to skip your turn or \"quit\" to exit the game.\n")

            if playerMove.lower() == "quit":
                break

            if playerMove.lower() == "skip":
                continue

            parsedMove = LevelManager.parseMove(levelState, playerMove, currPlayer)
            while parsedMove == None and playerMove.lower() != "skip" and playerMove.lower() != "quit":
                playerMove = input("Invalid move! Try again: ")
                parsedMove = LevelManager.parseMove(levelState, playerMove, currPlayer) 

            if playerMove.lower() == "quit":
                break

            if playerMove.lower() == "skip":
                continue
        elif userType == 1:
            parsedMove = MinMaxAI.chooseMove(levelState, buildingTurnCounter, currPlayer, 10000)
        elif userType == 2:
            if currPlayer == 1:
                parsedMove = model([levelState], [buildingTurnCounter], modelPlayer, returnMove=True, stateBudget=5000)[0]
            else:
                reverseState = copy.deepcopy(levelState)
                reverseState.reverse()

                reverseTurnCounter = copy.deepcopy(buildingTurnCounter)
                reverseTurnCounter.reverse()

                for i in range(len(reverseState)):
                    for j in range(len(reverseState[i])):
                        if abs(reverseState[i][j]) > 1:
                            reverseState[i][j] *= -1
                            reverseTurnCounter[i][j] *= -1

                parsedMove = model([reverseState], [reverseTurnCounter], modelPlayer, returnMove=True, stateBudget=5000)[0]

                if parsedMove != None:
                    parsedMove[0][0] = 7 - parsedMove[0][0]
                    parsedMove[1][0] = 7 - parsedMove[1][0]
        elif userType == 3:
            random.seed(time.time())
            
            parsedMove = MinMaxAI.getValidMoves(levelState, currPlayer)
            if len(parsedMove) == 0:
                parsedMove = None
            else:
                parsedMove = random.choice(parsedMove)

        if parsedMove != None:
            levelState, buildingTurnCounter = LevelManager.makeMove(levelState, buildingTurnCounter, parsedMove)
            movesMade += 1
            print("Moves Made: " + str(movesMade))

        if userType != 0:
            Display.displayLevel(levelState, turnCounter, False)
            if currPlayer > 0:
                print("Player 1 (blue):")
            else:
                print("Player 2 (red):")
            
            if parsedMove != None:
                print("AI Move: " + str(parsedMove))
            else:
                print("No move made.")
            #_ = input("Type anything to continue. ")

    winner = LevelManager.checkWinCond(levelState)
    if winner == 1:
        print("Congratulations player 1 for winning!")
    elif winner == -1:
        print("Congratulations player 2 for winning!")

    _ = input()