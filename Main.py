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

    userTypes = [0, 0]
    if int(userIn) == 7:
        MenuManager.printHelpMenu()
        userIn = input()
        continue
    else:
        if int(userIn) == 2:
            userTypes = [0, 1]
        elif int(userIn) == 3:
            userTypes = [0, 2]
        elif int(userIn) == 4:
            userTypes = [1, 1]
        elif int(userIn) == 5:
            userTypes = [1, 2]
        elif int(userIn) == 6:
            userTypes = [2, 2]

    print("\nAvailable Levels: ")
    print(*os.listdir(levelLoc), sep=", ")
    levelName = input("Which level do you want to play? ")
    while(levelName not in os.listdir(levelLoc)):
        print(*os.listdir(levelLoc), sep=", ")
        levelName = input("Invalid level name. Which level do you want to play? ")

    turnCounter = 0
    levelState, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName)


    currPlayer = -1
    while LevelManager.checkWinCond(levelState) == 0:
        currPlayer *= -1
        levelState, buildingTurnCounter = LevelManager.incrementTurn(levelState, buildingTurnCounter)

        if currPlayer == 1:
            turnCounter += 1    
        
        Display.displayLevel(levelState, turnCounter, True)

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
            parsedMove = MinMaxAI.chooseMove(levelState, buildingTurnCounter, currPlayer)
        elif userType == 2:
            parsedMove = MinMaxAI.getValidMoves(levelState, currPlayer)
            if len(parsedMove) == 0:
                parsedMove = None
            else:
                parsedMove = parsedMove[0]

        if parsedMove != None:
            levelState, buildingTurnCounter = LevelManager.makeMove(levelState, buildingTurnCounter, parsedMove)

        if userType != 0:
            Display.displayLevel(levelState, turnCounter, True)
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