import LevelManager
import MinMaxAI
import copy
import NewRLModel
import torch
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

for modelBudget in [5000]:
    modelName = os.path.dirname(__file__) + "\\Models\\Final\\RLModelSelfPlay" + str(modelBudget) + "Budget"
    model = NewRLModel.MinMaxWarGamesAI(device).to(device)
    model.load_state_dict(torch.load(modelName))
    model.train()

    for minMaxBudget in [1000, 2500, 5000, 10000]:
        gamesWon = 0
        totalGames = 0
        for levelName in os.listdir(os.path.dirname(__file__) + "\\Levels\\Final\\Unseen\\"):
            for modelPlayer in [1, -1]:
                levelState, buildingTurnCounter = LevelManager.readLevel(os.path.dirname(__file__) + "\\Levels\\Final\\Unseen\\" + levelName)
               
                turnCounter = 0
                currPlayer = -1
                while LevelManager.checkWinCond(levelState) == 0:
                    currPlayer *= -1
                    levelState, buildingTurnCounter = LevelManager.incrementTurn(levelState, buildingTurnCounter)

                    if currPlayer == -1:
                        turnCounter += 1

                    parsedMove = None
                    if len(MinMaxAI.getValidMoves(levelState, currPlayer)) > 0:
                        if currPlayer == modelPlayer:
                            tempState = copy.deepcopy(levelState)
                            tempCounter = copy.deepcopy(buildingTurnCounter)
                            
                            if modelPlayer == -1:
                                tempState.reverse()
                                tempCounter.reverse()

                                for i in range(len(tempState)):
                                    for j in range(len(tempState[i])):
                                        if abs(tempState[i][j]) > 1:
                                            tempState[i][j] *= -1
                                            tempCounter[i][j] *= -1

                            parsedMove = model([tempState], [tempCounter], 1, returnMove=True, stateBudget=modelBudget)
                            if parsedMove != None:
                                parsedMove = parsedMove[0]
                                if modelPlayer == -1:
                                    parsedMove[0][0] = 7 - parsedMove[0][0]
                                    parsedMove[1][0] = 7 - parsedMove[1][0]
                        else:
                            parsedMove = MinMaxAI.chooseMove(levelState, buildingTurnCounter, currPlayer, stateBudget=minMaxBudget)

                    if parsedMove != None:
                        levelState, buildingTurnCounter = LevelManager.makeMove(levelState, buildingTurnCounter, parsedMove)

                winCond = LevelManager.checkWinCond(levelState)
                if winCond == modelPlayer:
                    gamesWon += 1
                totalGames += 1
        
        print("Model: " + str(modelBudget) + " vs. MinMaxAI: " + str(minMaxBudget))
        print("\tGames Won: " + str(gamesWon) + " / " + str(totalGames))