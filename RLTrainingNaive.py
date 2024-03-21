import RLAgentSupport
import RLModel
import MinMaxAI
import LevelManager
import torch
import Display
import copy

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = RLModel.WarGamesAI(device).to(device)
targetNet = RLModel.WarGamesAI(device).to(device)
targetNet.load_state_dict(model.state_dict())

levelLoc = "Levels/Final/"
levelName = "Level1.txt"

numEpochs = 100
learnRate = 0.001
model.train()

maxNumTurns = 200
selfPlayThreshold = 50

mutationPower = 0.1
targetStartUnits = 10

modelSavePath = "Models/"
modelName = "TestRLModel1"
checkpointEpoch = 5

maxReplayLen = 200
batchSize = 10
replay = []

gamesWon = 0
for epoch in range(numEpochs):
    levelState, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName, spawnUnits=False)
    levelState, buildingTurnCounter = RLAgentSupport.randomizeState(levelState, buildingTurnCounter, targetStartUnits / (len(levelState) * len(levelState[0])))

    optimizer = torch.optim.SGD(model.parameters(), lr=learnRate)
    modelPlayer = 1 if epoch % 2 == 0 else -1
    opposingPlayerType = 1 if epoch < selfPlayThreshold else 2

    if opposingPlayerType != 1:
        mutatedNetwork = RLModel.WarGamesAI(device).to(device)
        for param in mutatedNetwork.parameters():
            param.data += mutationPower * torch.randn_like(param)

    turnCounter = 0
    currPlayer = -1
    while LevelManager.checkWinCond(levelState) == 0 and turnCounter <= maxNumTurns:
        currPlayer *= -1
        levelState, buildingTurnCounter = LevelManager.incrementTurn(levelState, buildingTurnCounter)

        if currPlayer == -1:
            turnCounter += 1

        parsedMove = None
        if len(MinMaxAI.getValidMoves(levelState, currPlayer)) > 0:
            if currPlayer == modelPlayer:
                parsedMove = model(levelState, modelPlayer)
            else:
                if opposingPlayerType == 1:
                    parsedMove = MinMaxAI.chooseMove(levelState, buildingTurnCounter, currPlayer, stateBudget=10000)
                else:
                    parsedMove = mutatedNetwork(levelState, currPlayer)

        if parsedMove != None:
            newState = copy.deepcopy(levelState)
            newTurnCounter = copy.deepcopy(buildingTurnCounter)

            newState, newTurnCounter = LevelManager.makeMove(newState, newTurnCounter, parsedMove)
            
            if currPlayer == modelPlayer:
                optimizer.zero_grad()
                loss = RLAgentSupport.rlLoss(newState, levelState, modelPlayer)
                loss.backward()
                optimizer.step()

                print("Gane: " + str(epoch))
                Display.displayLevel(levelState, turnCounter, False)
                print(loss)
                #_ = input("Type anything to continue")
            
            levelState = copy.deepcopy(newState)
            buildingTurnCounter = copy.deepcopy(newTurnCounter)

    if LevelManager.checkWinCond(levelState) == modelPlayer:
        gamesWon += 1

    print("Games Won: " + str(gamesWon) + " / " + str(epoch + 1))

    if epoch % checkpointEpoch == 0:
        torch.save(model.state_dict(), modelSavePath + modelName + "Epoch" + str(checkpointEpoch))

torch.save(model.state_dict(), modelSavePath + modelName)