import RLAgentSupport
import RLModel
import MinMaxAI
import LevelManager
import torch
import Display

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = RLModel.WarGamesAI(device).to(device)

levelLoc = "Levels/Testing/"
levelName = "TestLevel1.txt"

numEpochs = 2
learnRate = 0.001
model.train()

maxNumTurns = 100
selfPlayThreshold = 1

mutationPower = 0.01
targetStartUnits = 40

modelSavePath = "Models/"
modelName = "TestRLModel1"
checkpointEpoch = 5

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

        if len(MinMaxAI.getValidMoves(levelState, currPlayer)) > 0:
            if currPlayer == modelPlayer:
                parsedMove = model(levelState, modelPlayer)
            else:
                if opposingPlayerType == 1:
                    parsedMove = MinMaxAI.chooseMove(levelState, buildingTurnCounter, currPlayer, stateBudget=1000)
                else:
                    parsedMove = mutatedNetwork(levelState, currPlayer)

        if parsedMove != None:
            levelState, buildingTurnCounter = LevelManager.makeMove(levelState, buildingTurnCounter, parsedMove)
            
            if currPlayer == modelPlayer:
                optimizer.zero_grad()
                loss = RLAgentSupport.rlLoss(levelState, modelPlayer)
                loss.backward()
                optimizer.step()

                Display.displayLevel(levelState, turnCounter, True)
                print(loss)
                #_ = input("Type anything to continue")

    if epoch % checkpointEpoch == 0:
        torch.save(model.state_dict(), modelSavePath + modelName + "Epoch" + str(checkpointEpoch))

torch.save(model.state_dict(), modelSavePath + modelName)