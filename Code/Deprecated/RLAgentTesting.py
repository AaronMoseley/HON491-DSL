import RLModel
import LevelManager
import torch
import os
import Display
import RLAgentSupport

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#model = RLModel.WarGamesAI(device).to(device)

levelLoc = "Levels/Testing/"

print("\nAvailable Levels: ")
print(*os.listdir(levelLoc), sep=", ")
levelName = input("Which level do you want to play? ")
while(levelName not in os.listdir(levelLoc)):
    print(*os.listdir(levelLoc), sep=", ")
    levelName = input("Invalid level name. Which level do you want to play? ")

turnCounter = 0
levelState, buildingTurnCounter = LevelManager.readLevel(levelLoc + levelName, spawnUnits=False)

Display.displayLevel(levelState, turnCounter, False)

levelState, buildingTurnCounter = RLAgentSupport.randomizeState(levelState, buildingTurnCounter, 0.15)

Display.displayLevel(levelState, turnCounter, False)

print(RLAgentSupport.rlLoss(levelState, 1))

#print(model(levelState, -1))