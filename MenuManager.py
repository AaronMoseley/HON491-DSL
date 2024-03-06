import os

def validSelection(userIn):
    if not userIn.isnumeric():
        return False
    
    if int(userIn) < 1 or int(userIn) > 7:
        return False
    
    return True

def printMainMenu():
    os.system("cls")
    print("Welcome to War Games. Please select from the below options. Type \"quit\" to quit.")
    print("1. 2-Player Game")
    print("2. Play Against Min-Max Search AI")
    print("3. Play Against Reinforcement Learning AI")
    print("4. Min-Max vs. Min-Max")
    print("5. Min-Max vs. Reinforcement Learning")
    print("6. Reinforcement Learning vs. Reinforcement Learning")
    print("7. Help Menu")

def printHelpMenu():
    os.system("cls")
    print("This is the help menu")
    print("Type anything to return to the menu")