from map import Map
from item import Item
from robotType import RobotType
import matplotlib.pyplot as plt
import random
from ui import UI
import sys
import time
import re
import threading

class Simulation(object):
    def __init__(self, createSim,mapSize,IMllm, randomPlacement=False, wantWaitzones=False, obstacles=[], amountFast=0,amountGripper=0, amountFrontLoader=0, maxStep=10000):
        self._createSim = createSim
        if self._createSim == True:
            self._mapSize = mapSize
            self._mymap = None
            self._randomPlacement=randomPlacement
            self._waitZones = wantWaitzones
            self._obstacles = obstacles
            self._amountFast = amountFast
            self._amountGripper = amountGripper
            self._amountFrontLoader = amountFrontLoader
            self._robotAmount = amountFast + amountGripper + amountFrontLoader
            self._maxStep = maxStep
            self._robotTypes = []
            self._step = 0
            self._tasks = []
            self._ui = UI()
            self._uiTasks = []
            self._uiConversation = []
            self._taskID = 0
            self._IMllm = IMllm

    def sim(self):
        if self._createSim == True:
            mapSize = self._mapSize
            randomPlacement = self._randomPlacement
            waitZones = self._waitZones
            amountFast = self._amountFast
            amountGripper = self._amountGripper
            amountFrontLoader = self._amountFrontLoader
            obstacles= self._obstacles
            # ---------- END OF SIMULATION PARAMS ----------
            if randomPlacement is True:
                pos = self.generateRandomStartPos(amountGripper + amountFast + amountFrontLoader, mapSize)
                fast = RobotType(robotType="fast", amount=amountFast, color="blue", posVec=pos[0:amountFast])
                gripper = RobotType(robotType="grip", amount=amountGripper, color="red",
                                    posVec=pos[amountFast:amountFast + amountGripper])
                frontLoader = RobotType(robotType="load", amount=amountFrontLoader, color="green",
                                        posVec=pos[amountFast + amountGripper:amountFast + amountGripper + amountFrontLoader])
                self._robotTypes.append(fast)
                self._robotTypes.append(gripper)
                self._robotTypes.append(frontLoader)
            else:
                posFast = [(0, 3), (6, 0)]
                posGripper = [(0, 0), (0, 6)]
                posFrontLoader = [(1, 1), (2, 6)]
                fast = RobotType(robotType="fast", amount=amountFast, color="blue", posVec=posFast)
                gripper = RobotType(robotType="grip", amount=amountGripper, color="red", posVec=posGripper)
                frontLoader = RobotType(robotType="load", amount=amountFrontLoader, color="green", posVec=posFrontLoader)
                self._robotTypes.append(fast)
                self._robotTypes.append(gripper)
                self._robotTypes.append(frontLoader)
            allRobotTypes = [fast, gripper, frontLoader]
            mymap = Map(size_x=mapSize[0], size_y=mapSize[1], robotTypes=[fast, gripper, frontLoader],obstacles=obstacles, wantWaitzones=waitZones)
            self._mymap = mymap

            # Items
            leaves = Item("Leaves", (3, 3))
            bowl = Item("Bowl",(6,2))
            apple = Item("Apple",(6,2))
            #self._mymap.addMultipleItemsToItemList([leaves,apple,bowl])
            self._mymap.addMultipleItemsToItemList([leaves,bowl])
            
            
            # Start of simulation
            self._mymap.showMap()
            while True:
                
                loadUi = threading.Thread(target=self.loadUIInput)
                loadUi.start()
                
                #start new tasks if new tasks are available
                self.startReadyTasks(self._mymap)
                if self._ui.simRunning is True:
                    #break if sim has reached maxStep
                    if self._mymap.getStep() > self._maxStep:
                        break
                    #move all robots
                    for robotType in allRobotTypes:
                        robotType.moveAllRobots(self._IMllm,self._mymap)

                    #check if the tasks are complated
                    self.checkIfTasksAreCompleted()

                    #Update sim variable step and show map
                    self._step = mymap.getStep()
                    self._mymap.showMap()
                else:
                    time.sleep(0.5)
                if self._ui.exit == True:
                    print("Exiting program")
                    plt.close("all")
                    sys.exit(0)
            #plt.show()

    def generateRandomStartPos(self, amount, mapSize):
        pos = []
        while len(pos) < amount:
            x = random.randrange(0, mapSize[0])
            y = random.randrange(1, mapSize[1])

            # To make sure no two robots are placed on top of eachother
            check = 0
            # print(x,y)
            for i in range(len(pos)):
                if (x, y) == pos[i]:
                    check += 1
            if check == 0:
                pos.append((x, y))
        return pos

    def getStep(self):
        return self._step

    def getAmountOfRobots(self):
        return self._robotAmount
    
    def newTaskID(self):
        self._taskID += 1

    def getAmountOfRobotTypes(self):
        amount = 0
        if self._amountFast > 0:
            amount += 1
        if self._amountGripper > 0:
            amount += 1
        if self._amountFrontLoader > 0:
            amount += 1
        return amount

    def getPosOfAllRobots(self):
        pos = []
        for robotT in self._robotTypes:
            for robot in robotT.getRobots():
                pos.append(robot.getPos())
        return pos

    def defineTask(self, taskText):
        task = {
            "task": taskText,
            "taskID" : self._taskID,
            "robotType": None,
            "waypoints": None,
            "status": "Not Started",
            "robot": None,
            "IMMessages" : [],
            "state" : 0
        }
        print("newtask", task)
        self.newTaskID()
        self._tasks.append(task)

    def startReadyTasks(self,mymap): 
        processingTasks = self.getProcessingTasks()
        readyTasks = []
        for taskObject in processingTasks:
            taskText,taskID,robotType,waypoints,status,robot, IMMessages, state = self.getTaskInfo(taskObject)
            if robotType != None:
                readyTasks.append(taskObject)
                
        if len(readyTasks) > 0:
            for taskObject in readyTasks:
                taskText,taskID,robotType,waypoints,status,robot, IMMessages, state = self.getTaskInfo(taskObject)
                for robotT in self._robotTypes:
                    if robotT._robotType == robotType and "Idle" in robotT.getStateofRobots():
                        robot = robotT.deployRobot(taskObject["waypoints"], self._mymap)
                        taskObject["robot"] = robot
                        print("Starting task:", taskObject)
                        overview = robot.IMOverview(taskObject)
                        IMMessages = self.generateStartIMMessages(taskText=taskText,overview=overview)
                        self.changeTask(taskID,robotType,waypoints,"Ongoing",robot,IMMessages,state)
                        robot.setState("Busy")
                        robot.setTaskObject(taskObject)
                        robot.IM(IMllm=self._IMllm,mymap=mymap)
                    else:
                        if robotT._robotType == robotType:
                            print("Cannot start the following task at moment, please wait.", taskObject)

    def checkIfTasksAreCompleted(self):
        ongoingTasks = self.getOngoingTasks()
        if len(ongoingTasks) > 0:
            for task in ongoingTasks:
                taskText,taskID,robotType,waypoints,status,robot, IMMessages, state = self.getTaskInfo(task)
                if len(robot.getWaypoints()) == 0 and robot.getTarget() == robot.getPos():
                    self.changeTask(taskID,robotType,waypoints,"Completed",robot, IMMessages, state)
    

    def getTaskInfo(self, task):
        taskText = task["task"]
        taskID = task["taskID"]
        robotType = task["robotType"]
        waypoints = task["waypoints"]
        status = task["status"]
        robot = task["robot"]
        IMMessages = task["IMMessages"]
        state = task["state"]
        return taskText,taskID,robotType,waypoints,status,robot, IMMessages, state
    

    def getTasks(self):
        return self._tasks

    def getNotStartedTasks(self):
        notStartedTasks = []
        for task in self.getTasks():
            if task["status"] == "Not Started":
                notStartedTasks.append(task)
        return notStartedTasks
    
    def getProcessingTasks(self):
        processingTasks = []
        for task in self.getTasks():
            if task["status"] == "Processing":
                processingTasks.append(task)
        return processingTasks
    
    def getOngoingTasks(self):
        ongoingTasks = []
        for task in self.getTasks():
            if task["status"] == "Ongoing":
                ongoingTasks.append(task)
        return ongoingTasks

    def getCompletedTasks(self):
        completedTasks = []
        for task in self.getTasks():
            if task["status"] == "Completed":
                completedTasks.append(task)
        return completedTasks

    def decodeLLMOutput(self,task, llmOutput): 
        print("Task:",task)
        print("finishObs:",llmOutput)
        taskText,taskID,robotType,waypoints,status,robot, IMMessages, state = self.getTaskInfo(task)
        robotType = llmOutput["selected_robot"]
        if robotType == "fast_transportation_robot" or robotType == "1":
            robotType = "fast"
        elif robotType == "robot_with_gripper" or robotType == "2":
            robotType = "grip"
        elif robotType == "robot_with_front_loader" or robotType == "3":
            robotType = "load"
        print("RobotType:",robotType)
        waypoints = self.getLLMWaypoints(llmOutput["EXTRACTED_COORDINATE(S)"])
        print("Waypoints:",waypoints)
        self.changeTask(taskID,robotType,waypoints,status,robot,IMMessages, state)
        print("Updated Task:",task)

    def getLLMWaypoints(self, llmOutput):
        coordinates = []
        waypoints = []
        coordinates = re.findall(r'\d+',llmOutput)
        for i in range(0,len(coordinates),2):
            x = int(coordinates[i])
            y = int(coordinates[i+1])
            waypoints.append((x,y))
        return waypoints

    def loadUIInput(self):
        if self._ui.taskReady is True:
            uiTask = self._ui.text[6:-1]
            self._uiTasks.append(uiTask)
            self.defineTask(taskText=uiTask)
            self._ui.taskReady = False
        elif self._ui.conversationReady is True:
            uiConversation = self._ui.text[14:-1]
            self._uiConversation.append(uiConversation)
            self._ui.conversationReady = False

    def getConversationList(self):
        return self._uiConversation

    def getUITasks(self):
        return self._uiTasks
    
    def changeTask(self, taskID, robotType, waypoints, status, robot, IMMessages, state):
        for task in self.getTasks():
            if task["taskID"] == taskID:
                task["robotType"] = robotType
                task["waypoints"] = waypoints
                task["status"] = status
                task["robot"] = robot
                task["IMMessages"] =IMMessages
                task["state"] = state
                
                                
    def getStatesOfAllRobots(self):
        states = []
        for robotType in self._robotTypes:
            typeStates = robotType.getStateofRobots()
            states.append((robotType.getRobotType(),typeStates))
        return states
    
    def getPositionsOfAllRobots(self):
        positions = []
        for robotType in self._robotTypes:
            typePositions = robotType.getPositionOfRobots()
            positions.append((robotType.getRobotType(),typePositions))
        return positions
    
    
    def generateStartIMMessages(self,taskText,overview):
        messages = [
        {
            "role": "system",
            "content": "An overview and steps of a task is given to you. Your job is only to choose the correct function, depeding on the step in the overview."
        },
        {
            "role": "user",
            "content": "Task: " + taskText
        },
        {
            "role": "user",
            "content": "Overview with steps of the task: \"" + overview + "  \""
        },
        {
            "role": "user",
            "content": "Please choose the correct function for step 1 in the overview."
        }
        ]
        return messages