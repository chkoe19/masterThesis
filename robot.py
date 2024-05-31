import networkx as nx
from llama_cpp import Llama
from main import toolsIM,succesDetector,returnModelPath,readPromptsFromTxt
import json
modelPath = returnModelPath()


class Robot(object):
    def __init__(self, pos, color,robotType, name):
        self._pos = pos
        self._prevPos = None
        self._color = color
        self._robotType = robotType
        self._name = name
        self._waypoints = []
        self._target = None
        self._path = []
        self._inventory = []
        self._state = "Idle"
        self._stuck = 0
        self._taskObject = None

    def setTaskObject(self, taskObject):
        self._taskObject = taskObject
        
    def getTaskObject(self):
        return self._taskObject

    def getPos(self):
        return self._pos

    def getColor(self):
        return self._color

    def getName(self):
        return self._name

    def getTarget(self):
        return self._target

    def setWaypoints(self, newWaypoints):
        self._waypoints = newWaypoints

    def getWaypoints(self):
        return self._waypoints

    def setTarget(self):
        waypoints = self.getWaypoints()
        if len(waypoints) > 0:
            newTarget = self.getWaypoints().pop(0)
            self._target = newTarget

    def calcPath(self, mymap):
        tempMap = mymap.getPathPlanningMap(self.getPos(),self.getTarget())
        if self._stuck > 1:
            if not self.getTarget() == self.getPath()[0]:
                tempMap.remove_node(self.getPath()[0])
        self._path = nx.shortest_path(tempMap, source=self.getPos(), target=self.getTarget())
        self._path.pop(0)

    def getPath(self):
        return self._path

    def setPath(self, newPath):
        self._path = newPath

    def move(self, mymap):
        if len(self.getPath()) > 0:
            if not self.getPath()[0] in mymap.getRobotPositions(): #Do not move onto places where other robots are placed
                self._prevPos = self.getPos()
                self._pos = self.getPath()[0]
                self.stepTaken(mymap)
                self._stuck = 0
                return
            else:
                self._stuck += 1
                print("robot in my path")
            if self._stuck > 1:
                self.calcPath(mymap)

    def stepTaken(self, mymap):
        self._path.pop(0)
        robotPos = self.getPos()
        if robotPos == self.getTarget():
            self.setState("Idle")
            if robotPos in mymap.getItemsPos():
                for item in mymap.getItemList():
                    if robotPos == item.getPos() and self._inventory is None:
                        self._inventory = item
                        item.grabItem(self._robotType)
            elif self._inventory != None:
                inventoryItem = self._inventory
                inventoryItem.placeItem(robotPos)
                self._inventory = None
            self.setTarget() #Sets new target if such exists
            if robotPos != self.getTarget():
                self.calcPath(mymap)
        if self._inventory is not None:
            self._inventory.updatePos(robotPos)

    def getState(self):
        return self._state

    def getPrevPos(self):
        return self._prevPos

    def setName(self,newName):
        self._name = newName

    def setState(self, newState):
        self._state = newState
        name = self.getName()[:-1]
        if newState == "Busy":
            self.setName(name+"B")
        elif newState == "Idle":
            self.setName(name+"I")



    def IM(self, IMllm, mymap):
        IMllm.reset()
        taskText,taskID,robotType,waypoints,status,robot, IMMessages, state = self.getTaskInfo(self.getTaskObject())
        robotPos = self.getPos()
        functionName = "None"
        succes = ""
        
        act = IMllm.create_chat_completion(messages=IMMessages, max_tokens=1000, temperature=0.0, top_p=1.0,
                        repeat_penalty=1.1, top_k=1, tools=toolsIM,tool_choice="auto")
        
        IMMessages.pop()
        
        functionName = act["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        print(functionName)
        
        if functionName == "move_robot":
            
            self.setTarget() #Sets new target if such exists
            if robotPos != self.getTarget():
                self.calcPath(mymap)
            
            IMMessages.append({
            "role": "user",
            "content": "Please choose the correct function for step " + str(state+2) + " in the overview."})
            print(IMMessages[-1]["content"])
        
        elif functionName == "pick_up_object":
            sceneDetector = ["nothing"]
            if robotPos in mymap.getItemsPos():
                for item in mymap.getItemList():
                    if robotPos == item.getPos():
                        sceneDetector.append(item.getName())
            
            print("Scene detector:", sceneDetector)
            messagesPickUp = [
            {
            "role": "user",
            "content": "A task and a list of objects are given to you. You should choose the most appropriate object for the task. If there are no appropriate object, you should just choose \'nothing\' from the list of objects. An example could be, Task: Move banana from 1,3 to 2,7. List of objects: [\'nothing\', \'banana\']. The chosen object should be banana, since the task says so. Another example could be, Task: Move banana from 2,6 to 1,2. List of objects: [\'nothing\', \'apple\']. The chosen object should be nothing, since there are no appropriate object in the list of objects."
            },
            {
            "role": "user",
            "content": "Task: " + taskText + "List of objects: " + str(sceneDetector)
            }
            ]
            act = IMllm.create_chat_completion(messages=messagesPickUp, max_tokens=1000, temperature=0.0, top_p=1.0,
                        repeat_penalty=1.1, top_k=1, tools=toolsIM,tool_choice={"type": "function", "function": {"name": "pick_up_object"}})
            act_args = json.loads(act["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],strict=False)
            itemName = act_args["object"]

            succes = succesDetector(0.8)
            
            if succes == True:
                print("Action: Succes")
                print("Robot picked up:",itemName)
                for item in mymap.getItemList():
                        if robotPos == item.getPos() and itemName == item.getName():
                            item.grabItem(self._robotType)
                            self._inventory.append(item)
                IMMessages.append({
                "role": "user",
                "content": "Please choose the correct function for step " + str(state+2)+ " in the overview."})
                print(IMMessages[-1]["content"])
            else:
                print("Action: Failed")
                state -= 1
                IMMessages.append({
                "role": "user",
                "content": "Please choose the correct function for step " + str(state+2) + " in the overview."})
                print(IMMessages[-1]["content"])
        
        elif functionName == "place_object":
            succes = succesDetector(0.9)
            
            if succes == True:
                print("Action: Succes")
                item = self._inventory.pop()
                print("Robot placed:",item.getName())
                item.placeItem(robotPos)
                IMMessages.append({
                "role": "user",
                "content": "Please choose the correct function for step " + str(state+2)+ " in the overview."})
                print(IMMessages[-1]["content"])
            else:
                print("Action: Failed")
                state -= 1
                IMMessages.append({
                "role": "user",
                "content": "Please choose the correct function for step " + str(state+2)+ " in the overview."})
                print(IMMessages[-1]["content"])
        

        elif functionName == "task_completed":
            self._target = None
            self.setState("Idle")
        
        state += 1
        self.changeTask(robotType,waypoints,status,robot, IMMessages, state)
    
    def takeStep(self,mymap):
        if len(self.getPath()) > 0:
            if not self.getPath()[0] in mymap.getRobotPositions(): #Do not move onto places where other robots are placed
                self._prevPos = self.getPos()
                self._pos = self.getPath()[0]
                self._path.pop(0)
                self._stuck = 0
            else:
                self._stuck += 1
                print("robot in my path")
            if self._stuck > 1:
                self.calcPath(mymap)
            if self._inventory != []:
                for item in self._inventory:
                    item.updatePos(self._pos)
                
                
                
    def IMOverview(self, taskObject):
        taskText = taskObject["task"]
        robotPos = self.getPos()
        llmExplain = Llama(
        model_path=modelPath,
        seed = 1,
        n_threads=4, # CPU cores
        n_ctx = 1024, # Allows for larger message history
        n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        n_gpu_layers=-1, # Change this value based on your model and your GPU VRAM pool.
        verbose=False # True for messages from Llama)
        )
        
        systemPrompt = readPromptsFromTxt("Inner Monologue/IMoverview.txt")[0]
        
        llmExplain.reset()
        messagesOverviewIM = [
            {
            "role": "system",
            #"content" : "You are a robot task planner and when a task and robot position is provided to you, you must explain which actions are needed to solve the task. The task may consist of multiple of the following actions: move_robot, pick_up_object, place_object and task_completed. Please include the necessary steps in the taskplanning with the possible actions presented before. It is only necessary to use the actions so the robot can complete the task, so if the robot do not need to move the actions move_robot should not be used. Remember that the action task_complete should always be used when the task is complete and should be part of the steps. Please keep your output as concise as possible and it is not needed to give any explanations of actions chosen. Please do not include any coordinates as they will not be used. The next thing given to you will be the task and the robot position."
            "content": systemPrompt
            },
            {
            "role": "user",
            "content": "The task is:" + taskText + " Robot position: " + str(robotPos) +"."
            },
        ]
        
        first = True
        stringOut = ""
        for out in llmExplain.create_chat_completion(messages=messagesOverviewIM, max_tokens=1000, temperature=0.0, top_p=1.0, repeat_penalty=1.1, top_k=1,stream=True):
            if first is True:
                first = False
            elif out["choices"][0]["finish_reason"] == "stop":
                print("\n")
                break
            else:
                stringOut += out["choices"][0]["delta"]["content"]
                print(out["choices"][0]["delta"]["content"],sep='',end="")
        return stringOut
    
    
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
    
    def changeTask(self, robotType, waypoints, status, robot, IMMessages, state):
        self._taskObject["robotType"] = robotType
        self._taskObject["waypoints"] = waypoints
        self._taskObject["status"] = status
        self._taskObject["robot"] = robot
        self._taskObject["IMMessages"] = IMMessages
        self._taskObject["state"] = state