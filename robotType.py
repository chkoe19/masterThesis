from robot import Robot

class RobotType(object):
    def __init__(self, robotType, amount, color, posVec =[]):
        self._robotType = robotType
        self._typeAmount = amount
        self._typeColor = color
        self._robots = []
        self._posVec = posVec
        self.setupRobots()

    def getRobotType(self):
        return self._robotType

    def getAmount(self):
        return self._typeAmount

    def getColor(self):
        return self._typeColor

    def addRobot(self, Robot):
        self._robots.append(Robot)

    def setupRobots(self):
        for i in range(self.getAmount()):
            x = self._posVec[i][0]
            y = self._posVec[i][1]
            robot = Robot((x,y), self.getColor(),self.getRobotType(), self.getRobotType()+"-"+ "I")#str(i))
            self.addRobot(robot)

    def getRobots(self):
        return self._robots

    def getIdleRobots(self):
        idleRobots = []
        for robot in self._robots:
            #robot.setState("NotIdle")
            if robot.getState() == "Idle":
                idleRobots.append(robot)
        return idleRobots

    def getClosestRobotToFirstWaypoint(self, waypoints=[], idleRobots=[]):
        target = waypoints[0]
        minDist = 100000
        outRobot = None
        for robot in idleRobots:
            x, y = robot.getPos()
            xT, yT = target
            dist = abs(x-xT) + abs(y-yT)
            if minDist > dist:
                minDist = dist
                outRobot = robot
        return outRobot

    def deployRobot(self, waypoints, mymap):
        idleRobots = self.getIdleRobots()
        if len(idleRobots) == 0:
            print("Error, No available robots - Not implemented yet")
            return None
        if len(idleRobots) == 1:
            robot = idleRobots[0]
            robot.setWaypoints(waypoints)
            return robot
        if len(idleRobots) > 1:
            robot = self.getClosestRobotToFirstWaypoint(waypoints, idleRobots)
        robot.setWaypoints(waypoints)
        return robot

    def moveAllRobots(self,IMllm,mymap):
        for robot in self._robots:
            #robot.move(mymap)
            if robot.getPos() == robot.getTarget():
                robot.IM(IMllm = IMllm,mymap=mymap)
            else:
                robot.takeStep(mymap)

    def getStateofRobots(self):
        states = []
        for robot in self.getRobots():
            state = robot.getState()
            states.append(state)
        return states
    
    def getPositionOfRobots(self):
        positions = []
        for robot in self.getRobots():
            position = robot.getPos()
            positions.append(position)
        return positions