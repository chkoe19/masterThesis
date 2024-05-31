import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches
import matplotlib.image as mpimg
import matplotlib.ticker as ticker
from item import Item

arm=mpimg.imread('simFigures/RobotArmDraw.png')
load =mpimg.imread('simFigures/RobotLoadDraw.png')
fast = mpimg.imread('simFigures/RobotFastDraw.png')

robotImg={"load": (load,0.65), "fast":(fast,0.70), "grip":(arm,0.65)}

leaves=mpimg.imread('simFigures/Leaves.png')
bowl=mpimg.imread('simFigures/Bowl.png')
apple=mpimg.imread('simFigures/Apple.png')
itemsImg={"Leaves":(leaves,0.57), "Bowl":(bowl,0.70), "Apple":(apple,0.65)}

matplotlib.use("TkAgg")
plt.rcParams['figure.figsize'] = [15, 11]
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
class Map(object):
    def __init__(self, size_x, size_y, robotTypes =[],obstacles=[], wantWaitzones=False):
        self._my_map = nx.Graph()
        self._size_x = size_x
        self._size_y = size_y
        self._default_label = ""
        self._default_color = "white"
        self._waitzone_color = "teal"
        self._waitzone_list = []
        self._step = 0
        self._wantWaitzones = wantWaitzones
        self._itemList = []
        self._robotTypes = robotTypes
        self._sleepTime = 5
        self._obstacles = obstacles
        self._fig, self._ax = plt.subplots()
 
        self.setupMap()

    def getPossibleMoves(self,robot):
        robotPos = robot.getPos()
        check = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        possible = []
        for pos in check:
            checkPos = (robotPos[0] + pos[0], robotPos[1] + pos[1])
            if checkPos in self._my_map.nodes:
                if self._my_map.nodes[checkPos]["Label"] == "" or self._my_map.nodes[checkPos]["Label"] == "Free":
                    possible.append(checkPos)
        return possible

    def getRobotPositions(self):
        pos = []
        for robotType in self._robotTypes:
            for robot in robotType.getRobots():
                pos.append(robot.getPos())
        return pos

    def getStatesofAllRobots(self):
        states = []
        for robotType in self._robotTypes:
            for robotState in robotType.getStateofRobots():
                states.append(robotState)
        return states

    def getItemsPos(self):
        pos = []
        for item in self.getItemList():
            pos.append(item.getPos())
        return pos

    def getPathPlanningMap(self, robotPos, target):
        localMap = self._my_map.copy()
        
        for node in self._obstacles:
            localMap.remove_node(node)
        
        for item in self._itemList:
            itemPos = item.getPos()
            
            if target != itemPos and robotPos != itemPos and itemPos in localMap.nodes:
                localMap.remove_node(itemPos)

        return localMap

    def getStep(self):
        return self._step

    def showMap(self):
        plt.gcf()
        while len(self._fig.axes)>1:
            self._fig.axes[1].remove()
        self._ax.clear()
        
        figWidth =  self._size_x
        figHeight = self._size_y
        
        self._ax.set_xlim((0,figWidth))
        self._ax.xaxis.set_major_locator(ticker.IndexLocator(base=1.0,offset=0.5))
        self._ax.set_ylim((0,figHeight))
        self._ax.yaxis.set_major_locator(ticker.IndexLocator(base=1.0,offset=0.5))
        self._ax.set_title(f"Step: {self.getStep()}")
        labels = {}
        robotsAndItems = []
        itemsLargerThanZero = []
        if len(self.getItemList()) > 0:
            for item in self.getItemList():
                name = item.getName()
                pos = item.getPos()
                itemsLargerThanZero.append(pos)
                robotsAndItems.append((item,itemsImg[name]))
                self._my_map.nodes[pos]["Label"] = name
        if self._wantWaitzones is True:
            for waitzone in self.getWaitzoneList():
                self._my_map.nodes[waitzone[0]]["Label"] = waitzone[1]
        if len(self._robotTypes) > 0:
            for robotType in self._robotTypes:
                for robot in robotType.getRobots():
                    pos = robot.getPos()
                    robotsAndItems.append((robot,robotImg[robotType.getRobotType()]))
                    prevPos = robot.getPrevPos()
                    self._my_map.nodes[pos]["Label"] = robot.getName()
                    if prevPos is not None and prevPos not in itemsLargerThanZero and prevPos not in self.getRobotPositions():
                        self._my_map.nodes[prevPos]["Label"] = self._default_label

        for node in nx.nodes(self._my_map):
            labels[node] = self._my_map.nodes[node]["Label"]

        scalex = figWidth/self._size_x
        scaley = figHeight/self._size_y
        pos = {((n[0] * scalex), (n[1]*scaley)) for n in nx.nodes(self._my_map)}
        for node in pos:
            if node in self._obstacles:
                self._ax.add_patch(matplotlib.patches.Rectangle((node[0],node[1]),scalex,scaley,linewidth =1,edgecolor="black",facecolor="black"))
            else:
                self._ax.add_patch(matplotlib.patches.Rectangle((node[0],node[1]),scalex,scaley,linewidth =1,edgecolor="black",facecolor="none"))
        
        trans=self._ax.transData.transform
        trans2=self._fig.transFigure.inverted().transform


        posImg = {n:((n[0] * scalex)+(scalex/2), (n[1]*scaley)+(scaley/2)) for n in nx.nodes(self._my_map)}

        for object,itemImg in reversed(robotsAndItems):
            pos = object.getPos()
            img,imgSize = itemImg
            if isinstance(object,Item):
                if object._inInventory == True:
                    if object._inInventoryType == "fast":
                        imgSize = imgSize/3
                        posOut= (posImg[pos][0]-0.1,posImg[pos][1]+0.35)
                        xx,yy=trans(posOut) # figure coordinates
                        xa,ya=trans2((xx,yy)) # axes coordinates
                    elif object._inInventoryType == "grip":
                        imgSize = imgSize/3
                        posOut= (posImg[pos][0]-0.3,posImg[pos][1]+0.175)
                        xx,yy=trans(posOut) # figure coordinates
                        xa,ya=trans2((xx,yy)) # axes coordinates
                    elif object._inInventoryType == "load":
                        imgSize = imgSize/3
                        posOut= (posImg[pos][0]-0.30,posImg[pos][1]+0.05)
                        xx,yy=trans(posOut) # figure coordinates
                        xa,ya=trans2((xx,yy)) # axes coordinates
                else:
                    if object.getName() == "Leaves":
                        posOut= (posImg[pos][0],posImg[pos][1]-0.15)
                        xx,yy=trans(posOut) # figure coordinates
                        xa,ya=trans2((xx,yy)) # axes coordinates
                    else:
                        xx,yy=trans(posImg[pos]) # figure coordinates
                        xa,ya=trans2((xx,yy)) # axes coordinates
            else:
                xx,yy=trans(posImg[pos]) # figure coordinates
                xa,ya=trans2((xx,yy)) # axes coordinates
            p2=imgSize/2.0

            imgAx = plt.axes([xa-p2,ya-p2, imgSize, imgSize])
            imgAx.set_aspect('equal')
            imgAx.imshow(img)
            imgAx.axis('off')
                
        plt.gcf()
        self._ax.axis('on')
        xlabels = []
        ylabels = []
        for i in range(0,self._size_x):
            xlabels.append(str(i))

        for i in range(0,self._size_y):
            ylabels.append(str(i))
        
        self._ax.set_xticklabels(xlabels)
        self._ax.set_yticklabels(ylabels)
        self._ax.tick_params(axis=u'both',which=u'both', length =0)

        
        plt.pause(self._sleepTime)
        self._step += 1

    def addItemToItemlist(self, item):
        self._itemList.append(item)

    def addMultipleItemsToItemList(self, listOfItems):
        for item in listOfItems:
            self.addItemToItemlist(item)

    def getItemList(self):
        return self._itemList

    def getStep(self):
        return self._step

    def getSizeX(self):
        return self._size_x

    def getSizeY(self):
        return self._size_y

    def getWaitzoneList(self):
        return self._waitzone_list

    def appendToWaitzoneList(self, input):
        self._waitzone_list.append(input)

    def setupMap(self):
        self._fig.set_figheight(13.61)
        self._fig.set_figwidth(12.8)
        xmap = self.getSizeX()
        ymap = self.getSizeY()
        for x in range(0, xmap):
            for y in range(0, ymap):
                self._my_map.add_node((x, y))
                self._my_map.nodes[(x, y)]["Label"] = self._default_label
                if (x == xmap - 1 and y == ymap-1 or x == 0 and y == 0 or x == 0 and y == ymap-1 or x == xmap - 1 and y == 0) and self._wantWaitzones is True:
                    self._my_map.nodes[(x, y)]["Label"] = "waitzone"

                if x > 0:
                    self._my_map.add_edge((x - 1, y), (x, y))
                if y > 0:
                    self._my_map.add_edge((x, y - 1), (x, y))


        if self._wantWaitzones is True:
            for node in self._my_map.nodes:
                if self._my_map.nodes[node]["Label"] == "waitzone":
                    self.appendToWaitzoneList((node, "Free"))
                    
        #scale sizes of images -> size * 1/(largest size)
        if xmap>ymap:
            largestSize = xmap
        else:
            largestSize = ymap
        
        for object in robotImg:
            img, size = robotImg[object] 
            robotImg[object] = (img,size*(1/largestSize))
        
        for object in itemsImg:
            img, size = itemsImg[object] 
            itemsImg[object] = (img,size*(1/largestSize))