
class Item(object):
    def __init__(self, name, pos):
        self._name = name
        self._pos = pos
        self._inInventory = False
        self._inInventoryType = None

    def getName(self):
        return self._name

    def getAmount(self):
        return self._amount

    def getPos(self):
        return self._pos

    def grabItem(self, robotType):
        self._inInventory = True
        self._inInventoryType = robotType

    def updatePos(self,newPos):
        self._pos = newPos
        
    def placeItem(self, newPos):
        self._inInventory = False
        self._inInventoryType = None
        self._pos = newPos