from z3 import String,Int

class Variable(object):
    def __init__(self, name, domain = [], variable_type = "int"):
        self.name = name
        self.domain = domain
        self.type = variable_type
        if self.type == "str":
            self.var = String(self.name)
        elif self.type == "int":
            self.var = Int(self.name)
        self.value = None

class Element(object):
    def __init__(self, origin_element):

        self.id = origin_element["id"]
        self.type = origin_element["type"]

        if "order" in origin_element:
            self.order = origin_element["order"]
        if "width" in origin_element:
            self.origin_width = origin_element["width"]
        if "height" in origin_element:
            self.origin_height = origin_element["height"]
        if "importance" in origin_element:
            self.importance = origin_element["importance"]
        if "icon" in origin_element:
            self.icon = origin_element["icon"]

        self.x = Variable(self.id + '_x')
        self.y = Variable(self.id + '_y')

        if self.type == "canvas":
            self.children = []
            self.horizon_arrangement = Variable(self.id + "_horizon_arrangement", ["left", "center", "right"], "int")
            self.vertical_arrangement = Variable(self.id + "_vertical_arrangement", ["top", "center", "bottom"], "int")
            self.margin = Variable(self.id + "_margin", [10, 20, 30, 40, 50])
        elif self.type == "leaf":
            self.width = Variable(self.id + "_width")
            self.height = Variable(self.id + "_height")
        else:
            self.children = []
            self.arrangement = Variable(self.id + "_arrangement", ["horizontal", "vertical"],"int")
            self.alignment = Variable(self.id + "_alignment", ["left", "center", "right"],"int")
            self.padding = Variable(self.id + "_padding", [10, 20, 30, 40, 50],"int")
            self.width = Int(self.id + "_width")
            self.height = Int(self.id + "_height")
            if self.type == "alternative group":
                self.alternative = Variable(self.id + "_alternative")


