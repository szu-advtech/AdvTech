import z3
import random
from element import Element
import constraints
from copy import copy


class Solver(object):
    def __init__(self, elements):
        self.elements = dict()
        self.generate_elements(elements)
        self.z3_solver = z3.Solver()
        self.result_list = []
        self.unassigned_var_list = []
        self.result_size = 0
        self.decisive_var_list = self.get_decisive_var_list()
        constraints.CANVAS_WIDTH = elements[0]["width"]
        constraints.CANVAS_HEIGHT = elements[0]["height"]
        self.construct_constraints()

    def generate_elements(self, origin_elements):
        siblings = []
        for i in range(0, len(origin_elements)):
            origin_element = origin_elements[i]
            children = None
            if "children" in origin_element:
                children = self.generate_elements(origin_element["children"])
            element = Element(origin_element)
            self.elements[element.id] = element
            if children is not None:
                element.children.extend(children)
            siblings.append(element)
        return siblings

    def get_decisive_var_list(self):
        var_list = []
        for element in self.elements.values():
            if element.type == "canvas":
                var_list.append(element.horizon_arrangement)
                var_list.append(element.vertical_arrangement)
                var_list.append(element.margin)
            elif element.type != "leaf":
                var_list.append(element.arrangement)
                var_list.append(element.alignment)
                var_list.append(element.padding)
                if hasattr(element, 'alternate'):
                    var_list.append(element.alternate)
        return var_list

    def construct_constraints(self):
        for element in self.elements.values():
            if element.type == "canvas":
                constraints.add_canvas_constraints(self.z3_solver, element)
            elif element.type == "leaf":
                constraints.add_leaf_constraints(self.z3_solver, element)
            else:
                constraints.add_group_constraints(self.z3_solver, element)

    def solve(self):
        self.unassigned_var_list = copy(self.decisive_var_list)
        while self.result_size < 20:
            assigned_var_list = []
            res = self.branch_and_bound(assigned_var_list)
            if res is not None:
                self.result_list.append(res)
                self.result_size += 1
        return self.result_list

    def branch_and_bound(self, assigned_var_list):
        if len(self.unassigned_var_list) == 0:
            if str(self.z3_solver.check()) == 'sat':
                res = self.generate_single_result()
                self.unassigned_var_list = copy(self.decisive_var_list)
                for i in range(0, len(self.decisive_var_list)):
                    self.z3_solver.pop()
                for var in self.decisive_var_list:
                    var.value = None
                return res
        else:
            var_index = random.randint(0, len(self.unassigned_var_list) - 1)
            var = self.unassigned_var_list.pop(var_index)
            assigned_var_list.append(var)
            domain = var.domain[0:len(var.domain)]
            random.shuffle(domain)
            for i in range(0, len(domain)):
                var.value = var.domain.index(domain[i])
                if var.name == "proximity" or var.name == "margin":
                    value = var.domain[var.value]
                else:
                    value = var.value
                self.z3_solver.add(var.var == value)
                self.z3_solver.push()
                if str(self.z3_solver.check()) == 'sat':
                    res = self.branch_and_bound(assigned_var_list)
                    if res is not None:
                        return res
                self.z3_solver.pop()
            var.value = None
            self.unassigned_var_list.append(var)

    def generate_single_result(self):
        res = dict()
        z3_model = self.z3_solver.model()
        for element in self.elements.values():
            if element.type == "leaf":
                new_element = dict()
                new_element["id"] = element.id
                new_element["icon"] = element.icon
                x = int(z3_model[element.x.var].as_string())
                y = int(z3_model[element.y.var].as_string())
                width = int(z3_model[element.width.var].as_string())
                height = int(z3_model[element.height.var].as_string())
                new_element["x"] = x
                new_element["y"] = y
                new_element["width"] = width
                new_element["height"] = height
                res[new_element["id"]] = new_element
        return res
