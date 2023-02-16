from z3 import Or, And, If

CANVAS_WIDTH = 0
CANVAS_HEIGHT = 0


def add_canvas_constraints(solver, element):
    # solver.assert_and_track(element.x.var == 0, 'fix canvas x')
    # solver.assert_and_track(element.y.var == 0, 'fix canvas y')
    solver.assert_and_track(element.horizon_arrangement.var >= 0, 'low bound horizon arrangement canvas')
    solver.assert_and_track(element.horizon_arrangement.var < len(element.horizon_arrangement.domain),
                            'high bound horizon arrangement canvas')
    solver.assert_and_track(element.vertical_arrangement.var >= 0, 'low bound vertical arrangement canvas')
    solver.assert_and_track(element.vertical_arrangement.var < len(element.vertical_arrangement.domain),
                            'high bound verticval arrangement canvas')

    margin_list = []
    for margin in element.margin.domain:
        margin_list.append(element.margin.var == margin)
    solver.assert_and_track(Or(margin_list), "margin values canvas")

    main_group = element.children[0]
    solver.assert_and_track(main_group.x.var >= (0 + element.margin.var), "left canvas " + main_group.id)
    solver.assert_and_track(main_group.y.var >= (0 + element.margin.var), "top canvas " + main_group.id)
    solver.assert_and_track((main_group.x.var + main_group.width) <= (0 + CANVAS_WIDTH - element.margin.var),
                            "right canvas " + main_group.id)
    solver.assert_and_track((main_group.y.var + main_group.height) <= (0 + CANVAS_HEIGHT - element.margin.var),
                            "bottom canvas " + main_group.id)

    top_index = element.vertical_arrangement.domain.index("top")
    center_index = element.vertical_arrangement.domain.index("center")
    is_top = element.vertical_arrangement.var == top_index
    is_center = element.vertical_arrangement.var == center_index
    top_arrange = main_group.y.var == (0 + element.margin.var)
    bottom_arrange = (main_group.y.var + main_group.height) == (0 + CANVAS_HEIGHT - element.margin.var)
    center_arrange = (main_group.y.var + (main_group.height / 2)) == (0 + (CANVAS_HEIGHT / 2))
    solver.assert_and_track(If(is_top, top_arrange, If(is_center, center_arrange, bottom_arrange)),
               'vertical arrange canvas')

    left_index = element.horizon_arrangement.domain.index("left")
    center_index = element.horizon_arrangement.domain.index("center")
    is_left = element.horizon_arrangement.var == left_index
    is_center = element.horizon_arrangement.var == center_index
    left_arrange = main_group.x.var == (0 + element.margin.var)
    right_arrange = (main_group.x.var + main_group.width) == (0 + CANVAS_WIDTH - element.margin.var)
    center_arrange = (main_group.x.var + (main_group.width / 2)) == (0 + (CANVAS_WIDTH / 2))
    solver.assert_and_track(If(is_left, left_arrange, If(is_center, center_arrange, right_arrange)),
               'horizon arrange canvas')

def add_leaf_constraints(solver, element):
    solver.assert_and_track(element.x.var >= 0, "low bound of " + element.id + " h")
    solver.assert_and_track(element.y.var >= 0, "low bound of " + element.id + " v")
    solver.assert_and_track((element.x.var + element.width.var) <= CANVAS_WIDTH, "high bound of " + element.id + " h")
    solver.assert_and_track((element.y.var + element.height.var) <= CANVAS_HEIGHT, "high bound of " + element.id + " v")
    # alter
    # if (hasattr(element, 'alternate')):
    #     alternate = element.variables.alternate.z3
    #     or_values = []
    #     for alt_value in element.variables.alternate.domain:
    #         or_values.append(alternate == alt_value)
    #     solver.assert_and_track(z3.Or(or_values), "leaf " + element.id + " alternate in domain")
    if (element.importance == "normal"):
        solver.assert_and_track(element.width.var == element.origin_width, "equal of " + element.id + " width")
        solver.assert_and_track(element.height.var == element.origin_height, "equal of " + element.id + " height")
    elif (element.importance == "low"):
        solver.assert_and_track(element.width.var <= element.origin_width, "width lt constr for " + element.id)
        solver.assert_and_track(element.height.var <= element.origin_height, "height lt constr for " + element.id)
        solver.assert_and_track(element.width.var / element.height.var == element.origin_width / element.origin_height,
                                "aspect ratio constr for " + element.id)
        solver.assert_and_track(element.width.var >= 20, "width gt constr for " + element.id)
        solver.assert_and_track(element.height.var >= 20, "height gt constr for " + element.id)

    elif (element.importance == "high"):
        solver.assert_and_track(element.width.var >= element.origin_width, "width gt constr for " + element.id)
        solver.assert_and_track(element.height.var >= element.origin_height, "height gt constr for " + element.id)
        solver.assert_and_track(element.width.var / element.height.var == element.origin_width / element.origin_height,
                                "aspect ratio constr for " + element.id)
        # solver.assert_and_track(element.width.var <= CANVAS_WIDTH, "width constr lt for " + element.id)
        # solver.assert_and_track(element.height.var <= CANVAS_HEIGHT, "height constr lt for " + element.id)


def max_width_constraint(child_i, widest_i, children):
    if child_i < len(children):
        widest_child = children[widest_i]
        next_child = children[child_i]
        if (widest_child.type == "leaf"):
            widest_child_width = widest_child.width.var
        else:
            widest_child_width = widest_child.width
        if (next_child.type == "leaf"):
            next_child_width = next_child.width.var
        else:
            next_child_width = next_child.width
        return If(widest_child_width > next_child_width,
                  max_width_constraint(child_i + 1, widest_i, children),
                  max_width_constraint(child_i + 1, child_i, children))
    else:
        if children[widest_i].type == "leaf":
            return children[widest_i].width.var
        else:
            return children[widest_i].width


def max_height_constraint(child_i, tallest_i, children):
    if child_i < len(children):
        tallest_child = children[tallest_i]
        next_child = children[child_i]
        if tallest_child.type == "leaf":
            tallest_child_height = tallest_child.height.var
        else:
            tallest_child_height = tallest_child.height
        if (next_child.type == "leaf"):
            next_child_height = next_child.height.var
        else:
            next_child_height = next_child.height
        return If(tallest_child_height > next_child_height,
                  max_height_constraint(child_i + 1, tallest_i, children),
                  max_height_constraint(child_i + 1, child_i, children))
    else:
        if children[tallest_i].type == "leaf":
            return children[tallest_i].height.var
        else:
            return children[tallest_i].height

def add_group_constraints(solver, element):
    solver.assert_and_track(element.x.var >= 0, "low bound of " + element.id + " h")
    solver.assert_and_track(element.y.var >= 0, "low bound of " + element.id + " v")
    solver.assert_and_track((element.x.var + element.width) <= CANVAS_WIDTH, "high bound of " + element.id + " h")
    solver.assert_and_track((element.y.var + element.height) <= CANVAS_HEIGHT, "high bound of " + element.id + " v")

    solver.assert_and_track(element.arrangement.var >= 0, "group arrangement low bound for " + element.id)
    solver.assert_and_track(element.arrangement.var < len(element.arrangement.domain),
                            "group arrangement high bound for " + element.id)
    solver.assert_and_track(element.alignment.var >= 0, "group alignment low bound for " + element.id)
    solver.assert_and_track(element.alignment.var < len(element.alignment.domain),
                            "group alignment high bound for " + element.id)

    padding_list = []
    for padding in element.padding.domain:
        padding_list.append(element.padding.var == padding)
    solver.assert_and_track(Or(padding_list), "group padding for domain for " + element.id)

    children = element.children
    for i in range(0, len(children)):
        element1 = children[i]
        child_x = element1.x.var
        child_y = element1.y.var
        if element1.type == "leaf":
            child_width = element1.width.var
            child_height = element1.height.var
        else:
            child_width = element1.width
            child_height = element1.height
        solver.assert_and_track(child_x >= element.x.var,
                                "bounding box left for " + element1.id)
        solver.assert_and_track(child_y >= element.y.var,
                                "bounding box top for " + element1.id)
        solver.assert_and_track((child_x + child_width) <= (element.x.var + element.width),
                                "bounding box right for " + element1.id)
        solver.assert_and_track((child_y + child_height) <= (element.y.var + element.height),
                                "bounding box bottom for " + element1.id)

    is_vertical = element.arrangement.var == element.arrangement.domain.index("vertical")
    if element.order == "important":
        order_const_list_horizon = []
        order_const_list_vertical = []
        children = element.children
        for i in range(0, len(children) - 1):
            child1 = children[i]
            child1_x = child1.x.var
            child1_y = child1.y.var
            if child1.type == "leaf":
                child1_width = child1.width.var
                child1_height = child1.height.var
            else:
                child1_width = child1.width
                child1_height = child1.height
            child2 = children[i + 1]
            child2_x = child2.x.var
            child2_y = child2.y.var
            order_const_vertical = (child1_y + child1_height + element.padding.var) == child2_y
            order_const_horizon = (child1_x + child1_width + element.padding.var) == child2_x
            order_const_list_horizon.append(order_const_horizon)
            order_const_list_vertical.append(order_const_vertical)
        solver.assert_and_track(If(is_vertical, And(order_const_list_vertical), And(order_const_list_horizon)),
                                "group arrangement for " + element.id)

    total_width = 0
    total_height = 0
    children = element.children
    for i in range(0, len(children)):
        element1 = children[i]
        if (element1.type == "leaf"):
            child_width = element1.width.var
            child_height = element1.height.var
        else:
            child_width = element1.width
            child_height = element1.height
        total_width += child_width
        total_height += child_height
        if i != 0:
            total_width += element.padding.var
            total_height += element.padding.var
    solver.assert_and_track(If(is_vertical, element.height == total_height, element.width == total_height),
                            "bounding box total for " + element.id)

    max_width_const = element.width == (max_width_constraint(1, 0, children))
    max_height_const = element.height == (max_height_constraint(1, 0, children))
    solver.assert_and_track(If(is_vertical, max_width_const, max_height_const),
                            "bounding box max for " + element.id)


    is_pre = element.alignment.var == 0
    is_center = element.alignment.var == 1
    is_vertical = element.arrangement.var == element.arrangement.domain.index("vertical")
    children = element.children
    for element1 in children:
        child_x = element1.x.var
        child_y = element1.y.var
        if element1.type == "leaf":
            child_width = element1.width.var
            child_height = element1.height.var
        else:
            child_width = element1.width
            child_height = element1.height
        left_aligned = (child_x == element.x.var)
        right_aligned = (child_x + child_width) == (element.x.var + element.width)
        h_center_aligned = (child_x + (child_width / 2)) == (element.x.var + (element.width / 2))
        top_aligned = child_y == element.y.var
        bottom_aligned = (child_y + child_height) == (element.y.var + element.height)
        v_center_aligned = (child_y + (child_height / 2)) == (element.y.var + (element.height / 2))
        horizontal = If(is_pre, top_aligned, If(is_center, v_center_aligned, bottom_aligned))
        vertical = If(is_pre, left_aligned, If(is_center, h_center_aligned, right_aligned))
        solver.assert_and_track(If(is_vertical, vertical, horizontal),
                        "group alignment for " + element1.id)

    children = element.children
    for i in range(0, len(children)):
        for j in range(0, len(children)):
            if i != j:
                element1 = children[i]
                element2 = children[j]
                element1_x = element1.x.var
                element1_y = element1.y.var
                if element1.type == "leaf":
                    element1_width = element1.width.var
                    element1_height = element1.height.var
                else:
                    element1_width = element1.width
                    element1_height = element1.height
                element2_x = element2.x.var
                element2_y = element2.y.var
                if element2.type == "leaf":
                    element2_width = element2.width.var
                    element2_height = element2.height.var
                else:
                    element2_width = element2.width
                    element2_height = element2.height
                left = element1_x + element1_width + element.padding.var <= element2_x
                right = element2_x + element2_width + element.padding.var <= element1_x
                top = element1_y + element1_height + element.padding.var <= element2_y
                bottom = element2_y + element2_height + element.padding.var <= element1_y
                solver.assert_and_track(Or(left, right, top, bottom),
                                "non-overlapping in group " + element1.id + " and " + element2.id)

    if element.type == "repeat group":
        if len(children):
            for i in range(0, len(children) - 1):
                element1 = children[i]
                element1_arrangement = element1.arrangement.var
                element1_alignment = element1.alignment.var
                element1_padding = element1.padding.var

                element2 = children[i + 1]
                element2_arrangement = element2.arrangement.var
                element2_alignment = element2.alignment.var
                element2_padding = element2.padding.var
                solver.assert_and_track(element1_arrangement == element2_arrangement,
                                        "arrangement repeat for " + element1.id + " and " + element2.id)
                solver.assert_and_track(element1_alignment == element2_alignment,
                                        "alignment repeat for " + element1.id + " and " + element2.id)
                solver.assert_and_track(element1_padding == element2_padding,
                                        "padding repeat for " + element1.id + " and " + element2.id)

                for j in range(0, len(element1.children)):
                    solver.assert_and_track(
                        element1.children[j].width.var == element2.children[j].width.var, "width repeat for " + element1.children[j].id + " and " + element2.children[j].id)
                    solver.assert_and_track(
                        element1.children[j].height.var == element2.children[
                            j].height.var, "height repeat for " + element1.children[j].id + " and " + element2.children[j].id)