#!/usr/bin/env python
#-*- coding:utf-8   -*-

class Robot:
    '''
        Robot 类作为命令的接口主要用于控制机器人，并保存部分机器人自身信息。
    '''

    def __init__(self, name, control_center, type):
        # 机器人名字，用于后续的ros服务或话题
        self.robotname = name
        # 任务队列，用于优先级调度
        # 任务队列，暂时弃用，选择使用behavior_tree的
        self.task_queue = Queue.PriorityQueue()
        # 当前任务
        self.cur_task = None
        # 控制中心，用于环境信息回调
        self.control_center = control_center
        self.robot_status = {"position": {"x": 0, "y": 0, "z": 0, "r": 0}}

        self.type = type

        self.behavior_tree = PriorityTask(name=name)
        self.cur_tree_json = tree_to_json(self.behavior_tree)
        self.main_thread = threading.Thread(target=self.main_)
        self.main_thread.setDaemon(True)
        self.main_thread.start()
        self.pose = {"x": 0, "y": 0, "z": 0, "yaw": 0, "angle": 0}
        self.task = None
        self.lock = False
        self.tran = False
        self.led_actors = []
        self.tool_list = []

        # 机器人感知到的环境信息
        envir_topic_name = '/' + self.robotname + "/envir_info"
        self.envir_subscriber = rospy.Subscriber(envir_topic_name, String, self.env_info_cb, queue_size=1)
        # 机器人位置信息
        self.pose_subscriber = rospy.Subscriber("/" + self.robotname + "/pose", Pose, self.pose_callback, queue_size=1)

    def get_status(self):
        return self.behavior_tree.status

    def main_(self):
        rospy.loginfo("机器人启动")

        while True:
            # 应该不会出现
            if self.behavior_tree.get_status() is None:
                self.behavior_tree.run()

            tree_now = tree_to_json(self.behavior_tree)
            # 行为树状态修改时再更新
            if not self.cur_tree_json == tree_now:
                print("{} 机器人行为树状态：".format(self.robotname))
                print(tree_now)
                self.cur_tree_json = tree_now

                if self.behavior_tree.get_status() == TreeNode.STATUS_RUNNING:
                    # 当有节点的状态改变，需要再次执行run，后面的节点状态才会从None转向running
                    self.behavior_tree.run()

                if self.behavior_tree.get_status() == TreeNode.STATUS_SUCCESS:
                    print("所有任务成功，清空 {} 行为树".format(self.robotname))
                    self.task = None
                    self.behavior_tree.clear()

                if self.behavior_tree.get_status() == TreeNode.STATUS_FAILURE:
                    rospy.logerr("{} 行为树执行失败".format(self.robotname))

                if self.behavior_tree.get_status() == TreeNode.STATUS_PAUSE:
                    rospy.logwarn("{} 行为树暂停执行".format(self.robotname))

            time.sleep(1)

    def get_cur_task(self):
        self.find_from_tree(self.behavior_tree)
        if (isinstance(self.cur_task, ServiceTask)):
            return self.cur_task
        else:
            return None

    def find_from_tree(self, node):
        if (node.get_status() == None):
            return
        if (node.get_status() != None):
            self.cur_task = node
            for child in node.children:
                self.find_from_tree(child)


    def add_task(self, task):
        print("分配任务给：" + str(self.robotname))
        # 如果任务是非指定的，就让它为当前机器人的名字，用于任务重新分配。
        if not task.isSpecify:
            task.robotname = self.robotname
        self.behavior_tree.add_task(task)

        # 将行为树发送到前端
        tree_json = tree_to_json(self.behavior_tree)
        send_tree2_web(tree_json)

    def remove_task(self, task):
        self.behavior_tree.remove_task(task)

    def env_info_cb(self, data):
        # ros会为callback自动创建线程。
        pkg = dict()
        pkg['name'] = self.robotname
        pkg['data'] = data.data
        self.control_center.perception.deal_msg(pkg, self.robotname)

    def pose_callback(self, data):
        update_pose(self.pose, data)

        rospy.set_param("robot/pose/" + self.robotname, self.pose)

class ServiceTask(TreeNode):
    """
        调用ros service
    """

    def __init__(
        self,
        name,
        service,
        service_type,
        request,
        prior=10,
        action_type=None,
        result_cb=None,
        wait_for_service=True,
        timeout=5,
    ):
        super(ServiceTask, self).__init__(name, prior=prior)
        self.action_type = action_type
        self.result = None
        self.request = request
        self.timeout = timeout
        self.result_cb = result_cb
        self.control_center = store["control_center"]

        if wait_for_service:
            rospy.wait_for_service(service, timeout=self.timeout)

        # Create a service proxy
        self.service_proxy = rospy.ServiceProxy(service, service_type)
        self.service_proxy.wait_for_service()

    def run_(self):
        temp_ = json.loads(self.request.data)
        # 在action中的请求参数中添加robotname
        temp_['robotname'] = self.robotname
        self.request.data = json.dumps(temp_)
        result = self.service_proxy(self.request)
        self.callback(result)

    def callback(self, result):
        print("{}  动作执行结果: {}".format(self.name, result.result))
        if result.result == 'success':
            temp = json.loads(self.request.data)
            
            if (self.action_type == "COMMAND_PICK"):
                Perception(self.control_center).remove_item(temp["item"], self.get_robotname())
                delete_model(temp["item"]["detail"])
            elif (self.action_type == "COMMAND_PUT"):
                Perception(self.control_center).add_item(temp["item"], self.get_robotname())
            elif (self.action_type == "COMMAND_CLEAR"):
                delete_model(temp["item"]["detail"])
            self.set_status(TreeNode.STATUS_SUCCESS)
        elif result.result == 'failure':
            self.set_status(TreeNode.STATUS_FAILURE)
        elif result.result == 'loop':
            self.set_status(TreeNode.STATUS_LOOP)
            print("loop执行中···")
        elif result.result == 'preempted':
            self.set_status(None)

    def run(self):
        try:
            if self.get_status() == -1 or self.get_status() == None:
                self.set_status(TreeNode.STATUS_RUNNING)

                # 异步执行，不用等待结果。
                self._thread = threading.Thread(target=self.run_)
                self._thread.start()

                if self.result_cb is not None:
                    pass
                return self.status
            # 如果动作的状态为成功或者失败，直接返回
            elif self.get_status() == TreeNode.STATUS_SUCCESS or self.get_status() == TreeNode.STATUS_FAILURE:
                return self.status
            elif self.get_status() == TreeNode.STATUS_LOOP:
                print("进入loop执行")
                self.run_()
            else:
                return self.status
        except:
            rospy.logerr(sys.exc_info())
            return TreeNode.STATUS_FAILURE

    def pause(self):
        if (self.status == 0):
            print("{}  动作节点接收到pause命令".format(self.name))
            # 适合导航动作，如果是机械臂动作，还需要重置等动作
            self.status = None
            # 直接执行，不用加入到行为树中
            print("{}  执行pause".format(self.robotname))
            cancel_task = create_cancel_navigation_task(self.robotname)
            cancel_task.robotname = self.robotname
            cancel_task.run()

    def reset(self):
        self.status = None

class TreeNode(object):
    '''
        行为树的基类，所有的其他行为树结点都基础于这个类
    '''

    # 行为树的状态
    STATUS_RUNNING = 0
    STATUS_SUCCESS = 1
    STATUS_FAILURE = 2
    STATUS_PAUSE = 3
    STATUS_LOOP = 4

    def __init__(self, name, prior=0):
        self.name = name
        self.isSpecify = False
        self.robotname = None
        # 用于记录之前机器人执行的情况，主要是任务失败的时候使用。
        self.history_log = []
        self.prior = prior
        self.status = None
        self.children = []

        # self.control_center = control_center

    def reset(self):
        self.status = None
        for child in self.children:
            child.reset()

    def clear(self):
        self.status = None
        self.children = []

    def set_robotname(self, robotname):
        self.isSpecify = True
        self.robotname = robotname
        for child in self.children:
            child.set_robotname(robotname)

    def get_robotname(self):
        return self.robotname

    def set_status(self, s):
        self.status = s

    def get_status(self):
        return self.status

    def set_prior(self, s):
        self.prior = s

    def get_prior(self):
        return self.prior

    def add_task(self, t):
        # 将子任务插入到最后
        self.children.append(t)

    def remove_task(self, t):
        self.children.remove(t)

    def insert_task(self, index, task):
        self.children.insert(index, task)

    def run(self):
        pass

    def pause(self):
        pass

    def go_on(self):
        pass

class SelectorTask(TreeNode):
    '''
        选择器结点，在子任务中从左到右执行，如果当前任务失败了就执行下一个
        只有全部任务都失败了才会返回失败。
    '''

    def __init__(self, name, prior=0):
        super(SelectorTask, self).__init__(name, prior)

    def run(self):
        for t in self.children:
            t.robotname = self.robotname
            t.status = t.run()
            if t.status != TreeNode.STATUS_FAILURE:
                return t.status

        return TreeNode.STATUS_FAILURE

    def pause(self):
        if (self.status == 0):
            self.status = 3
            for sub_task in self.children:
                sub_task.pause()

    def go_on(self):
        if (self.status == 3):
            self.status = 0
            for t in self.children:
                t.go_on()

class SequenceTask(TreeNode):
    '''
        序列器结点，在子任务中从左向右执行，如果当前任务失败了就返回错误。
        只有全部任务都执行成功才返回成功。
    '''

    def __init__(self, name, prior=0):
        super(SequenceTask, self).__init__(name, prior)

    def run(self):
        for t in self.children:
            t.robotname = self.robotname
            t.status = t.run()
            if t.status != TreeNode.STATUS_SUCCESS:
                self.set_status(t.status)
                return t.status
        self.set_status(TreeNode.STATUS_SUCCESS)
        return TreeNode.STATUS_SUCCESS

    def pause(self):

        if (self.status == 0):
            self.status = 3
            for sub_task in self.children:
                sub_task.pause()

    def go_on(self):
        if (self.status == 3):
            self.status = 0
            for t in self.children:
                t.go_on()

class ParallelTask(TreeNode):
    '''
        并行器结点，所有子节点同时运行。指定数量的子任务返回成功则返回成功。
        并行策略：
        1、成功子任务数量大于等于指定数量则返回成功
        2、失败子任务数量大于总任务数与指定数量的差则返回失败
        3、其他返回运行中
    '''

    def __init__(self, name, task_num=0, prior=0):
        super(ParallelTask, self).__init__(name, prior)
        self.task_num = task_num
        # 默认所有子任务返回成功则返回成功
        if (task_num == 0):
            task_num = len(self.children)

    def subtask_run(self, childen):
        childen.run()

    def run(self):
        success_num = 0
        fail_num = 0
        for t in self.children:
            t.robotname = self.robotname
            task_thread = threading.Thread(target=self.subtask_run, kwargs={"childen": t})
            task_thread.start()

        for t in self.children:
            if (t.status == TreeNode.STATUS_SUCCESS):
                success_num += 1
            elif (t.status == TreeNode.STATUS_FAILURE):
                fail_num += 1

        if (success_num >= self.task_num):
            self.set_status(TreeNode.STATUS_SUCCESS)
            return TreeNode.STATUS_SUCCESS
        elif (fail_num > len(self.children) - self.task_num):
            self.set_status(TreeNode.STATUS_FAILURE)
            return TreeNode.STATUS_FAILURE
        else:
            self.set_status(TreeNode.STATUS_RUNNING)
            return TreeNode.STATUS_RUNNING

    def pause(self):
        if (self.status == 0):
            self.status = 3
            for sub_task in self.children:
                sub_task.pause()

    def go_on(self):
        if (self.status == 3):
            self.status = 0
            for t in self.children:
                t.go_on()

class LoopTask(TreeNode):
    '''
        重复结点，按照重复策略执行子任务。注意使用场景，目前对重复节点的设计理念是当重复执行时，子任务将全部重新执行，
        但有时候，子任务失败，并不需要整个任务全部重新执行，例如多个导航任务，但对于抓取子任务，或者叫识别并抓取任务，
        有时候就需要识别和抓取都重新执行，虽然这个重复过程可以在callback里面实现，或者直接是机器人端就会实现，但其他团队
        实现机器人功能时，可能不会做这个，为了更好的兼容（兜底），同时其他地方也可能有重复的需要，所以专门实现一个重复节点。
        重复策略：
        1、子任务全部成功，则本节点状态为成功，如果子任务失败，则尝试重新执行(最高n次)，注意子任务节点将全部重新执行。
        2、子任务重复执行n次。目前看意义不大，暂时不实现。
    '''

    def __init__(self, name, count=3, prior=0):
        super(LoopTask, self).__init__(name, prior)
        self.count = count

    def run(self):
        for i, t in enumerate(self.children):
            t.robotname = self.robotname
            t.status = t.run()

            if (t.status == TreeNode.STATUS_FAILURE and self.count > 0):
                detail = "机器人" + self.robotname + "任务" + t.name + "执行失败，重复次数还剩" + str(self.count) + "次！"
                data = {"type": "msg", "detail": detail}
                store["m2h_interaction"].publish(str(data))
                t.status = None
                self.count = self.count - 1
                self.set_status(TreeNode.STATUS_RUNNING)
                return TreeNode.STATUS_RUNNING
            elif (t.status == TreeNode.STATUS_FAILURE and self.count <= 0):
                self.set_status(t.status)
                return t.status
            elif (t.status != TreeNode.STATUS_SUCCESS):
                self.set_status(t.status)
                return t.status

        self.set_status(TreeNode.STATUS_SUCCESS)
        return TreeNode.STATUS_SUCCESS

    def pause(self):
        if (self.status == 0):
            self.status = 3
            for sub_task in self.children:
                sub_task.pause()

    def go_on(self):
        if (self.status == 3):
            self.status = 0
            for t in self.children:
                t.go_on()

class PriorityTask(TreeNode):
    '''
        优先级结点，子任务根据优先级进行排序
    '''

    def __init__(self, name, prior=0):
        super(PriorityTask, self).__init__(name, prior)

    def add_task(self, t):
        t.robotname = self.name
        self.insert_task(self._find_insert_position(t.prior), t)

    def _find_insert_position(self, prior):
        lo = 0
        hi = len(self.children) - 1
        mid = 0
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if self.children[mid].prior == prior:
                return mid
            elif self.children[mid].prior < prior:
                hi = mid - 1
            else:
                lo = mid + 1
        return lo

    def run(self):
        self.set_status(TreeNode.STATUS_RUNNING)
        for t in self.children:
            t.status = t.run()
            # 如果任务在running或者失败，设置状态并返回
            if t.status != TreeNode.STATUS_SUCCESS:
                self.set_status(t.status)
                return t.status
        # 如果跳过了上面的return，且状态为success，children不为空，则是整颗行为树成功
        if len(self.children) != 0:
            self.set_status(TreeNode.STATUS_SUCCESS)
        return TreeNode.STATUS_SUCCESS

    def pause(self):
        if (self.status == TreeNode.STATUS_SUCCESS or self.status == TreeNode.STATUS_FAILURE or self.status == None):
            print("{}  优先级节点状态为 {} ，无法暂停".format(self.name, self.status))
            return

        if (self.status == 0):
            self.status = 3
            for sub_task in self.children:
                sub_task.pause()

    def go_on(self):

        if (self.status == 3):
            rospy.logwarn("{} 行为树恢复执行".format(self.name))
            self.status = 0
            for t in self.children:
                t.go_on()

last_dot_tree = ''
# 可视化打印行为树
class Print_Tree:

    def __init__(self):
        self.tree_plot = ''

    def _tree(self, node, prefix=[]):
        space = '     '
        branch = '│   '
        tee = '├─ '
        last = '└─ '

        if len(prefix) == 0:
            self.tree_plot = ''

        self.tree_plot += json.dumps(
            str(''.join(prefix)) + str(str(node["type"]).ljust(13, " ") + "status: " + str(node["status"]).ljust(5, " ") + " name: " + str(node["name"])),
            ensure_ascii=False) + "\r\n"

        if len(prefix) > 0:
            prefix[-1] = branch if prefix[-1] == tee else space

        for item in node["children"]:
            if (item["children"] != []):
                self._tree(item, prefix + [tee])
            else:
                self._tree(item, prefix + [last])

        return self.tree_plot

# 转为json格式的字符串
def tree_to_json(root):

    def add_node(node):
        ret = dict()
        type = node.__class__.__name__
        ret["type"] = type[:-4]
        if (ret["type"] == "Service"):
            ret["type"] = "Action"
        if (ret["type"] == "Selector"):
            ret["type"] = "Fallback"

        ret['name'] = node.name
        ret['status'] = node.status
        ret['children'] = []
        for c in node.children:
            ret['children'].append(add_node(c))
        return ret

    return Print_Tree()._tree(add_node(root))

def print_tree(root, dotfilepath=None):
    gr = pgv.AGraph(strict=True, directed=True, rotate='0', bgcolor='white', ordering="out")
    gr.node_attr['fontsize'] = '9'
    gr.node_attr['color'] = 'black'

    if dotfilepath is None:
        dotfilepath = os.path.expanduser('~') + '/.ros/tree.dot'

    global last_dot_tree

    # Add the root node
    gr.add_node(root.name)
    node = gr.get_node(root.name)

    if root.status == TreeNode.STATUS_RUNNING:
        node.attr['fillcolor'] = 'yellow'
        node.attr['style'] = 'filled'
        node.attr['border'] = 'bold'
    elif root.status == TreeNode.STATUS_SUCCESS:
        node.attr['color'] = 'green'
    elif root.status == TreeNode.STATUS_FAILURE:
        node.attr['color'] = 'red'
    else:
        node.attr['color'] = 'black'

    def add_edges(root):
        for c in root.children:
            if isinstance(c, (SequenceTask)):
                gr.add_node(c.name.encode('unicode-escape'), shape="cds")
            elif isinstance(c, (SelectorTask)):
                gr.add_node(c.name.encode('unicode-escape'), shape="diamond")
            else:
                gr.add_node(c.name)

            gr.add_edge((root.name, c.name))
            node = gr.get_node(c.name)

            if c.status == TreeNode.STATUS_RUNNING:
                node.attr['fillcolor'] = 'yellow'
                node.attr['style'] = 'filled'
                node.attr['border'] = 'bold'
            elif c.status == TreeNode.STATUS_SUCCESS:
                node.attr['color'] = 'green'
            elif c.status == TreeNode.STATUS_FAILURE:
                node.attr['color'] = 'red'
            else:
                node.attr['color'] = 'black'

            if c.children != []:
                add_edges(c)

    add_edges(root)

    current_dot_tree = gr.string()

    if current_dot_tree != last_dot_tree:
        print("修改行为树")
        gr.write(dotfilepath)
        last_dot_tree = gr.string()