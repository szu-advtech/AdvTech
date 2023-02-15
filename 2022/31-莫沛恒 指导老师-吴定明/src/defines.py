class VM:
    def __init__(self, vm_id, cpu, mem, price):
        self.id = vm_id
        self.cpu = cpu
        self.mem = mem
        self.cpu_now = cpu
        self.mem_now = mem
        self.price = price
        self.stop_use_clock = 0
        self.used_time = 0


class JOB:
    def __init__(self, arrival_time, j_id, j_type, cpu, mem, ex, duration):
        self.arrival_time = arrival_time
        self.start_time = None
        self.finish_time = None
        self.id = j_id
        self.type = j_type
        self.cpu = cpu
        self.mem = mem
        self.ex = ex
        self.ex_placed = 0
        self.duration = duration
        self.ex_placement_list = []
        self.running = False
        self.finished = False

    def __lt__(self, other):
        return self.finish_time < other.finish_time


