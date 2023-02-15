import numpy as np
import defines as defs
import workload
import copy


vm_types = 3

# vm1_total = 4
# vm1_cpu = 4
# vm1_mem = 16
# vm1_price = 0.24
#
# vm2_total = 4
# vm2_cpu = 8
# vm2_mem = 32
# vm2_price = 0.48
#
# vm3_total = 4
# vm3_cpu = 12
# vm3_mem = 48
# vm3_price = 0.72

vm1_total = 3
vm1_cpu = 4
vm1_mem = 12
vm1_price = 1

vm2_total = 3
vm2_cpu = 8
vm2_mem = 24
vm2_price = 2

vm3_total = 3
vm3_cpu = 12
vm3_mem = 36
vm3_price = 3

j_total = 3
j_types = 3
j_cpu_max = 4
j_cpu_min = 1
j_mem_max = 12
j_mem_min = 1
j_ex_max = 8
j_ex_min = 1

j1_cpu = 2
j1_mem = 4
j1_ex = 3
j1_time = 60

j2_cpu = 4
j2_mem = 8
j2_ex = 2
j2_time = 100

j3_cpu = 6
j3_mem = 8
j3_ex = 2
j3_time = 80

job_features = 4
features = (vm1_total + vm2_total + vm3_total) * 2 + job_features

JOBS = []
VMS = []

cluster_state_init = []
cluster_state_min = []
cluster_state_max = []

max_episode_cost = 0
min_avg_job_duration = 0


def gen_cluster_state(job_idx, jobs, vms):
    cluster_state = []
    i = 0
    while i < len(vms):
        cluster_state.append(vms[i].cpu_now)
        cluster_state.append(vms[i].mem_now)
        i += 1
    # cluster_state.append(jobs[job_idx].id)
    cluster_state.append(jobs[job_idx].type)
    cluster_state.append(jobs[job_idx].cpu)
    cluster_state.append(jobs[job_idx].mem)
    cluster_state.append(jobs[job_idx].ex - jobs[job_idx].ex_placed)
    return cluster_state


def gen_cluster_state_min():
    cluster_state = []
    i = 0
    while i < len(VMS):
        cluster_state.append(0)
        cluster_state.append(0)
        i += 1
    cluster_state.append(1)
    cluster_state.append(j_cpu_min)
    cluster_state.append(j_mem_min)
    cluster_state.append(j_ex_min)
    return cluster_state


def gen_cluster_state_max():
    cluster_state = []
    i = 0
    while i < len(VMS):
        cluster_state.append(VMS[i].cpu)
        cluster_state.append(VMS[i].mem)
        i += 1
    cluster_state.append(j_types)
    cluster_state.append(j_cpu_max)
    cluster_state.append(j_mem_max)
    cluster_state.append(j_ex_max)
    return cluster_state


def fetch_jobs_workload():
    global JOBS
    JOBS = copy.deepcopy(workload.JOBS_WORKLOAD)


def init_jobs():
    fetch_jobs_workload()


def init_vms():
    global VMS
    VMS = []
    for i in range(vm1_total):
        VMS.append(defs.VM(len(VMS), vm1_cpu, vm1_mem, vm1_price))
    for j in range(vm2_total):
        VMS.append(defs.VM(len(VMS), vm2_cpu, vm2_mem, vm2_price))
    for k in range(vm3_total):
        VMS.append(defs.VM(len(VMS), vm3_cpu, vm3_mem, vm3_price))


def init_cluster():
    init_jobs()
    init_vms()
    global cluster_state_init
    global cluster_state_max
    global cluster_state_min
    cluster_state_init = gen_cluster_state(0, JOBS, VMS)
    cluster_state_min = gen_cluster_state_min()
    cluster_state_max = gen_cluster_state_max()

    vms_total_price = 0
    jobs_total_time = 0

    global max_episode_cost
    global min_avg_job_duration
    for i in range(len(VMS)):
        vms_total_price += VMS[i].price
    for i in range(len(JOBS)):
        jobs_total_time += JOBS[i].duration

    max_episode_cost = vms_total_price * jobs_total_time
    min_avg_job_duration = float(jobs_total_time) / len(JOBS)

    # print('Number of VMS {0}'.format(len(VMS)))
    # print('Number of Jobs {0}'.format(len(JOBS)))
    # print('MAX Episode VM Cost {0}'.format(total_episode_cost))
    # print('MIN Episode AVG Job Duration {0}'.format(min_avg_job_duration))
