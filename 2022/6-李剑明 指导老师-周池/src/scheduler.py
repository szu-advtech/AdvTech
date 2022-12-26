#!/usr/bin/env python3

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from functools import partial
from json import dumps as json_dumps, loads as json_loads
from os import getenv
from re import match as re_match
from requests import get as requests_get, post as requests_post
from requests.models import Response
from sys import exit
from time import perf_counter, sleep
from unittest.mock import Mock

from bidict import bidict
from kubernetes import client as k8s_client, config as k8s_config, watch as k8s_watch

from health_server import HealthServer


class Scheduler():
    def __init__(self):
        '''Load kubernetes config and connect to the API server.'''
        try:
            k8s_config.load_incluster_config()
        except Exception as e:
            exit('Unable to load cluster config: {}'.format(e))
        self.api = k8s_client.CoreV1Api()
        self.name = getenv('BOREAS_SCHEDULER_NAME', 'boreas-scheduler')
        self.namespace = getenv('BOREAS_SCHEDULER_NAMESPACE', 'default')
        self.reserved_kublet_cpu = int(getenv('BOREAS_SCHEDULER_RESERVED_KUBLET_CPU', 100))
        self.reserved_kublet_ram = int(getenv('BOREAS_SCHEDULER_RESERVED_KUBLET_RAM', 50))
        self.default_request_cpu = int(getenv('BOREAS_SCHEDULER_DEFAULT_REQUEST_CPU')) if getenv('BOREAS_SCHEDULER_DEFAULT_REQUEST_CPU') else None
        self.default_request_ram = int(getenv('BOREAS_SCHEDULER_DEFAULT_REQUEST_RAM')) if getenv('BOREAS_SCHEDULER_DEFAULT_REQUEST_RAM') else None
        self.no_solution_found_warning = getenv('BOREAS_SCHEDULER_NO_SOLUTION_FOUND_WARNING', False)
        self.require_scheduler_name_spec = getenv('BOREAS_SCHEDULER_REQUIRE_SCHEDULER_NAME_SPEC', True)
        self.optimizer = self.Optimizer(int(getenv('BOREAS_OPTIMIZER_PORT', 9001)),
                                        getenv('BOREAS_OPTIMIZER_OPTIONS', '--solver, lex-or-tools'))
        self.nicknames = bidict() # we naively assume names don't change during runtime; TODO don't


    class Optimizer():
        def __init__(self, port, options):
            self.port = port
            self.options = [x.strip() for x in options.split(',')] if options else None


        def optimize(self, spec):
            query_url = 'http://127.0.0.1:{}/process'.format(self.port)
            try:
                return requests_post(query_url, data=json_dumps(spec)).json()
            except ValueError:
                raise Exception('No JSON object could be decoded from API response.')


    def set_nickname(self, name):
        '''Zephyrus2 gives dash symbols in names meaning so a workaround like this is needed.'''
        if '-' not in name:
            return name
        else:
            try:
                nickname = name.replace('-', '_')
                self.nicknames[nickname] = name
            except bidict.ValueDuplicationError:
                raise Exception('Both keys and values must be unique in bidict')
            return nickname


    def get_nickname(self, name):
        if '_' not in name or name not in self.nicknames:
            return name
        else:
            return self.nicknames[name]


    def cpu_convertion(self, input):
        '''k8s allows for fractional CPUs, in two units: 100m (millicpu/millicores) or 0.1.'''
        try:
            if input.endswith('m'):
                return int(input[:-1])
            else:
                return int(float(input) * 1000)
        except ValueError:
            raise Exception('Argument not a CPU measurement: \'{}\''.format(input))


    def ram_convertion(self, input):
        '''k8s allows the following RAM suffixes: E, Ei, P, Pi, T, Ti, G, Gi, M, Mi, K or Ki.'''
        try:
            ram, unit = re_match(r'(\d+)(\w+)', input).groups()
        except AttributeError:
            raise Exception('Argument not a RAM measurement: \'{}\''.format(input))
        if unit == 'M': return int(ram) # Megabyte
        if unit == 'Mi': return int(int(ram) * 1.049) # Mebibyte
        if unit == 'K': return int(int(ram) / 1000) # Kilobyte
        if unit == 'Ki': return int(int(ram) / 976.562) # Kibibyte 
        if input.isdigit(): return int(int(input) / 1e+6) # Byte
        if unit == 'G': return int(int(ram) * 1000) # Gigabyte
        if unit == 'Gi': return int(int(ram) * 1073.742) # Gibibyte
        if unit == 'T': return int(int(ram) * 1e+6) # Terabyte
        if unit == 'Ti': return int(int(ram) * 1.1e+6) # Tebibyte
        if unit == 'P': return int(int(ram) * 1e+9) # Petabyte
        if unit == 'Pi': return int(int(ram) * 1.126e+9) # Pebibyte
        if unit == 'E': return int(int(ram) * 1e+12) # Exabyte
        if unit == 'Ei': return int(int(ram) * 1.153e+12) # Exbibyte
        raise Exception('Unrecognized RAM measurement: \'{}\''.format(input))


    def nodes_usage(self):
        '''Gather the resource usage on each worker node.'''
        nodes = defaultdict(lambda: {'cpu': 0, 'memory': 0})
        for pod in self.api.list_pod_for_all_namespaces(watch=False).items:
            if pod.status.phase == 'Running' and pod.metadata.namespace != 'kube-system':
                for container in pod.spec.containers:
                    if container.resources.requests:
                        try:
                            nodes[pod.spec.node_name]['cpu'] += self.cpu_convertion(container.resources.requests['cpu'])
                        except KeyError as e:
                            if self.default_request_cpu:
                                print('Warning: CPU request for {}\'s {} container missing, used default ({}).'.format(pod.metadata.name, container.name, self.default_request_cpu))
                                nodes[pod.spec.node_name]['cpu'] += self.cpu_convertion(self.default_request_cpu)
                            else:
                                raise Exception('Resource request for {}\'s {} container lacks {} key.'.format(pod.metadata.name, container.name, e))
                        try:
                            nodes[pod.spec.node_name]['memory'] += self.ram_convertion(container.resources.requests['memory'])
                        except KeyError as e:
                            if self.default_request_ram:
                                print('Warning: memory request for {}\'s {} container missing, used default ({}).'.format(pod.metadata.name, container.name, self.default_request_ram))
                                nodes[pod.spec.node_name]['memory'] += self.ram_convertion(self.default_request_ram)
                            else:
                                raise Exception('Resource request for {}\'s {} container lacks {} key.'.format(pod.metadata.name, container.name, e))
                    else:
                        if self.default_request_cpu and self.default_request_ram:
                            print('Warning: {}\'s {} container lacks resource requests, used defaults (CPU: {} and memory: {}).'.format(pod.metadata.name, container.name, self.default_request_cpu, self.default_request_ram))
                            nodes[pod.spec.node_name]['cpu'] += self.cpu_convertion(self.default_request_cpu)
                            nodes[pod.spec.node_name]['memory'] += self.ram_convertion(self.default_request_ram)
                        else:
                            raise Exception('Error: {}\'s {} container lacks resource requests.'.format(pod.metadata.name, container.name))
        return nodes


    def nodes_available(self):
        '''Gather specs for available nodes.'''
        nodes_usage = self.nodes_usage()
        ready_nodes = {}
        for n in self.api.list_node().items:
            for status in n.status.conditions:
                if status.status == 'True' and status.type == 'Ready' and not n.spec.unschedulable and 'node-role.kubernetes.io/master' not in n.metadata.labels:
                    cpu = self.cpu_convertion(n.status.allocatable['cpu']) - nodes_usage[n.metadata.name]['cpu'] - self.reserved_kublet_cpu
                    ram = self.ram_convertion(n.status.allocatable['memory']) - nodes_usage[n.metadata.name]['memory'] - self.reserved_kublet_ram
                    ready_nodes[self.set_nickname(n.metadata.name)] = {'num': 1, 'resources': {'RAM': ram, 'cpu': cpu}} #TODO add "cost"
        return ready_nodes

    def nodes_capacity(self):
        '''Gather whole space for all nodes'''
        ready_nodes = {}
        for n in self.api.list_node().items:
            for status in n.status.conditions:
                if status.status == 'True' and status.type == 'Ready' and not n.spec.unschedulable and 'node-role.kubernetes.io/master' not in n.metadata.labels:
                    cpu = self.cpu_convertion(n.status.allocatable['cpu']) - self.reserved_kublet_cpu
                    ram = self.ram_convertion(n.status.allocatable['memory']) - self.reserved_kublet_ram
                    ready_nodes[self.set_nickname(n.metadata.name)] = {'num': 1, 'resources': {'RAM': ram, 'cpu': cpu}} #TODO add "cost"
        return ready_nodes

    def pod_requirements(self, event):
        '''Sums up the resource requirements of containers in a pod.'''
        total_cpu = 0
        total_ram = 0
        for container in event['object'].spec.containers:
            try:
                total_cpu += self.cpu_convertion(container.resources.requests['cpu'])
                total_ram += self.ram_convertion(container.resources.requests['memory'])
            except KeyError as e:
                raise Exception('Resource request for {}\'s {} container lacks {} key.'.format(event['object'].metadata.name, container.name, e))
        return {'resources': {'RAM': total_ram, 'cpu': total_cpu}}


    def get_set_name(self, event):
        '''Gets the name of a pod's replica set by striping the pod name off its template hash.'''
        return event['object'].metadata.owner_references[0].name.replace('-{}'.format(event['object'].metadata.labels['pod-template-hash']), '')


    def get_event_by_name(self, name, replica_sets):
        '''Find an event by name among the replica sets.'''
        for dicts in replica_sets.values():
            for event in dicts:
                if event['object'].metadata.name == name:
                    return event
        return False


    def pod_affinities(self, event, pod_nickname, labels, replica_sets=None):
        '''Gather the affinity/anti-affinity requirements for a pod.'''
        affinities = []
        if event['object'].spec.affinity and event['object'].spec.affinity.pod_anti_affinity:
            for selector in event['object'].spec.affinity.pod_anti_affinity.required_during_scheduling_ignored_during_execution:
                if selector.topology_key == 'kubernetes.io/hostname':
                    for expression in selector.label_selector.match_expressions:
                        if expression.operator == 'In':
                            for value in expression.values:
                                if replica_sets and event['object'].metadata.labels[expression.key] and event['object'].metadata.labels[expression.key] == value:
                                    affinities.append('(forall ?x in locations: (?x.{} <= 1))'.format(pod_nickname))
                                elif replica_sets and labels[value][expression.key] and self.get_event_by_name(labels[value][expression.key][0], replica_sets):
                                    affinities.append('(forall ?x in locations: (?x.{} > 0 impl ?x.{} = 0))'.format(pod_nickname, self.set_nickname(self.get_set_name(self.get_event_by_name(labels[value][expression.key][0], replica_sets)))))
                                else:
                                    for entity in labels[value][expression.key]:
                                        if entity != event['object'].metadata.name:
                                            affinities.append('(forall ?x in locations: (?x.{} > 0 impl ?x.{} = 0))'.format(pod_nickname, self.set_nickname(entity)))
                        else:
                            print('Warning: anti-affinity operator \'{}\' not currently supported.'.format(expression.operator))
                        # TODO add support for 'Exists', 'NotIn' and 'DoesNotExist' operators
                # TODO selector.label_selector.match_labels
                # TODO selector.namespaces
                else:
                    raise Exception('Admission controller is modified or LimitPodHardAntiAffinityTopology has been disabled.')
            if event['object'].spec.affinity.pod_anti_affinity.preferred_during_scheduling_ignored_during_execution:
                print('Warning: preferred pod anti-affinity not currently supported.')
        if event['object'].spec.affinity and event['object'].spec.affinity.pod_affinity:
            for selector in event['object'].spec.affinity.pod_affinity.required_during_scheduling_ignored_during_execution:
                if selector.topology_key == 'kubernetes.io/hostname':
                    for expression in selector.label_selector.match_expressions:
                        if expression.operator == 'In':
                            for value in expression.values:
                                if replica_sets and labels[value][expression.key] and self.get_event_by_name(labels[value][expression.key][0], replica_sets):
                                    affinities.append('(forall ?x in locations: (?x.{} > 0 impl ?x.{} > 0))'.format(pod_nickname, self.set_nickname(self.get_set_name(self.get_event_by_name(labels[value][expression.key][0], replica_sets)))))
                                else:
                                    for entity in labels[value][expression.key]:
                                        if entity != event['object'].metadata.name:
                                            affinities.append('(forall ?x in locations: (?x.{} > 0 impl ?x.{} > 0))'.format(pod_nickname, self.set_nickname(entity)))
                        else:
                            print('Warning: affinity operator \'{}\' not currently supported.'.format(expression.operator))
                        # TODO add support for 'Exists', 'NotIn' and 'DoesNotExist' operators
                # TODO selector.label_selector.match_labels
                # TODO selector.namespaces
                else:
                    print('Warning: affinity topology key \'{}\' not currently supported.'.format(selector.topology_key))
            if event['object'].spec.affinity.pod_affinity.preferred_during_scheduling_ignored_during_execution:
                print('Warning: preferred pod affinity not currently supported.')
        if len(affinities) > 0:
            return ' and '.join(affinities)
        else:
            return None


    def get_labels_and_sets(self, events, runningEvents=False):
        '''Gather the labels of batched and present pods, and worker nodes. Split out replica sets.'''
        regular_events = []
        replica_sets = defaultdict(list)
        labels = defaultdict(lambda: defaultdict(list))
        for event in events:
            for key, value in event['object'].metadata.labels.items():
                labels[value][key] += [event['object'].metadata.name]
            if event['object'].metadata.owner_references and event['object'].metadata.owner_references[0].kind == 'ReplicaSet':
                replica_sets[self.set_nickname(self.get_set_name(event))].append(event)
            else:
                regular_events.append(event)

        if runningEvents:
            for event in runningEvents:
                if event['object'].metadata.owner_references and event['object'].metadata.owner_references[0].kind == 'ReplicaSet':
                    replica_sets[self.set_nickname(self.get_set_name(event))].append(event)
                    for key, value in event['object'].metadata.labels.items():
                        labels[value][key] += [event['object'].metadata.name]

        for pod in self.api.list_pod_for_all_namespaces(watch=False).items:
            if pod.metadata.namespace != 'kube-system':
                for key, value in pod.metadata.labels.items():
                    labels[value][key] += [pod.metadata.name]
        for node in self.api.list_node().items:
            if 'node-role.kubernetes.io/master' not in node.metadata.labels:
                for key, value in node.metadata.labels.items():
                    labels[value][key] += [node.metadata.name]
        return regular_events, replica_sets, labels


    def schedule(self, name, node, namespace):
        '''Create a binding object to schedule the pod.'''
        target = k8s_client.V1ObjectReference(kind = 'Node', api_version = 'v1', name = node)
        meta = k8s_client.V1ObjectMeta(name = name)
        body = k8s_client.V1Binding(target = target, metadata = meta)
        self.api.create_namespaced_binding(namespace=namespace, body=body, _preload_content=False)
        print('Scheduling \'{}\' on {}'.format(name, node))


    def evict(self, pod, node, namespace):
        '''Evict the binded pod and reschedule it'''
        try:
            name = pod.metadata.name
            owner_name = pod.metadata.owner_references[0].name
            print("> Ready to reschedule pod [{}]".format(name))

            record_pod = []
            pod_list = self.api.list_namespaced_pod(self.namespace)
            for elem in pod_list.items:
                if elem.metadata.owner_references and elem.metadata.owner_references[0].name == owner_name:
                    record_pod.append(elem.metadata.name)

            meta = k8s_client.V1ObjectMeta(name=name, namespace=namespace)
            body = k8s_client.V1Eviction(metadata=meta)
            self.api.create_namespaced_pod_eviction(name=name, namespace=namespace, body=body)
            print("> Had evicted pod [{}]".format(name))

            new_pod_name = ""
            new_pod_create_flag = False
            old_pod_delete_flag = False
            while not (new_pod_create_flag and old_pod_delete_flag):
                pod_list = self.api.list_namespaced_pod(self.namespace)
                old_pod_delete_flag = True
                for elem in pod_list.items:
                    if elem.metadata.name == name:
                        old_pod_delete_flag = False
                        print("> Waiting for pod [{}] terminating...".format(name))
                        sleep(1)
                    if elem.metadata.owner_references and elem.metadata.owner_references[0].name == owner_name and not new_pod_create_flag:
                        if elem.metadata.name not in record_pod:
                            new_pod_name = elem.metadata.name
                            print("> New pod has genenrated: [{}]".format(new_pod_name))
                            new_pod_create_flag = True
            return new_pod_name
        except Exception as e:
            print("> Unexcept error:")
            print(e)


    def warn_no_solution_found(self, event, namespace):
        '''Add event message to pod description when optimizer is unable to find solution.'''
        object = k8s_client.V1ObjectReference(api_version = 'v1',
                                          kind = 'Pod',
                                          name = event['object'].metadata.name,
                                          namespace = namespace,
                                          resource_version = event['object'].metadata.resource_version,
                                          uid = event['object'].metadata.uid)
        meta = k8s_client.V1ObjectMeta(name = event['object'].metadata.name)
        source = k8s_client.V1EventSource(component = self.name)
        timestamp = datetime.now(timezone.utc)
        body = k8s_client.V1Event(count = 1,
                              first_timestamp = timestamp,
                              involved_object = object,
                              last_timestamp = timestamp,
                              message = "Optimizer was unable to find solution for batch containing pod.",
                              metadata = meta,
                              reason = 'FailedScheduling',
                              source = source,
                              type = 'Warning')
        self.api.create_namespaced_event(namespace=namespace, body=body, _preload_content=False)


    def wait_for_optimizer(self):
        '''Wait for optimizer's readiness probe to return HTTP 200.'''
        response = Mock(spec=Response)
        response.status_code = 503
        while response.status_code != 200:
            try:
                response = requests_get('http://127.0.0.1:{}/health'.format(self.optimizer.port))
            except:
                print('Waiting for optimizer at port {} to get ready'.format(self.optimizer.port))
                sleep(1)
                pass


    def schedule_events(self, events):
        '''Gather pod and node details, and schedule the pods.'''
        spec = {}
        spec['locations'] = self.nodes_available()
        spec['components'] = {}
        if self.optimizer.options:
            spec['options'] = self.optimizer.options
        spec['specification'] = ''
        regular_events, replica_sets, labels = self.get_labels_and_sets(events)
        for event in regular_events:
            pod_nickname = self.set_nickname(event['object'].metadata.name)
            spec['components'][pod_nickname] = self.pod_requirements(event)
            if spec['specification']: spec['specification'] += ' and '
            spec['specification'] += '{} > 0'.format(pod_nickname)
            affinities = self.pod_affinities(event, pod_nickname, labels)
            if affinities: spec['specification'] += ' and {}'.format(affinities)
        for set_name, set_events in replica_sets.items():
            set_nickname = self.set_nickname(set_name)
            spec['components'][set_nickname] = self.pod_requirements(set_events[0])
            if spec['specification']: spec['specification'] += ' and '
            spec['specification'] += '{} > {}'.format(set_nickname, len(set_events)-1)
            affinities = self.pod_affinities(set_events[0], set_nickname, labels, replica_sets)
            if affinities: spec['specification'] += ' and {}'.format(affinities)
        spec['specification'] += '; cost; (sum ?y in components: ?y)'
        print(json_dumps(spec, sort_keys=True, indent=4))
        result = self.optimizer.optimize(spec)
        print(json_dumps(result, sort_keys=True, indent=4))
        if 'configuration' in result:
            try:
                for node in result['configuration']['locations']:
                    node_name = self.get_nickname(node)
                    for pod in result['configuration']['locations'][node]['0']:
                        if pod in replica_sets:
                            for i in range(result['configuration']['locations'][node]['0'][pod]):
                                pod_name = replica_sets[pod].pop()['object'].metadata.name
                                self.schedule(pod_name, node_name, self.namespace)
                        else:
                            pod_name = self.get_nickname(pod)
                            self.schedule(pod_name, node_name, self.namespace)
            except k8s_client.rest.ApiException as e:
                raise Exception(json_loads(e.body)['message'])
            finally:
                return True
        else:
            print('Warning: no configuration returned from optimizer: {}'.format(result['error']))
            if self.no_solution_found_warning:
                for event in regular_events:
                    self.warn_no_solution_found(event, self.namespace)
            return False


    def reschedule_events(self, events, runningEvents):
        '''Check all pod and try to reschedule it'''
        spec = {}
        spec['locations'] = self.nodes_capacity()
        spec['components'] = {}
        if self.optimizer.options:
            spec['options'] = self.optimizer.options
        spec['specification'] = ''
        regular_events, replica_sets, labels = self.get_labels_and_sets(events, runningEvents)
        for event in regular_events:
            pod_nickname = self.set_nickname(event['object'].metadata.name)
            spec['components'][pod_nickname] = self.pod_requirements(event)
            if spec['specification']: spec['specification'] += ' and '
            spec['specification'] += '{} > 0'.format(pod_nickname)
            affinities = self.pod_affinities(event, pod_nickname, labels)
            if affinities: spec['specification'] += ' and {}'.format(affinities)
        for set_name, set_events in replica_sets.items():
            set_nickname = self.set_nickname(set_name)
            spec['components'][set_nickname] = self.pod_requirements(set_events[0])
            if spec['specification']: spec['specification'] += ' and '
            spec['specification'] += '{} > {}'.format(set_nickname, len(set_events)-1)
            affinities = self.pod_affinities(set_events[0], set_nickname, labels, replica_sets)
            if affinities: spec['specification'] += ' and {}'.format(affinities)
        spec['specification'] += '; cost; (sum ?y in components: ?y)'

        print(json_dumps(spec, sort_keys=True, indent=4))
        result = self.optimizer.optimize(spec)
        print(json_dumps(result, sort_keys=True, indent=4))
        
        if 'configuration' in result:
            try:
                delay_list = []
                for node in result['configuration']['locations']:
                    node_name = self.get_nickname(node)
                    for pod in result['configuration']['locations'][node]['0']:
                        if pod in replica_sets:
                            for i in range(result['configuration']['locations'][node]['0'][pod]):
                                temp_pod = replica_sets[pod].pop()['object']
                                pod_name = temp_pod.metadata.name
                                pod_status = temp_pod.status.phase
                                if pod_status == 'Pending':
                                    # Delay scheduling Pending pod
                                    delay_list.append([pod_name, node_name])
                                elif pod_status == 'Running':
                                    if temp_pod.spec.node_name != node_name:
                                        new_pod_name = self.evict(temp_pod, node_name, self.namespace)
                                        delay_list.append([new_pod_name, node_name])
                                    else:
                                        print("> Keeping pod [{}] in node {}, no schedule".format(pod_name, node_name))
                                else:
                                    print("Error: Unhandled pod status: {}".format(pod_status))
                for elem in delay_list:
                    self.schedule(elem[0], elem[1], self.namespace)
            except k8s_client.rest.ApiException as e:
                raise Exception(json_loads(e.body)['message'])
        else:
            print('Warning: no configuration returned from optimizer: {} AGAIN'.format(result['error']))
            if self.no_solution_found_warning:
                for event in regular_events:
                    self.warn_no_solution_found(event, self.namespace)


    def get_event_batch(self, previous, batch_limit, time_limit):
        '''Returns a batch of N events, or what could be found within the time limit.'''
        w = k8s_watch.Watch()
        batch = []
        runningBatch = []
        for event in w.stream(self.api.list_namespaced_pod, self.namespace, timeout_seconds=time_limit):
            if event['object'].status.phase == 'Running' and (not self.require_scheduler_name_spec or event['object'].spec.scheduler_name == self.name):
                runningBatch.append(event)
            if event['object'].status.phase == 'Pending' and (not self.require_scheduler_name_spec or event['object'].spec.scheduler_name == self.name):
                if not previous or not event['object'].metadata.name in previous:
                    batch.append(event)
                    if len(batch) == batch_limit:
                        w.stop()
                        return batch
        return batch, runningBatch


    async def start(self, batch_limit, time_limit):
        '''Receive and process scheduling events once the optimizer is ready.'''
        self.wait_for_optimizer()
        print('Listning for scheduling events to process.')
        previous = None
        while True:
            batch, runningBatch = await asyncio.to_thread(partial(self.get_event_batch, previous, batch_limit, time_limit))
            if batch:
                start_time = perf_counter()
                scheduleFlag = self.schedule_events(batch)
                finish_time = perf_counter()
                previous = [d['object'].metadata.name for d in batch]
                print('Finished processing {} pods in {:.4f} seconds'.format(len(batch), finish_time-start_time))
                
                if not scheduleFlag:
                    print("===========[Trying Reschedule]===========")
                    start_time = perf_counter()
                    self.reschedule_events(batch, runningBatch)
                    finish_time = perf_counter()
                    print('Finished processing {} pods in {:.4f} seconds'.format(len(batch), finish_time-start_time))

                batch.clear()


if __name__ == '__main__':
    HealthServer(port=int(getenv('BOREAS_SCHEDULER_HEALTHPORT', 10251))).start()
    scheduler = Scheduler()
    asyncio.run(scheduler.start(batch_limit=int(getenv('BOREAS_SCHEDULER_BATCHSIZE', 99)),
                                time_limit=int(getenv('BOREAS_SCHEDULER_BATCHTIME', 30))))
