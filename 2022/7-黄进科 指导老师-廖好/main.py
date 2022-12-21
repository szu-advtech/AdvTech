import bnlearn as bn
from platform import node
import matplotlib.pyplot as plt
import utils as utils
from Entropic_Peeling import entropic_peeling
from Entropic_Enumeration import entropic_enumeration
import datetime
import os
# This method is copied from test_small_network() but doesn't test the Entropic Enumeraton algorithm
# Test medium networks: ["Child", "Alarm", "Insurance"]
def test_medium_network():
    output_folder = 'output/images/' + datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S') + "/"
    os.makedirs(output_folder)
    dags = ["data/child.bif", "alarm", "data/insurance.bif"]
    graphs = ["Child", "Alarm", "Insurance"]
    graphs_edges = ['(25 edges)', '(46 edges)', '(52 edges)']
    # common
    algorithms = ["PC", "Entropic Peeling", "GES"]
    colors = ["purple", "green", "red"]
    samples = [1000, 3000, 6000, 10000, 30000, 60000, 100000]
    xlabel = 'Samples'
    ylabel = 'SHD'
    # dealing with a medium graph will cost a lot of times, so here we reduce the iteration times
    # but you can change it whatever you want.
    each_sampling_iterations = 2
    print("each_sampling_iterations = %d" %each_sampling_iterations)
    begintime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("begintime:%s" %(begintime))
    # TODO change the range from 0
    for i in range(0, len(dags)):
        pc_line_x = []
        pc_line_y = []
        ep_line_x = []
        ep_line_y = []
        # ee_line_x = []
        # ee_line_y = []
        ges_line_x = []
        ges_line_y = []
        print("-----------------------------------------------------")
        utils.disable_print()
        model = bn.import_DAG(dags[i])
        utils.enable_print()
        correct_adjmat = model['adjmat']
        for sample in samples:
            iterations = each_sampling_iterations
            pc_shd_sum = 0
            ep_shd_sum = 0
            # ee_shd_sum = 0
            ges_shd_sum = 0
            for j in range(iterations):
                df = bn.sampling(model, n=sample) # get dataset
                utils.disable_print()
                pc_adjmat = bn.dag2adjmat((bn.structure_learning.fit(df, methodtype="cs"))['pdag'])
                utils.enable_print()
                ep_adjmat = entropic_peeling(df)
                # ee_adjmat = entropic_enumeration(df)
                ges_adjmat = utils.ges_wrapper(df)
                
                pc_shd = utils.SHD(correct_adjmat, pc_adjmat)
                ep_shd = utils.SHD(correct_adjmat, ep_adjmat)
                # ee_shd = utils.SHD(correct_adjmat, ee_adjmat)
                ges_shd = utils.SHD(correct_adjmat, ges_adjmat)
                pc_shd_sum += pc_shd
                ep_shd_sum += ep_shd
                # ee_shd_sum += ee_shd
                ges_shd_sum += ges_shd
            # update
            pc_line_x.append(sample)
            pc_line_y.append(pc_shd_sum/iterations)
            ep_line_x.append(sample)
            ep_line_y.append(ep_shd_sum/iterations)
            # ee_line_x.append(sample)
            # ee_line_y.append(ee_shd_sum/iterations)
            ges_line_x.append(sample)
            ges_line_y.append(ges_shd_sum/iterations)
            print("graph : %s, sample=%d, time=%s" %(graphs[i], sample, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            print("avg pc shd = %f" %(pc_shd_sum/iterations))
            print("avg peeling shd = %f" %(ep_shd_sum/iterations))
            # print("avg enumeration shd = %f" %(ee_shd_sum/iterations))
            print("avg ges shd = %f" %(ges_shd_sum/iterations))
        plt.title(graphs[i] + graphs_edges[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # update
        plt.plot(pc_line_x, pc_line_y, color=colors[0], marker='o')
        plt.plot(ep_line_x, ep_line_y, color=colors[1], marker='o', linestyle="dashed")
        # plt.plot(ee_line_x, ee_line_y, color=colors[2], marker='o')
        plt.plot(ges_line_x, ges_line_y, color=colors[2], marker='o', linestyle="dashed")
        # common
        plt.legend(algorithms)
        plt.savefig(output_folder + graphs[i] + ".jpg")
        plt.clf() #clear figure
        print("output img successfully: %s" %(output_folder + graphs[i] + ".jpg"))
        print("-----------------------------------------------------")
    endtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("endtime:%s" %(endtime))
    return 0
# Test samll networks:["Earthquake", "Survey", "Cancer", "Asia", "Sachs"]
def test_small_network():
    output_folder = 'output/images/' + datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S') + "/"
    os.makedirs(output_folder)
    # test graph
    # dags = ["data/earthquake.bif", "data/survey.bif", "data/cancer.bif", "asia", "sachs"]
    dags = ["data/earthquake.bif", "data/survey.bif", "data/cancer.bif", "asia", "sachs"]
    graphs = ["Earthquake", "Survey", "Cancer", "Asia", "Sachs"]
    graphs_edges = ['(4 edges)', '(6 edges)', '(4 edges)', '(8 edges)', "(17 edges)"]
    # graph info
    algorithms = ["PC", "Entropic Peeling", "Entropic Enumeration", "GES"]
    colors = ["purple", "green", "blue", "red"]
    samples = [1000, 3000, 6000, 10000, 30000, 60000, 100000]
    xlabel = 'Samples'
    ylabel = 'SHD'
    # you can change each_sampling_iterations whatever you want
    # the larger each_sampling_iterations is, the more running time it costs
    each_sampling_iterations = 1
    begintime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("begintime:%s" %(begintime))
    for i in range(len(dags)):
        pc_line_x = []
        pc_line_y = []
        ep_line_x = []
        ep_line_y = []
        ee_line_x = []
        ee_line_y = []
        ges_line_x = []
        ges_line_y = []
        print("-----------------------------------------------------")
        model = bn.import_DAG(dags[i])
        correct_adjmat = model['adjmat']
        for sample in samples:
            iterations = each_sampling_iterations
            pc_shd_sum = 0
            ep_shd_sum = 0
            ee_shd_sum = 0
            ges_shd_sum = 0
            for j in range(iterations):
                df = bn.sampling(model, n=sample) # get dataset
                pc_adjmat = bn.dag2adjmat((bn.structure_learning.fit(df, methodtype="cs"))['pdag'])
                ep_adjmat = entropic_peeling(df)
                ee_adjmat = entropic_enumeration(df)
                ges_adjmat = utils.ges_wrapper(df)
                
                pc_shd = utils.SHD(correct_adjmat, pc_adjmat)
                ep_shd = utils.SHD(correct_adjmat, ep_adjmat)
                ee_shd = utils.SHD(correct_adjmat, ee_adjmat)
                ges_shd = utils.SHD(correct_adjmat, ges_adjmat)
                pc_shd_sum += pc_shd
                ep_shd_sum += ep_shd
                ee_shd_sum += ee_shd
                ges_shd_sum += ges_shd
            # store result
            pc_line_x.append(sample)
            pc_line_y.append(pc_shd_sum/iterations) # store average
            ep_line_x.append(sample)
            ep_line_y.append(ep_shd_sum/iterations)
            ee_line_x.append(sample)
            ee_line_y.append(ee_shd_sum/iterations)
            ges_line_x.append(sample)
            ges_line_y.append(ges_shd_sum/iterations)
            print("graph : %s, sample=%d, time=%s" %(graphs[i], sample, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            print("avg pc shd = %f" %(pc_shd_sum/iterations))
            print("avg peeling shd = %f" %(ep_shd_sum/iterations))
            print("avg enumeration shd = %f" %(ee_shd_sum/iterations))
            print("avg ges shd = %f" %(ges_shd_sum/iterations))
        plt.title(graphs[i] + graphs_edges[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # update
        plt.plot(pc_line_x, pc_line_y, color=colors[0], marker='o')
        plt.plot(ep_line_x, ep_line_y, color=colors[1], marker='o', linestyle="dashed")
        plt.plot(ee_line_x, ee_line_y, color=colors[2], marker='o')
        plt.plot(ges_line_x, ges_line_y, color=colors[3], marker='o', linestyle="dashed")
        # common
        plt.legend(algorithms)
        plt.savefig(output_folder + graphs[i] + ".jpg")
        plt.clf() #clear figure
        print("output img successfully: %s" %(output_folder + graphs[i] + ".jpg"))
        print("-----------------------------------------------------")
    endtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("endtime:%s" %(endtime))
    return 0
def main():
    test_small_network()
    print("Small networks test finished.")
    # test_medium_network()
    # print("Medium networks test finished.")
main()