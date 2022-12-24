#pragma once

#include<iostream>
#include <unordered_set>
#include <vector>
#include <queue>
#include <stack>
#include <set>
using namespace std;


class MCA
{
private:
	int s = 0, n = 0;
	vector<vector<int>> graph;

    //Minimum Cost Arboroscence algorithm
    void minCostArboroscence(vector<vector<int>>& graph, int n)
    {
        vector<vector<int>> reduced_graph(n); //graph with minimum inedge weight subtracted
        vector<vector<int>> zero_inedge(n);   //graph with one zero weight inedge in all connected nodes
        vector<vector<int>> temp_graph(n);    //an extra graph which will be helpful later

        //Initialising reduced_graph and zero_inedge graph
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                reduced_graph[i].push_back(-1);
                zero_inedge[i].push_back(-1);
            }
        }

        vector<int> min_inedge(n, -1); //array to store value of minimum inedge for all verticies (-1 for non connected nodes)

        //Filling the min-inedge array
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (graph[i][j] != -1)
                {
                    if (min_inedge[j] == -1)
                    {
                        min_inedge[j] = graph[i][j];
                    }
                    else
                    {
                        min_inedge[j] = (graph[i][j] < min_inedge[j]) ? graph[i][j] : min_inedge[j];
                    }
                }
            }
        }

        //Constructing the reduced_graph
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (graph[i][j] != -1)
                {
                    reduced_graph[i][j] = graph[i][j] - min_inedge[j]; //reduced value = actual value - min inedge weight
                }
            }
        }

        //Constructing zero_inedge graph by selecting atmost one zero inedge for each edge (if it exists)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (reduced_graph[j][i] == 0)
                {
                    zero_inedge[j][i] = 0;
                    break;
                }
            }
        }

        //Detecting cycle in zero_inedge graph
        //DFS is used for cycle detection

        //colour = 0 means the node is undiscovered
        //colour = 1 means the node is discovered and still in the dfs_stack (or recursion stack)
        //colour = 2 means the node is discovered, finished and no longer in DFS stack
        vector<int> colour(n, 0);

        stack<int> dfs_stack; //DFS/Recursion stack

        dfs_stack.push(1); //push 1st node into stack
        colour[1] = 1;     //mark it discovered

        int flag = 0; //flag variable

        bool cycle = false; //boolean to store the truth value if a cycle exists or not
        int cycle_end = 0;  //supposed variable where cycle begins/ends

        //DFS
        while (!dfs_stack.empty())
        {
            int u = dfs_stack.top();
            flag = 0;

            for (int v = 0; v < n; v++)
            {
                if (zero_inedge[u][v] != -1)
                {
                    //Tree edge
                    if (!colour[v])
                    {
                        colour[v] = 1;
                        dfs_stack.push(v);
                        flag = 1;
                        break;
                    }

                    //If v is already in the recursion stack, u to v is a back-edge, that is there exists a cycle
                    else if (colour[v] == 1)
                    {
                        cycle = true;
                        cycle_end = v;
                        break;
                    }
                }
            }

            //No more processing required if a cycle exists
            if (cycle)
            {
                break;
            }

            //If a tree edge has been found, continue to next iteration
            if (flag)
            {
                continue;
            }

            //Reaches here only if no tree edge is found
            dfs_stack.pop(); //u is popped
            colour[u] = 2;   //u is marked as finished

            //This is here to ensure that all DFS trees will be evaluated/all nodes would be accessed (because we may not find a cycle in 1st DFS tree but in the second)
            if (dfs_stack.empty())
            {
                for (int i = 0; i < n; i++)
                {
                    //If any node is not discovered...
                    if (!colour[i])
                    {
                        //...push it into the stack and repeat for the tree with this root
                        dfs_stack.push(i);
                        colour[i] = 1;
                        break;
                    }
                }
            }
        }

        //If a cycle exists
        if (cycle)
        {
            set<int> supnode_nodes; //a set to contain indicies of nodes that will form the supernode

            //dfs_stack will contain all the nodes that form a cycle, with the cycle_end node at the beginning
            while (dfs_stack.top() != cycle_end)
            {
                supnode_nodes.insert(dfs_stack.top());
                dfs_stack.pop();
            }
            supnode_nodes.insert(cycle_end);

            int supernode_index = *(supnode_nodes.begin()); //choosing the lowest index to be supernode

            //Filling the temp_graph with reduced_graph matrix
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    temp_graph[i].push_back(reduced_graph[i][j]);
                }
            }

            vector<int> in_supnode_index(n, -1);  //vector to maintain to which node in the cycle was there an edge that was selected in supernode
            vector<int> out_supnode_index(n, -1); //vector to maintain from which node in the cycle was there an edge that was selected in supernode

            set<int>::iterator it;
            for (it = supnode_nodes.begin(); it != supnode_nodes.end(); it++)
            {
                int u = *it;

                //For all edges
                for (int v = 0; v < n; v++)
                {
                    //Only connections from and to supernode matter, internal connections don't matter
                    if (supnode_nodes.find(v) == supnode_nodes.end())
                    {
                        //Selecting minimum weight out_edge from supernode
                        if (temp_graph[u][v] != -1)
                        {
                            if (temp_graph[supernode_index][v] == -1)
                            {
                                temp_graph[supernode_index][v] = temp_graph[u][v];
                                out_supnode_index[v] = u;
                            }
                            else if (temp_graph[supernode_index][v] > temp_graph[u][v])
                            {
                                temp_graph[supernode_index][v] = temp_graph[u][v];
                                out_supnode_index[v] = u; //storing the index of node in cycle whose out_edge is selected
                            }
                        }

                        //Selecting minimum weight in_edge from supernode
                        if (temp_graph[v][u] != -1)
                        {
                            if (temp_graph[v][supernode_index] == -1)
                            {
                                temp_graph[v][supernode_index] = temp_graph[v][u];
                                in_supnode_index[v] = u;
                            }
                            else if (temp_graph[v][supernode_index] > temp_graph[v][u])
                            {
                                temp_graph[v][supernode_index] = temp_graph[v][u];
                                in_supnode_index[v] = u; //storing the index of node in cycle whose in_edge is selected
                            }
                        }
                    }
                }
            }

            //Disconnecting all other nodes of the cycle from the graph
            for (it = supnode_nodes.begin(); it != supnode_nodes.end(); it++)
            {
                int u = *it;
                for (int v = 0; v < n; v++)
                {
                    if (u != supernode_index)
                    {
                        temp_graph[u][v] = -1;
                        temp_graph[v][u] = -1;
                    }
                }
            }

            //Recursively calling minCostArboroscence function on the new (supernode) graph

            minCostArboroscence(temp_graph, n);

            //We have now obtained Minimum Cost Arboroscence for the supernode graph

            //Unwrapping the supernode using the in_supnode_index and out_supnode_index formed originally
            for (int v = 0; v < n; v++)
            {
                if (in_supnode_index[v] != -1)
                {
                    temp_graph[v][in_supnode_index[v]] = temp_graph[v][supernode_index];

                    //only disconnect from the supernode_index vertex if it wasn't originally its connection
                    if (in_supnode_index[v] != supernode_index)
                    {
                        temp_graph[v][supernode_index] = -1;
                    }
                }

                if (out_supnode_index[v] != -1)
                {
                    temp_graph[out_supnode_index[v]][v] = temp_graph[supernode_index][v];

                    //only disconnect from the supernode_index vertex if it wasn't originally its connection
                    if (out_supnode_index[v] != supernode_index)
                    {
                        temp_graph[supernode_index][v] = -1;
                    }
                }
            }

            //Selecting the supernode cycle edges
            for (it = supnode_nodes.begin(); it != supnode_nodes.end(); it++)
            {
                int u = *it;

                bool inedge_flag = false; //boolean to store if the node has an inedge from the minCostArboroscence graph obtained
                for (int v = 0; v < n; v++)
                {
                    if (temp_graph[v][u] != -1)
                    {
                        inedge_flag = true;
                        break;
                    }
                }

                //If it doesn't have an inedge, the edge in the zero_inedge graph is selected as the inedge for that node
                if (!inedge_flag)
                {
                    for (int v = 0; v < n; v++)
                    {
                        if (zero_inedge[v][u] != -1)
                        {
                            temp_graph[v][u] = zero_inedge[v][u];
                        }
                    }
                }
            }
        }

        //If there was no cycle...
        if (!cycle)
        {
            //...the zero_inedge graph contains the final edges
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    temp_graph[i].push_back(zero_inedge[i][j]);
                }
            }
        }

        //temp_graph contains the final edges...
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                //...so if an edge doesn't exist in temp_graph, disconnect it from original graph
                if (temp_graph[i][j] == -1)
                {
                    graph[i][j] = -1;
                }
            }
        }
    }

    void printResult()
    {
        total_cost = 0;          

        parent.resize(n, -1);
        dist.resize(n, - 1);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (graph[i][j] != -1)
                {
                    total_cost += graph[i][j];
                }
            }
        }

        vector<bool> accessed(n, false);
        queue<int> bfs_queue;

        bfs_queue.push(s);
        accessed[s] = true;

        parent[s] = 0; 
        dist[s] = 0;   

        while (!bfs_queue.empty())
        {
            int u = bfs_queue.front();
            bfs_queue.pop();

            for (int i = 0; i < n; i++)
            {
                if ((graph[u][i] != -1) && (!accessed[i]))
                {
                    accessed[i] = true;
                    bfs_queue.push(i);
                    parent[i] = u;                   
                    dist[i] = dist[u] + graph[u][i]; 
                }
            }
        }

        // for print
        // cout << total_cost << " ";
        // for (int i = 0; i < n; i++)
        // {
        //     cout << dist[i] << " ";
        // }
        // cout << endl << "----Parents----" << endl;
        // for (int i = 0; i < n; i++)
        // {
        //     cout << parent[i] << " ";
        // }
        // cout << endl;
    }

public:
    vector<int> parent;
	vector<int> dist;
    int total_cost;

	MCA(vector<vector<int>>& graph)
	{
		n = graph.size();
        s = 0;
        this->graph = graph;
		
	}

	void calMCA()
	{

        //Run the Minimum Cost Arboroscence algorithm
        minCostArboroscence(graph, n);

        //Print the result for this testcase
        printResult();

        // cout << "END" << endl;
	}

    int Get_total_cost()
    {
        return total_cost;
    }
};
