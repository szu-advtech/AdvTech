#include <cstdio>
#include <string>
#include <cstdlib>
#include "greedy.h"
#include "graph.h"
#include "common.h"
#include "cascade.h"

using namespace std;

Greedy::Greedy() 
{
	file = "greedy.txt";
}

void Greedy::Build(IGraph& gf, int num, ICascade& cascade)
{
	n = gf.GetN();
	top = num;
	d.resize(num, 0);
	list.resize(num, 0);

	bool *used= new bool[n];
	memset(used, 0, sizeof(bool)*n);
	int* set = new int[num];

	double old = 0.0;

	double *improve = new double[n];
	int *lastupdate = new int[n];
	int *heap = new int[n];
	for (int i=0; i<n; i++)
	{
		heap[i] = i;
		lastupdate[i] = -1;
		improve[i] = (double)(n+1);//initialize largest
	}

	for (int i=0; i<top; i++)
	{
		int ccc = 0;
		while (lastupdate[heap[0]] != i)
		{
			ccc++;
			lastupdate[heap[0]] = i;
			set[i] = heap[0];
			improve[heap[0]] = cascade.Run(NUM_ITER, i+1, set) - old;
			
			char tmpfilename[200];
			sprintf_s(tmpfilename, "tmp/%02d%05d.txt", i, heap[0]);

			int x = 0;
			while (x*2+2<=n-i)
			{
				int newx=x*2+1;
				if ((newx+1<n-i) && (improve[heap[newx]]<improve[heap[newx+1]]))
					newx++;
				if (improve[heap[x]]<improve[heap[newx]])
				{
					int t=heap[x];
					heap[x] = heap[newx];
					heap[newx] = t;
					x = newx;
				}
				else
					break;
			}
		}

		used[heap[0]] = true;
		set[i] = heap[0];
		list[i] = heap[0];
		d[i] = improve[heap[0]];
		old+=d[i];

		heap[0] = heap[n-i-1];
		int x = 0;
		while (x*2+2<=n-i)//bug should-1
		{
			int newx=x*2+1;
			if ((newx+1<n-i) && (improve[heap[newx]]<improve[heap[newx+1]]))	//bug should-1
				newx++;
			if (improve[heap[x]]<improve[heap[newx]])
			{
				int t=heap[x];
				heap[x] = heap[newx];
				heap[newx] = t;
				x = newx;
			}
			else
				break;
		}
	}

	WriteToFile(file, gf);

	SAFE_DELETE_ARRAY(set);
	SAFE_DELETE_ARRAY(heap);
	SAFE_DELETE_ARRAY(lastupdate);
	SAFE_DELETE_ARRAY(improve);
	SAFE_DELETE_ARRAY(used);
}

void Greedy::BuildRanking(IGraph& gf, int num, ICascade& cascade)
{
	this->n = gf.GetN();
	this->top = num;
	d.resize(num, 0);
	list.resize(num, 0);

	vector<double> improve(n, 0.0);
	int tmp[1];
	for (int i=0; i<n; i++)
	{
		tmp[0] = i;
		improve[i] = cascade.Run(NUM_ITER, 1, tmp);
		if (improve[i] > list[top - 1])
		{
			d[top - 1] = improve[i];
			list[top - 1] = i;
			int j = top - 2;
			while(j>=0)
			{
				if (improve[i] > d[j])
				{
					int int_tmp = list[j];
					double inf_tmp = d[j];
					list[j] = i;
					d[j] = improve[i];
					list[j + 1] = int_tmp;
					d[j + 1] = inf_tmp;
				}
				else
					break;
				j--;
			}
		}
	}

	WriteToFile(file, gf);
}

void Greedy::BuildFromFile(IGraph& gf, const char* name)
{
	ReadFromFile(name, gf);
}
