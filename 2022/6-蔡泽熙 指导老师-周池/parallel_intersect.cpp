//g++ parallel_intersect.cpp -o parallel_intersect -lgeos -fopenmp && ./parallel_intersect
#include <iostream>
#include <omp.h>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <cfloat>
#include <geos/io/WKTReader.h>
#include <geos/io/WKTReader.inl>
#include <geos/geom/GeometryFactory.h>
#include <geos/geom/Geometry.h>
#include <geos/geom/Envelope.h>
#include <geos/geom/Envelope.inl>
#include <geos/index/strtree/STRtree.h>
using namespace std;
using namespace geos::io;
using namespace geos::geom;
using namespace geos::index::strtree;
class candidate
{
public:
	int indexa; //存储候选对的r在数据集R中的下标
	int indexb; //存储候选对的s在数据集S中的下标
	double centerx; //候选对中心点的x坐标 
	double centery; //候选对中心点的y坐标
	double weight; //候选对的权重
};
class gridcell
{
public:
	Envelope envelope; //MBR
	double weight; //权重 
};
vector<candidate> C; //存储候选对 
int N; //target number of cells
vector<gridcell> G; //GlobalMBR
int P; //maximum OpenMP tasks
int counter; //number of OpenMP tasks in queue R
queue<gridcell> R;
int T; //threshold
int k; //split factor
vector<Geometry*> a; //存储R的信息 
vector<Geometry*> b; //存储S的信息 
clock_t astart;
bool cmp(gridcell x, gridcell y)
{
	return x.weight > y.weight; //降序 
}
void find_candidates()
{
	ifstream file1("WKT/STATE.csv");
	string wkt;
	STRtree treea; //R的R树
	GeometryFactory::Ptr factory = GeometryFactory::create();
	WKTReader reader(*factory); //将wkt转换为geom
	gridcell tempr;
	tempr.weight = 0;
	G.push_back(tempr);
	double minx = DBL_MAX;
	double maxx = DBL_MIN;
	double miny = DBL_MAX;
	double maxy = DBL_MIN;
	int indexa = 0;
	astart = clock();
	while (getline(file1, wkt))
	{
		wkt = wkt.substr(1, wkt.rfind("\"") - 1); //csv文件读出wkt
		//wkt = wkt.substr(wkt.find("\t")+1, wkt.rfind("\t")-wkt.find("\t")-1); //二进制或tsv文件读出wkt
		unique_ptr<Geometry> geom = reader.read(wkt); //生成geom
		geom->setSRID(indexa); //设置R的geom的SRID
		const Envelope* envelope = geom->getEnvelopeInternal(); //生成MBR
		if (envelope->getMinX() < minx)
			minx = envelope->getMinX();
		if (envelope->getMaxX() > maxx)
			maxx = envelope->getMaxX();
		if (envelope->getMinY() < miny)
			miny = envelope->getMinY();
		if (envelope->getMaxY() > maxy)
			maxy = envelope->getMaxY();
		treea.insert(envelope, geom.get()); //插入MBR
		a.push_back((Geometry*)geom.release());
		indexa++;
	}
	cout << (double)(clock() - astart) / CLOCKS_PER_SEC << endl;
	ifstream file2("WKT/PRIMARYROADS.csv");
	int indexb = 0;
	astart = clock();
	while (getline(file2, wkt))
	{
		wkt = wkt.substr(1, wkt.rfind("\"") - 1); //csv文件读出wkt
		//wkt = wkt.substr(wkt.find("\t")+1, wkt.rfind("\t")-wkt.find("\t")-1); //二进制或tsv文件读出wkt
		unique_ptr<Geometry> geom = reader.read(wkt); //生成geom
		geom->setSRID(indexb); //设置S的geom的SRID
		const Envelope* envelope = geom->getEnvelopeInternal(); //生成MBR
		if (envelope->getMinX() < minx)
			minx = envelope->getMinX();
		if (envelope->getMaxX() > maxx)
			maxx = envelope->getMaxX();
		if (envelope->getMinY() < miny)
			miny = envelope->getMinY();
		if (envelope->getMaxY() > maxy)
			maxy = envelope->getMaxY();
		vector<void*> result;
		treea.query(envelope, result); //根据S的MBR对R的R树进行查询
		for (int i = 0; i < result.size(); i++)
		{
			Geometry* geoma = (Geometry*)result[i];
			const Envelope* envelopea = geoma->getEnvelopeInternal(); //计算result的MBR 
			Envelope intersect;
			bool flag = envelope->intersection(*envelopea, intersect); //MBR的交集
			if (flag)
			{
				candidate temp;
				temp.indexa = geoma->getSRID(); //ri
				temp.indexb = indexb; //sj
				temp.centerx = (intersect.getMaxX() + intersect.getMinX()) / 2.0; //pij.x
				temp.centery = (intersect.getMaxY() + intersect.getMinY()) / 2.0; //pij.y
				size_t m = geoma->getNumPoints();
				size_t n = geom->getNumPoints();
				temp.weight = (n + m) * log(n + m); //wij
				C.push_back(temp);
				G[0].weight += temp.weight;
			}
		}
		b.push_back((Geometry*)geom.release());
		indexb++;
	}
	cout << (double)(clock() - astart) / CLOCKS_PER_SEC << endl;
	G[0].envelope = Envelope(maxx, minx, maxy, miny);
	while (!R.empty())
	{
		R.pop(); //清空R
	}
	//for(int i = 0; i < C.size(); i++)
	//cout<<"(r"<<C[i].indexa<<", s"<<C[i].indexb<<"), center: ("<<C[i].centerx<<", "<<C[i].centery<<"), weight: "<<C[i].weight<<endl;
}
void quadtree_partitioning()
{
	astart = clock();
	while (G.size() < N - T)
	{
		counter = 0;
#pragma omp parallel num_threads(P)
		{
#pragma omp master //主线程
			{
				for (int i = 0; i < P && i < G.size(); i++)
				{
					if (G[i].weight >= G[0].weight / k)
					{
						counter++;
#pragma omp task //四叉树分区 
						{
							gridcell temp1;
							temp1.envelope = Envelope((G[i].envelope.getMaxX() + G[i].envelope.getMinX()) / 2.0, G[i].envelope.getMinX(), (G[i].envelope.getMaxY() + G[i].envelope.getMinY()) / 2.0, G[i].envelope.getMinY());
							temp1.weight = 0;
							gridcell temp2;
							temp2.envelope = Envelope(G[i].envelope.getMaxX(), (G[i].envelope.getMaxX() + G[i].envelope.getMinX()) / 2.0, (G[i].envelope.getMaxY() + G[i].envelope.getMinY()) / 2.0, G[i].envelope.getMinY());
							temp2.weight = 0;
							gridcell temp3;
							temp3.envelope = Envelope(G[i].envelope.getMaxX(), (G[i].envelope.getMaxX() + G[i].envelope.getMinX()) / 2.0, G[i].envelope.getMaxY(), (G[i].envelope.getMaxY() + G[i].envelope.getMinY()) / 2.0);
							temp3.weight = 0;
							gridcell temp4;
							temp4.envelope = Envelope((G[i].envelope.getMaxX() + G[i].envelope.getMinX()) / 2.0, G[i].envelope.getMinX(), G[i].envelope.getMaxY(), (G[i].envelope.getMaxY() + G[i].envelope.getMinY()) / 2.0);
							temp4.weight = 0;
							for (int j = 0; j < C.size(); j++) //根据候选对的中心点找到对应的子单元格 
							{
								if (temp1.envelope.contains(C[j].centerx, C[j].centery))
									temp1.weight += C[j].weight;
								if (temp2.envelope.contains(C[j].centerx, C[j].centery))
									temp2.weight += C[j].weight;
								if (temp3.envelope.contains(C[j].centerx, C[j].centery))
									temp3.weight += C[j].weight;
								if (temp4.envelope.contains(C[j].centerx, C[j].centery))
									temp4.weight += C[j].weight;
							}
#pragma omp critical //保证只有一个线程执行 
							{
								R.push(temp1);
								R.push(temp2);
								R.push(temp3);
								R.push(temp4);
							}
						}
					}
				}
			}
		}
#pragma omp taskwait
		for (int i = 0; i < counter; i++)
		{
			vector<gridcell>::iterator k = G.begin();
			G.erase(k);  //删除前counter个元素
		}
		while (!R.empty())
		{
			G.push_back(R.front()); //将R赋值给G 
			R.pop(); //清空R 
		}
		sort(G.begin(), G.end(), cmp); //对G降序排序
	}
	/*cout << "0 " << G[0].weight << endl;
	for(int i = 1; i < G.size(); i++)
	{
		if(G[i].weight != G[i-1].weight)
		{
			cout << i << " " << G[i].weight << endl;
			if(G[i].weight == 0)
			break;
		}
	}
	cout << G.size() << endl;*/
	vector<vector<int> > index;
	int size;
	for (int i = 0; i < G.size(); i++) //将所有候选对分配到最后的分区结果中 
	{
		if (G[i].weight == 0)
		{
			size = i;
			break;
		}
		vector<int> cc;
		for (int j = 0; j < C.size(); j++)
		{
			if (G[i].envelope.contains(C[j].centerx, C[j].centery))
			{
				cc.push_back(j);
			}
		}
		index.push_back(cc);
	}
	cout << (double)(clock() - astart) / CLOCKS_PER_SEC << endl;
	/*#pragma omp parallel num_threads(size)
	{
		int num = omp_get_thread_num();
		for(int i = 0; i < index[num].size(); i++)
		{
			a[C[index[num][i]].indexa]->intersects(b[C[index[num][i]].indexb]);
		}
	}
	#pragma omp taskwait*/
	astart = clock();
	for (int i = 0; i < index[0].size(); i++)
	{
		a[C[index[0][i]].indexa]->intersects(b[C[index[0][i]].indexb]);
	}
	cout << (double)(clock() - astart) / CLOCKS_PER_SEC << endl;
	/*#pragma omp parallel num_threads(size)
	{
		int num = omp_get_thread_num();
		for(int i = 0; i < index[num].size(); i++)
		{
			a[C[index[num][i]].indexa]->intersection(b[C[index[num][i]].indexb]);
		}
	}
	#pragma omp taskwait*/
	astart = clock();
	for (int i = 0; i < index[0].size(); i++)
	{
		a[C[index[0][i]].indexa]->intersection(b[C[index[0][i]].indexb]);
	}
	cout << (double)(clock() - astart) / CLOCKS_PER_SEC << endl;
}
int main()
{
	find_candidates();
	N = 10000;
	P = 4;
	T = 4 * P;
	k = 2;
	quadtree_partitioning();
	return 0;
}