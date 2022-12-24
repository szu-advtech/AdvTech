//g++ sequential_intersect.cpp -o sequential_intersect -lgeos && ./sequential_intersect
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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
vector<candidate> C; //存储候选对 
vector<Geometry*> a; //存储R的信息 
vector<Geometry*> b; //存储S的信息 
clock_t astart;
void find_candidates()
{
	ifstream file1("WKT/STATE.csv");
	string wkt;
	STRtree treea; //R的R树
	GeometryFactory::Ptr factory = GeometryFactory::create();
	WKTReader reader(*factory); //将wkt转换为geom
	int indexa = 0;
	astart = clock();
	while (getline(file1, wkt))
	{
		wkt = wkt.substr(1, wkt.rfind("\"") - 1); //csv文件读出wkt
		//wkt = wkt.substr(wkt.find("\t")+1, wkt.rfind("\t")-wkt.find("\t")-1); //二进制或tsv文件读出wkt
		unique_ptr<Geometry> geom = reader.read(wkt); //生成geom
		geom->setSRID(indexa); //设置R的geom的SRID
		const Envelope* envelope = geom->getEnvelopeInternal(); //生成MBR
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
			}
		}
		b.push_back((Geometry*)geom.release());
		indexb++;
	}
	cout << (double)(clock() - astart) / CLOCKS_PER_SEC << endl;
}
void intersect() //串行对所有候选对进行intersects和intersection操作 
{
	double sum1, sum2;
	sum1 = sum2 = 0;
	for (int i = 0; i < C.size(); i++)
	{
		astart = clock();
		a[C[i].indexa]->intersects(b[C[i].indexb]);
		sum1 += (double)(clock() - astart) / CLOCKS_PER_SEC;
		astart = clock();
		a[C[i].indexa]->intersection(b[C[i].indexb]);
		sum2 += (double)(clock() - astart) / CLOCKS_PER_SEC;
	}
	cout << sum1 << endl << sum2 << endl;
}
int main()
{
	find_candidates();
	intersect();
	return 0;
}