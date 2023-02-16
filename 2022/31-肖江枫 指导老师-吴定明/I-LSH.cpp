#include <iostream>
#include <random>
#include <array>
#include <algorithm>
#include <cmath>
//#include "B+Tree.cpp"

#define DIMENSION 100 //数据集维度d。
#define MLSHFUNCTIONS 10 //m个LSH函数。
#define DATAAMOUNT 100 //暂定有100条数据。

typedef struct {
	int dn; //datanum数据编号。
	double dp; //dataprojection数据投影值。
	bool visited; //是否已被访问？
}nppair; //(num,projection)pair结构体。

std::array<std::array<double, DIMENSION>, MLSHFUNCTIONS> LSH; //一个长为m的array，内容是一个长为d的double型array。
std::array<std::array<double, DIMENSION>, DATAAMOUNT> Data; //一个长为数据量长度的array，内容是一个长为d的double型array。
std::array<std::array<nppair, DATAAMOUNT>, MLSHFUNCTIONS> Projection; //一个长为m的array，内容是一个数据量长度的double型array。
std::array<double, DIMENSION> query; //一个长为d的double型查询向量。
std::array<double, MLSHFUNCTIONS> qp; //一个长为m的double型查询向量的投影向量。

std::random_device rd; //rd():真随机数函数。
std::default_random_engine randomgen{ rd() }; //randomgen():给定初始种子为真随机数的伪随机数函数。
//std::default_random_engine rng1; //默认初始种子的伪随机数函数。
//std::default_random_engine rng2{100}; //给定初始种子为100的伪随机数函数。
std::normal_distribution<double> p2stable(0, 1); //p2stable(randomgen):标准正态分布随机变量取值，需要随机数生成器。

void InitialLSHFunction() {
	std::cout << "正在随机初始化LSH函数......";
	for (int i = 0; i < MLSHFUNCTIONS; i++) {
		for (int j = 0; j < DIMENSION; j++) {
			LSH[i][j] = p2stable(randomgen);
			//std::cout << LSH[i][j] << std::endl;
		}
	}
	std::cout << "已完成！" << std::endl;
}//初始化m个d维向量，用于LSH。

void InitialData() {
	std::cout << "正在随机初始化数据向量......";
	for (int i = 0; i < DATAAMOUNT; i++) {
		for (int j = 0; j < DIMENSION; j++) {
			Data[i][j] = randomgen()/1000000; //随机数大小暂定
			//std::cout << ++testsum << std::endl;
		}
	}
	std::cout << "已完成！" << std::endl;
}//初始化DATAAMOUNT个d维向量，作为测试数据集。

double DotProduct(std::array<double, DIMENSION> a, std::array<double, DIMENSION> b) {
	double sum = 0;
	for (int d = 0; d < DIMENSION; d++) {
		sum += a[d] * b[d];
	}
	return sum;
}//d维向量内积。

void DataProject() {
	std::cout << "正在完成数据向量的LSH映射......";
	for (int i = 0; i < MLSHFUNCTIONS; i++) {
		for (int j = 0; j < DATAAMOUNT; j++) {
			Projection[i][j].dn = j+1;
			Projection[i][j].dp = DotProduct(LSH[i], Data[j]);
			//Projection[i][j].visited = false;
			//std::cout << Projection[i][j].dn << std::endl;
		}
	}
	std::cout << "已完成！" << std::endl;
}//计算出数据再m个投影维度的投影值。

void ResetDataProjectVisited() {
	for (int i = 0; i < MLSHFUNCTIONS; i++) {
		for (int j = 0; j < DATAAMOUNT; j++) {
			Projection[i][j].visited = false;
		}
	}
}//重置Projection的visited。

bool nppcompair(nppair a, nppair b) {
	return a.dp < b.dp;
}//nppair的比较器。

void ProjectionSort() {
	std::cout << "正在完成搜索顺序表的排序......";
	for (int i = 0; i < MLSHFUNCTIONS; i++) {
		sort(Projection[i].begin(), Projection[i].end(), nppcompair);
	}
	std::cout << "已完成！" << std::endl;
}//对整个Projection完成排序。

int QuickSearch(std::array<nppair, DATAAMOUNT> npp, int head, int rear, double qp) {
	if (head == rear - 1) return ((qp - npp[head].dp) < (npp[rear].dp - qp)) ? head : rear;
	int middle = (head + rear) / 2;
	return (npp[middle].dp > qp) ? QuickSearch(npp, head, middle, qp) : QuickSearch(npp, middle, rear, qp);
}//对nppair按一个维度的数据投影值二分查找。

void GetQuery() {
	std::cout << "现在开始输入查询向量。" << std::endl;
	for (int d = 0; d < DIMENSION; d++) {
		std::cout << "请输入查询向量的第" << d + 1 << "个分量：";
		std::cin >> query[d];
	}
	std::cout << "查询向量的输入已完成，其内容为：(" << query[0];
	for (int d = 1; d < DIMENSION; d++) {
		std::cout << ", " << query[d];
	}
	std::cout << ")。" << std::endl;
}//获取一个查询向量。

void QueryProject() {
	std::cout << "正在完成查询向量的LSH映射......";
	for (int i = 0; i < MLSHFUNCTIONS; i++) {
		qp[i] = DotProduct(query, LSH[i]);
		//std::cout << qp[i];
	}
	std::cout << "已完成！" << std::endl;
}//完成查询向量的投影。

double Distance(std::array<double, DIMENSION> a, std::array<double, DIMENSION> b) {
	double sum = 0;
	for (int d = 0; d < DIMENSION; d++) {
		sum += pow(a[d], 2) * pow(b[d], 2);
	}
	return sqrt(sum);
}//d维向量距离。

std::array<int, DATAAMOUNT> cn; //记录每条向量在投影维度离查询向量距离相近的次数。
int IncrementalGeto() {
	double mptemp=DBL_MAX; //记录各投影维度中距离最小值。
	int mtemp; //记录各投影维度中距离最小值所在维度编号。
	int otemp;//记录各投影维度中距离最小值的点编号。
	int ltemp, rtemp;
	int o;
	for (int m = 0; m < MLSHFUNCTIONS; m++) {
		o = QuickSearch(Projection[m], 0, DATAAMOUNT-1, qp[m]);
		if (Projection[m][o].visited == true) {
			ltemp = rtemp = o;
			while (Projection[m][ltemp].visited == true) {
				ltemp--;
				if (ltemp == -1) break;
			}
			while (Projection[m][rtemp].visited == true) {
				rtemp++;
				if (rtemp == DATAAMOUNT) break;
			}
			if ((ltemp == -1) && (rtemp == DATAAMOUNT)) {
				std::cout << "第" << m << "维度的投影值已被查询完毕。" << std::endl;
				continue;
			}
			else if (ltemp == -1) {
				o = rtemp;
			}
			else if (rtemp == DATAAMOUNT) {
				o = ltemp;
			}
			else {
				o = ((qp[m] - Projection[m][ltemp].dp) < (Projection[m][rtemp].dp - qp[m])) ? ltemp : rtemp;
			}
		}
		if (fabs(Projection[m][o].dp - qp[m]) < mptemp) {
			mptemp = Projection[m][o].dp - qp[m];
			mtemp = m;
			otemp = o;
		}
	}
	Projection[mtemp][otemp].visited = true;
	cn[otemp]++;
	return otemp;
}//递增查询

int testsum = 0;

int main(void) {
	std::cout << "I-LSH，启动！" << std::endl;
	InitialLSHFunction();
	InitialData();
	DataProject();
	ProjectionSort();
	//for (int i = 0; i < DATAAMOUNT; i++) {
	//	std::cout << "(" << Projection[0][i].dn << "," << Projection[0][i].dp << ")" << std::endl;
	//}
	char queryagain; //是否要继续查询？
	int ncan; //候选向量数量。
	double alpha, beta, c; //三个参数。
	double dmin; //与查询向量的距离最小值。
	int onum; //当前递增查询到向量的编号。
	//int mnum; //当前递增查询所处的投影维度编号。
	int omin=DATAAMOUNT; //与查询向量距离最小的向量编号。
	while (1) {
		queryagain = 'n';
		GetQuery();
		//for (int k = 0; k < DIMENSION; k++) query[k] = 1;//快速输入，测试用。
		QueryProject();
		ResetDataProjectVisited();
		ncan = 0;
		std::cout << "请输入α的值,0<α<1：";
		std::cin >> alpha;
		while ((alpha <= 0) || (alpha >= 1)) {
			std::cout << "请重新输入α的值，注意0<α<1！";
			std::cin >> alpha;
		}
		std::cout << "请输入β的值,0<β<1：";
		std::cin >> beta;
		while ((beta <= 0) || (beta>= 1)) {
			std::cout << "请重新输入β的值，注意0<β<1！";
			std::cin >> beta;
		}
		std::cout << "请输入c的值,c>1：";
		std::cin >> c;
		while (c <= 1) {
			std::cout << "请重新输入c的值，注意c>1！";
			std::cin >> c;
		}
		dmin = DBL_MAX;
		for (int d = 0; d < DATAAMOUNT; d++) cn[d] = 0;
		while (ncan < beta * DATAAMOUNT) {
			onum = IncrementalGeto();
			if (cn[onum] == floor(alpha * MLSHFUNCTIONS)) {
				if (Distance(Data[onum], query) < dmin) {
					dmin = Distance(Data[onum], query);
					omin = onum;
				}
				ncan++;
			}
		}//循环查找候选人
		std::cout << "查询向量的c-ANN是第" << omin + 1 << "条数据向量，其内容是：" << std::endl << "(" << Data[omin][0];
		for (int d = 1; d < DIMENSION; d++) {
			std::cout << ", " << Data[omin][d];
		}
		std::cout << ")" << std::endl << "是否要继续查询？若是，请输入y并回车。" << std::endl;
		std::cin >> queryagain;
		if (queryagain != 'y') break;
	}//循环查询事务

	std::cout << "感谢您的使用，祝您生活愉快！" << std::endl;
	return 0;
}