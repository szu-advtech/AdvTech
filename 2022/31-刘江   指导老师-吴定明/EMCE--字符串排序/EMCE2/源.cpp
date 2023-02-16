#include <iostream>
#include<algorithm>
#include<cctype>
#include<fstream>
#include <string>
using namespace std;
int getnum(int tmp);//获取一个数的位数
int getmax(int a[], int size);//获取数的最大位数
void count_sort(int a[], int size, int exp);//按照特定的某一位进行排序
void radixsort(int a[], int size);//基数排序


int main()
{
	int n = 0, maxsize = 0;
	string* A;//存放原始字符串
	cout << "请输入字符串的数量：\n";
	cin >> n;
	A = new string[n];
	cout << "请输入对应数量的字符串：\n";

	for (int i = 0; i != n; i++)
	{
		cin >> A[i];
		if (A[i].length() > maxsize)
			maxsize = A[i].length();
	}
	cout << maxsize << endl;//原始字符串中字符串最长的长度
	for (int i = 0; i != n; i++) //打印输入的字符串
	{
		cout << A[i] << "  ";
	}
	//(1)
	string** swp;//动态申请二维数组 n行 maxsize列----每行依次存放输入的字符串
	swp = new string * [n];
	for (int i = 0; i < n; i++) {
		swp[i] = new string[maxsize];
	}

	for (int i = 0; i < n; i++)//初始化二维数组为全0
	{
		for (int j = 0; j < maxsize; j++)
		{
			swp[i][j] = "0";
		}
	}
	//(2)
	string** swp2;//动态申请二维数组 maxsize行 n 列----每行依次存放---ULi
	swp2 = new string * [maxsize];
	for (int i = 0; i < maxsize; i++) {
		swp2[i] = new string[n];
	}
	for (int i = 0; i < maxsize; i++)//初始化二维数组为全0
	{
		for (int j = 0; j < n; j++)
		{
			swp2[i][j] = "0";
		}
	}

	cout << endl;
	cout << "全为 0 的swp数组：" << endl;
	for (int i = 0; i < n; i++)//打印二维数组元素
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}

	for (int i = 0; i != n; i++) //每一行存放输入的字符串，第一行对应输入的第一个字符串
	{
		string str = A[i];
		for (int j = 0; j < str.length(); j++)
		{
			swp[i][j] = str.substr(j, 1);
		}
	}

	cout << endl;
	cout << "存入原始字符串的swp数组：" << endl;
	for (int i = 0; i < n; i++)//打印二维数组元素
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}

	// B数组--存放ULi升序字符列表
	string* B;
	B = new string[n];

	for (int i = 0; i < n; i++)
	{
		B[i] = "0";
	}
	int swp2_i = 0;
	//把 ULi 存入swap2 二维数组内
	for (int j = 0; j < maxsize; j++)//swap--列
	{
		if (swp2_i != maxsize)
		{
			for (int i = 0; i < n; i++)//swap--行
			{
				B[i] = swp[i][j];//每一列对应一个ULi，按列读取
			}
			cout << endl;
			cout << "升序后的ULi" << endl;
			sort(B, B + n);//升序排列ULi内的元素（里面包含重复的元素）
			for (int i = 0; i != n; i++) //打印升序的ULi
			{
				cout << B[i] << "\t";
			}

			int swp2_j = 0;

			int B_i = 0;
			swp2[swp2_i][0] = B[0];
			for (int B_i = 1; B_i < n; B_i++)
			{
				if (swp2[swp2_i][swp2_j] != B[B_i])
				{
					swp2[swp2_i][swp2_j + 1] = B[B_i];
				}
				else {
					continue;
				}
				swp2_j++;
			}
		}
		swp2_i++;
	}
	
	/*
	//把swp数组内原有的0替换为$
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < maxsize; j++)
		{
			if (swp[i][j] == "0")
				swp[i][j] = "$";
		}
	}
	cout << endl;
	cout << "把swp数组内原有的0替换为 $ 后的swp数组：" << endl;
	for (int i = 0; i < n; i++)//打印二维数组元素
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}
	*/

	cout << endl;
	cout << "打印swp2 数组,每行对应一个ULi：" << endl;
	for (int i = 0; i < maxsize; i++)//行
	{
		for (int j = 0; j < n; j++)//列
		{
			cout << swp2[i][j];
		}
		cout << endl;
	}

	//-------------------------------

//（3）swp3数组存放 swp2 每个字符的编码值
	int** swp3;//动态申请二维数组 maxsize行 n 列----每行依次存放对应swp数组中对应单个字符的编码值
	swp3 = new int* [maxsize];
	for (int i = 0; i < maxsize; i++) {
		swp3[i] = new int[n];
	}
	for (int i = 0; i < maxsize - 1; i++)//初始化二维数组前maxsize-1行为0，最后一行特殊处理
	{
		for (int j = 0; j < n; j++) {
			swp3[i][j] = 0;
		}
	}
	for (int j = 0; j < n; j++) {
		if (j < 2) {
			swp3[maxsize - 1][j] = j;
		}
		else {
			if (swp2[maxsize - 1][j] != "0") {
				swp3[maxsize - 1][j] = j;
			}
			else {
				swp3[maxsize - 1][j] = 0;
			}
		}
	}

	cout << endl;
	cout << "打印swp3 数组,每行对应其ULi中字符的编码值：" << endl;
	for (int i = 0; i < maxsize; i++)//行
	{
		for (int j = 0; j < n; j++)//列
		{
			cout << swp3[i][j];
		}
		cout << endl;
	}

	//（4）M_M数组存放 swp2 每个字符的上下限
	int** M_M;//动态申请二维数组 maxsize行 2n 列----每行依次存放对应ULi中对应字符的上下限值
	M_M = new int* [maxsize];
	for (int i = 0; i < maxsize; i++) {
		M_M[i] = new int[2 * n];
	}
	for (int i = 0; i < maxsize; i++)//初始化二维数组前maxsize-1行为0，最后一行特殊处理
	{
		if (i != (maxsize - 1))
		{
			for (int j = 0; j < (2 * n); j++)
			{
				M_M[i][j] = 0;
			}
		}
		else {
			int j = 0;
			for (int k = 0; k < n; k++)
			{
				if (j != (2 * n)) {
					M_M[i][j] = swp3[maxsize - 1][k];
					M_M[i][j + 1] = swp3[maxsize - 1][k];
					j += 2;
				}
			}
		}

	}
	cout << endl;
	cout << "打印初始化的M_M 数组，每行对应ULi中各字符的上下限：" << endl;
	for (int i = 0; i < maxsize; i++)//行
	{
		for (int j = 0; j < 2 * n; j++)//列
		{
			cout << M_M[i][j];
		}
		cout << endl;
	}

	//计算ULi中各个字符的编码值
	string* c_ch;//存上下限字符
	c_ch = new string[n];
	for (int i = 0; i < n; i++)
	{
		c_ch[i] = "0";
	}

	for (int i = maxsize - 2; i >= 0; i--)//倒数第二行--从下往上填充--swp2
	{
		for (int j = 0; j < n; j++)//列---swp2
		{
			int c_ch_i = 0;//c_ch_i为c_ch数组下标值
			string min = "0", max = "0";//存储最小与最大字符
			int min_ch = 0, max_ch = 0;//下限值与上限值
			for (int i = 0; i < n; i++)
			{
				c_ch[i] = "0";
			}
			//找swp2数组内每个字符的c_ch最小与最大字符，遍历swp数组第i列的每一行
			for (int k = 0; k < n; k++)
			{
				if (swp2[i][j] == swp[k][i])
				{
					int flag = 0;
					for (int c_i = 0; c_i < n; c_i++)//避免往c_ch数组中插入重复的字符
					{
						if (c_ch[c_i] == swp[k][i + 1]) {
							flag = 1;
						}
					}
					if (flag == 0) {
						c_ch[c_ch_i] = swp[k][i + 1];
						c_ch_i += 1;
					}
				}
			}
			cout << endl;
			cout << "未排序的c_ch" << endl;
			for (int i = 0; i < n; i++)
			{
				cout << c_ch[i] << " ";
			}
			//排序---找最小字符与最大字符
			sort(c_ch, c_ch + n);//升序排列c_ch内的元素（里面包含重复的元素）

			cout << endl;
			cout << "排序后的c_ch" << endl;
			for (int i = 0; i < n; i++)
			{
				cout << c_ch[i] << " ";
			}
			//遍历--找最小的字符
			for (int k = 0; k < n; k++) {
				if (c_ch[k] != "0") {
					min = c_ch[k];
					break;
				}
			}
			//最大字符
			max = c_ch[n - 1];
			//遍历swp2对应的下一行，找到最小、最大字符的列下标，再从M_M数组内读取其下、上限值
			int swp2_i = i + 1;
			if (max != "0" && min != "0")
			{
				for (int swp2_j = 0; swp2_j < n; swp2_j++) {
					if (swp2[swp2_i][swp2_j] == min) {
						min_ch = M_M[swp2_i][2 * swp2_j];//获取下限值
					}
					if (swp2[swp2_i][swp2_j] == max) {
						max_ch = M_M[swp2_i][(2 * swp2_j) + 1];//获取上限值
					}
				}
				int Gi = M_M[i][2 * j] - min_ch;//当前字符的编码值
				M_M[i][(2 * j) + 1] = Gi + max_ch;//当前字符的上限值
				M_M[i][(2 * j) + 2] = M_M[i][(2 * j) + 1] + 1;//下一个字符的下限等于当前字符的上限+1
				swp3[i][j] = Gi;//存储单个字符的编码值
			}
		}
	}
	cout << endl;
	cout << "打印计算后的M_M 数组，每行对应ULi中各字符的上下限：" << endl;
	for (int i = 0; i < maxsize; i++)//行
	{
		for (int j = 0; j < 2 * n; j++)//列
		{
			cout << M_M[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << "打印swp3数组，对应swp2 每个元素的编码值：" << endl;
	for (int i = 0; i < maxsize; i++)//行
	{
		for (int j = 0; j < n; j++)//列
		{
			cout << swp3[i][j] << " ";
		}
		cout << endl;
	}


	//计算每个字符串对应的编码值
	int* Enc_str;
	Enc_str = new int[n];//存放每个字符串对应的编码值
	for (int i = 0; i < n; i++)//swp数组--行
	{
		int Enc = 0; //记录单个字符串的编码值
		for (int j = 0; j < maxsize; j++)//swp数组--列
		{
			for (int swp2_j = 0; swp2_j != n; swp2_j++) {
				if (swp[i][j] == swp2[j][swp2_j]) {
					Enc += swp3[j][swp2_j];
				}
			}
		}
		Enc_str[i] = Enc;//将第一个字符串的编码值存储
	}

	//打印每个字符串对应的编码值
	cout << endl;
	for (int i = 0; i != n; i++) //打印输入的字符串
	{
		cout << A[i] << "  ";
	}
	cout << endl;
	cout << "每个字符串对应的编码值为：" << endl;
	for (int i = 0; i < n; i++) {
		cout << Enc_str[i] << "\t";
	}

	int* Enc_sort;
	Enc_sort = new int[n];//用于排序
	for (int i = 0; i < n; i++)
	{
		Enc_sort[i] = Enc_str[i];
	}


	radixsort(Enc_sort, n);//调用基数排序

	cout << endl;
	cout << "排序后的编码值:" << endl;
	for (int i = 0; i < n; i++) {
		cout << Enc_sort[i] << "\t";
	}
	cout << endl;
	cout << "排序后的字符串:" << endl;
	for (int Enc_sort_i = 0; Enc_sort_i < n; Enc_sort_i++) {
		for (int Enc_str_i = 0; Enc_str_i < n; Enc_str_i++) {
			if (Enc_sort[Enc_sort_i] == Enc_str[Enc_str_i]) {
				cout << A[Enc_str_i] << "  ";
				break;
			}
		}
	}

}


//--------------------------------------
int getnum(int tmp)//获取一个数的位数
{
	int count = 0;
	while (tmp) {
		count++;
		tmp /= 10;
	}
	return count;
}
int getmax(int a[], int size)//获取数的最大位数
{
	int maxd = -100;
	for (int i = 0; i < size; i++) {
		maxd = max(maxd, getnum(a[i]));
	}
	return maxd;
}
void count_sort(int a[], int size, int exp)//按照特定的某一位进行排序
//其中的exp表示获取一个数的位数并按照该数在这个位数上的大小进行排序，比如exp=1时按照个位排序，exp=2时按照十位排序，以此类推
{
	int* output = (int*)malloc(size * sizeof(int));
	int buckets[10] = { 0 };
	memset(buckets, 0, sizeof(buckets));
	for (int i = 0; i < size; i++) {
		buckets[(a[i] / exp) % 10]++;
	}
	for (int i = 1; i < 10; i++) {
		buckets[i] += buckets[i - 1];//获取某一位在output数组中的位置,通过累加法，看一下从0-9一共占据了多少output的位置
	}
	for (int i = size - 1; i >= 0; i--) {
		output[buckets[(a[i] / exp) % 10] - 1] = a[i];
		buckets[(a[i] / exp) % 10]--;
	}
	for (int i = 0; i < size; i++) {
		a[i] = output[i];
	}
}
void radixsort(int a[], int size)//基数排序
{
	int exp;
	int max = getmax(a, size);
	for (exp = 1; exp < pow(10, max); exp *= 10) {
		count_sort(a, size, exp);
	}
}




