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

	/*cout << endl;
	cout << "全为 0 的swp数组：" << endl;
	for (int i = 0; i < n; i++)//打印二维数组元素
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}*/

	for (int i = 0; i != n; i++) //每一行存放输入的字符串，第一行对应输入的第一个字符串
	{
		string str = A[i];
		for (int j = 0; j < str.length(); j++)
		{
			swp[i][j] = str.substr(j, 1);
		}
	}

	/*cout << endl;
	cout << "存入原始字符串的swp数组：" << endl;
	for (int i = 0; i < n; i++)//打印二维数组元素
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}*/

	// B数组--存放ULi升序字符列表
	string* B;
	B = new string[n];

	for (int i = 0; i < n; i++)
	{
		B[i] = "0";
	}
	int swp2_i = 0;
	//把ULi存入swap2 二维数组内
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
	//把swp数组内原有的0替换为$
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < maxsize; j++)
		{
			if (swp[i][j] == "0")
				swp[i][j] = "$";
		}
	}
	/*cout << endl;
	cout << "把swp数组内原有的0替换为 $ 后的swp数组：" << endl;
	for (int i = 0; i < n; i++)//打印二维数组元素
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}*/


	/*cout << endl;
	cout << "打印swp2 数组：" << endl;
	for (int i = 0; i < maxsize; i++)//行
	{
		for (int j = 0; j < n; j++)//列
		{
			cout << swp2[i][j];
		}
		cout << endl;
	}*/

	//获取ULi长度
	int* ULi_len;
	ULi_len = new int[maxsize];//存放每个ULi对应的长度
	for (int i = 0; i < maxsize; i++)
	{
		ULi_len[i] = 0;
	}

	for (int i = 0; i < maxsize; i++)//行
	{
		int count = 0;
		for (int j = 0; j < n; j++)//列
		{
			if (swp2[i][0] != "0")
			{
				if (swp2[i][1] != "0") {
					if (swp2[i][j] != "0") {
						count++;
					}
				}
				else {
					count = 2;
				}
			}
			else {
				if ((swp2[i][1] != "0") && (swp2[i][2] != "0")) {
					if (swp2[i][j] != "0") {
						count++;
					}
				}
				else {
					count = 2;
				}
			}
		}
		ULi_len[i] = count;
	}
	/*cout << endl;
	cout << "各个ULi长度为：" << endl;
	for (int i = 0; i < maxsize; i++)
	{
		cout << ULi_len[i] << "\t";
	}*/

	//获取Zi
	int* Zi;
	Zi = new int[maxsize];//存放每个ULi对应的长度
	for (int i = 0; i < maxsize; i++)
	{
		Zi[i] = 1;
		if (i != (maxsize - 1)) {
			for (int j = (i + 1); j < maxsize; j++) {
				Zi[i] = ULi_len[j] * Zi[i];
			}
		}
	}
	cout << endl;
	cout << "各个Zi为：" << endl;
	for (int i = 0; i < maxsize; i++)
	{
		cout << Zi[i] << "\t";
	}

	//计算每个字符串对应的编码值
	int* Enc_str;
	Enc_str = new int[n];//存放每个字符串对应的编码值
	for (int i = 0; i < n; i++)//swp数组--行
	{
		int Enc = 0; //记录单个字符串的编码值
		for (int j = 0; j < maxsize; j++)//swp数组--列
		{
			int low = 0;//标识在Uli中的下标值
			int Enc_c = 0;//记录字符串中单个字符的编码值
			string str = "0";
			str = swp[i][j];
			for (int swp2_j = 0; swp2_j != n; swp2_j++)
			{
				if (swp2[j][swp2_j] == str)
				{
					//返回字符在ULi中的下标
					if (swp2[j][0] == "0") {
						if (swp2[j][2] != "0")
							low = (swp2_j - 1);
						else if (swp2[j][2] == "0")
							low = 1;
					}
					else {
						if (swp2[j][1] != "0")
							low = swp2_j;
						else if (swp2[j][1] == "0")
							low = 1;
					}
				}
			}
			//与Zi[j]相乘计算单个字符的编码值Enc
			Enc_c = (low * Zi[j]);
			Enc += Enc_c;
		}
		Enc_str[i] = Enc;//将第一个字符串的编码值存储
	}

	//打印每个字符串对应的编码值
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

	/*cout << endl;
	cout << "排序后的编码值:" << endl;
	for (int i = 0; i < n; i++) {
		cout << Enc_sort[i] << "\t";
	}*/
	cout << endl;
	cout << "排序后的字符串:" << endl;
	for (int Enc_sort_i = 0; Enc_sort_i < n; Enc_sort_i++) {
		for (int Enc_str_i = 0; Enc_str_i < n; Enc_str_i++) {
			if (Enc_sort[Enc_sort_i] == Enc_str[Enc_str_i]) {
				cout<< A[Enc_str_i]<<"  ";
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



