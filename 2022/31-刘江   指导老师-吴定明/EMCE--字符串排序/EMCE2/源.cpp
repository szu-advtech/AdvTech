#include <iostream>
#include<algorithm>
#include<cctype>
#include<fstream>
#include <string>
using namespace std;
int getnum(int tmp);//��ȡһ������λ��
int getmax(int a[], int size);//��ȡ�������λ��
void count_sort(int a[], int size, int exp);//�����ض���ĳһλ��������
void radixsort(int a[], int size);//��������


int main()
{
	int n = 0, maxsize = 0;
	string* A;//���ԭʼ�ַ���
	cout << "�������ַ�����������\n";
	cin >> n;
	A = new string[n];
	cout << "�������Ӧ�������ַ�����\n";

	for (int i = 0; i != n; i++)
	{
		cin >> A[i];
		if (A[i].length() > maxsize)
			maxsize = A[i].length();
	}
	cout << maxsize << endl;//ԭʼ�ַ������ַ�����ĳ���
	for (int i = 0; i != n; i++) //��ӡ������ַ���
	{
		cout << A[i] << "  ";
	}
	//(1)
	string** swp;//��̬�����ά���� n�� maxsize��----ÿ�����δ��������ַ���
	swp = new string * [n];
	for (int i = 0; i < n; i++) {
		swp[i] = new string[maxsize];
	}

	for (int i = 0; i < n; i++)//��ʼ����ά����Ϊȫ0
	{
		for (int j = 0; j < maxsize; j++)
		{
			swp[i][j] = "0";
		}
	}
	//(2)
	string** swp2;//��̬�����ά���� maxsize�� n ��----ÿ�����δ��---ULi
	swp2 = new string * [maxsize];
	for (int i = 0; i < maxsize; i++) {
		swp2[i] = new string[n];
	}
	for (int i = 0; i < maxsize; i++)//��ʼ����ά����Ϊȫ0
	{
		for (int j = 0; j < n; j++)
		{
			swp2[i][j] = "0";
		}
	}

	cout << endl;
	cout << "ȫΪ 0 ��swp���飺" << endl;
	for (int i = 0; i < n; i++)//��ӡ��ά����Ԫ��
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}

	for (int i = 0; i != n; i++) //ÿһ�д��������ַ�������һ�ж�Ӧ����ĵ�һ���ַ���
	{
		string str = A[i];
		for (int j = 0; j < str.length(); j++)
		{
			swp[i][j] = str.substr(j, 1);
		}
	}

	cout << endl;
	cout << "����ԭʼ�ַ�����swp���飺" << endl;
	for (int i = 0; i < n; i++)//��ӡ��ά����Ԫ��
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}

	// B����--���ULi�����ַ��б�
	string* B;
	B = new string[n];

	for (int i = 0; i < n; i++)
	{
		B[i] = "0";
	}
	int swp2_i = 0;
	//�� ULi ����swap2 ��ά������
	for (int j = 0; j < maxsize; j++)//swap--��
	{
		if (swp2_i != maxsize)
		{
			for (int i = 0; i < n; i++)//swap--��
			{
				B[i] = swp[i][j];//ÿһ�ж�Ӧһ��ULi�����ж�ȡ
			}
			cout << endl;
			cout << "������ULi" << endl;
			sort(B, B + n);//��������ULi�ڵ�Ԫ�أ���������ظ���Ԫ�أ�
			for (int i = 0; i != n; i++) //��ӡ�����ULi
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
	//��swp������ԭ�е�0�滻Ϊ$
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < maxsize; j++)
		{
			if (swp[i][j] == "0")
				swp[i][j] = "$";
		}
	}
	cout << endl;
	cout << "��swp������ԭ�е�0�滻Ϊ $ ���swp���飺" << endl;
	for (int i = 0; i < n; i++)//��ӡ��ά����Ԫ��
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}
	*/

	cout << endl;
	cout << "��ӡswp2 ����,ÿ�ж�Ӧһ��ULi��" << endl;
	for (int i = 0; i < maxsize; i++)//��
	{
		for (int j = 0; j < n; j++)//��
		{
			cout << swp2[i][j];
		}
		cout << endl;
	}

	//-------------------------------

//��3��swp3������ swp2 ÿ���ַ��ı���ֵ
	int** swp3;//��̬�����ά���� maxsize�� n ��----ÿ�����δ�Ŷ�Ӧswp�����ж�Ӧ�����ַ��ı���ֵ
	swp3 = new int* [maxsize];
	for (int i = 0; i < maxsize; i++) {
		swp3[i] = new int[n];
	}
	for (int i = 0; i < maxsize - 1; i++)//��ʼ����ά����ǰmaxsize-1��Ϊ0�����һ�����⴦��
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
	cout << "��ӡswp3 ����,ÿ�ж�Ӧ��ULi���ַ��ı���ֵ��" << endl;
	for (int i = 0; i < maxsize; i++)//��
	{
		for (int j = 0; j < n; j++)//��
		{
			cout << swp3[i][j];
		}
		cout << endl;
	}

	//��4��M_M������ swp2 ÿ���ַ���������
	int** M_M;//��̬�����ά���� maxsize�� 2n ��----ÿ�����δ�Ŷ�ӦULi�ж�Ӧ�ַ���������ֵ
	M_M = new int* [maxsize];
	for (int i = 0; i < maxsize; i++) {
		M_M[i] = new int[2 * n];
	}
	for (int i = 0; i < maxsize; i++)//��ʼ����ά����ǰmaxsize-1��Ϊ0�����һ�����⴦��
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
	cout << "��ӡ��ʼ����M_M ���飬ÿ�ж�ӦULi�и��ַ��������ޣ�" << endl;
	for (int i = 0; i < maxsize; i++)//��
	{
		for (int j = 0; j < 2 * n; j++)//��
		{
			cout << M_M[i][j];
		}
		cout << endl;
	}

	//����ULi�и����ַ��ı���ֵ
	string* c_ch;//���������ַ�
	c_ch = new string[n];
	for (int i = 0; i < n; i++)
	{
		c_ch[i] = "0";
	}

	for (int i = maxsize - 2; i >= 0; i--)//�����ڶ���--�����������--swp2
	{
		for (int j = 0; j < n; j++)//��---swp2
		{
			int c_ch_i = 0;//c_ch_iΪc_ch�����±�ֵ
			string min = "0", max = "0";//�洢��С������ַ�
			int min_ch = 0, max_ch = 0;//����ֵ������ֵ
			for (int i = 0; i < n; i++)
			{
				c_ch[i] = "0";
			}
			//��swp2������ÿ���ַ���c_ch��С������ַ�������swp�����i�е�ÿһ��
			for (int k = 0; k < n; k++)
			{
				if (swp2[i][j] == swp[k][i])
				{
					int flag = 0;
					for (int c_i = 0; c_i < n; c_i++)//������c_ch�����в����ظ����ַ�
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
			cout << "δ�����c_ch" << endl;
			for (int i = 0; i < n; i++)
			{
				cout << c_ch[i] << " ";
			}
			//����---����С�ַ�������ַ�
			sort(c_ch, c_ch + n);//��������c_ch�ڵ�Ԫ�أ���������ظ���Ԫ�أ�

			cout << endl;
			cout << "������c_ch" << endl;
			for (int i = 0; i < n; i++)
			{
				cout << c_ch[i] << " ";
			}
			//����--����С���ַ�
			for (int k = 0; k < n; k++) {
				if (c_ch[k] != "0") {
					min = c_ch[k];
					break;
				}
			}
			//����ַ�
			max = c_ch[n - 1];
			//����swp2��Ӧ����һ�У��ҵ���С������ַ������±꣬�ٴ�M_M�����ڶ�ȡ���¡�����ֵ
			int swp2_i = i + 1;
			if (max != "0" && min != "0")
			{
				for (int swp2_j = 0; swp2_j < n; swp2_j++) {
					if (swp2[swp2_i][swp2_j] == min) {
						min_ch = M_M[swp2_i][2 * swp2_j];//��ȡ����ֵ
					}
					if (swp2[swp2_i][swp2_j] == max) {
						max_ch = M_M[swp2_i][(2 * swp2_j) + 1];//��ȡ����ֵ
					}
				}
				int Gi = M_M[i][2 * j] - min_ch;//��ǰ�ַ��ı���ֵ
				M_M[i][(2 * j) + 1] = Gi + max_ch;//��ǰ�ַ�������ֵ
				M_M[i][(2 * j) + 2] = M_M[i][(2 * j) + 1] + 1;//��һ���ַ������޵��ڵ�ǰ�ַ�������+1
				swp3[i][j] = Gi;//�洢�����ַ��ı���ֵ
			}
		}
	}
	cout << endl;
	cout << "��ӡ������M_M ���飬ÿ�ж�ӦULi�и��ַ��������ޣ�" << endl;
	for (int i = 0; i < maxsize; i++)//��
	{
		for (int j = 0; j < 2 * n; j++)//��
		{
			cout << M_M[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	cout << "��ӡswp3���飬��Ӧswp2 ÿ��Ԫ�صı���ֵ��" << endl;
	for (int i = 0; i < maxsize; i++)//��
	{
		for (int j = 0; j < n; j++)//��
		{
			cout << swp3[i][j] << " ";
		}
		cout << endl;
	}


	//����ÿ���ַ�����Ӧ�ı���ֵ
	int* Enc_str;
	Enc_str = new int[n];//���ÿ���ַ�����Ӧ�ı���ֵ
	for (int i = 0; i < n; i++)//swp����--��
	{
		int Enc = 0; //��¼�����ַ����ı���ֵ
		for (int j = 0; j < maxsize; j++)//swp����--��
		{
			for (int swp2_j = 0; swp2_j != n; swp2_j++) {
				if (swp[i][j] == swp2[j][swp2_j]) {
					Enc += swp3[j][swp2_j];
				}
			}
		}
		Enc_str[i] = Enc;//����һ���ַ����ı���ֵ�洢
	}

	//��ӡÿ���ַ�����Ӧ�ı���ֵ
	cout << endl;
	for (int i = 0; i != n; i++) //��ӡ������ַ���
	{
		cout << A[i] << "  ";
	}
	cout << endl;
	cout << "ÿ���ַ�����Ӧ�ı���ֵΪ��" << endl;
	for (int i = 0; i < n; i++) {
		cout << Enc_str[i] << "\t";
	}

	int* Enc_sort;
	Enc_sort = new int[n];//��������
	for (int i = 0; i < n; i++)
	{
		Enc_sort[i] = Enc_str[i];
	}


	radixsort(Enc_sort, n);//���û�������

	cout << endl;
	cout << "�����ı���ֵ:" << endl;
	for (int i = 0; i < n; i++) {
		cout << Enc_sort[i] << "\t";
	}
	cout << endl;
	cout << "�������ַ���:" << endl;
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
int getnum(int tmp)//��ȡһ������λ��
{
	int count = 0;
	while (tmp) {
		count++;
		tmp /= 10;
	}
	return count;
}
int getmax(int a[], int size)//��ȡ�������λ��
{
	int maxd = -100;
	for (int i = 0; i < size; i++) {
		maxd = max(maxd, getnum(a[i]));
	}
	return maxd;
}
void count_sort(int a[], int size, int exp)//�����ض���ĳһλ��������
//���е�exp��ʾ��ȡһ������λ�������ո��������λ���ϵĴ�С�������򣬱���exp=1ʱ���ո�λ����exp=2ʱ����ʮλ�����Դ�����
{
	int* output = (int*)malloc(size * sizeof(int));
	int buckets[10] = { 0 };
	memset(buckets, 0, sizeof(buckets));
	for (int i = 0; i < size; i++) {
		buckets[(a[i] / exp) % 10]++;
	}
	for (int i = 1; i < 10; i++) {
		buckets[i] += buckets[i - 1];//��ȡĳһλ��output�����е�λ��,ͨ���ۼӷ�����һ�´�0-9һ��ռ���˶���output��λ��
	}
	for (int i = size - 1; i >= 0; i--) {
		output[buckets[(a[i] / exp) % 10] - 1] = a[i];
		buckets[(a[i] / exp) % 10]--;
	}
	for (int i = 0; i < size; i++) {
		a[i] = output[i];
	}
}
void radixsort(int a[], int size)//��������
{
	int exp;
	int max = getmax(a, size);
	for (exp = 1; exp < pow(10, max); exp *= 10) {
		count_sort(a, size, exp);
	}
}




