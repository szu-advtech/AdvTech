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

	/*cout << endl;
	cout << "ȫΪ 0 ��swp���飺" << endl;
	for (int i = 0; i < n; i++)//��ӡ��ά����Ԫ��
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}*/

	for (int i = 0; i != n; i++) //ÿһ�д��������ַ�������һ�ж�Ӧ����ĵ�һ���ַ���
	{
		string str = A[i];
		for (int j = 0; j < str.length(); j++)
		{
			swp[i][j] = str.substr(j, 1);
		}
	}

	/*cout << endl;
	cout << "����ԭʼ�ַ�����swp���飺" << endl;
	for (int i = 0; i < n; i++)//��ӡ��ά����Ԫ��
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}*/

	// B����--���ULi�����ַ��б�
	string* B;
	B = new string[n];

	for (int i = 0; i < n; i++)
	{
		B[i] = "0";
	}
	int swp2_i = 0;
	//��ULi����swap2 ��ά������
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
	//��swp������ԭ�е�0�滻Ϊ$
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < maxsize; j++)
		{
			if (swp[i][j] == "0")
				swp[i][j] = "$";
		}
	}
	/*cout << endl;
	cout << "��swp������ԭ�е�0�滻Ϊ $ ���swp���飺" << endl;
	for (int i = 0; i < n; i++)//��ӡ��ά����Ԫ��
	{
		for (int j = 0; j < maxsize; j++)
		{
			cout << swp[i][j];
		}
		cout << endl;
	}*/


	/*cout << endl;
	cout << "��ӡswp2 ���飺" << endl;
	for (int i = 0; i < maxsize; i++)//��
	{
		for (int j = 0; j < n; j++)//��
		{
			cout << swp2[i][j];
		}
		cout << endl;
	}*/

	//��ȡULi����
	int* ULi_len;
	ULi_len = new int[maxsize];//���ÿ��ULi��Ӧ�ĳ���
	for (int i = 0; i < maxsize; i++)
	{
		ULi_len[i] = 0;
	}

	for (int i = 0; i < maxsize; i++)//��
	{
		int count = 0;
		for (int j = 0; j < n; j++)//��
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
	cout << "����ULi����Ϊ��" << endl;
	for (int i = 0; i < maxsize; i++)
	{
		cout << ULi_len[i] << "\t";
	}*/

	//��ȡZi
	int* Zi;
	Zi = new int[maxsize];//���ÿ��ULi��Ӧ�ĳ���
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
	cout << "����ZiΪ��" << endl;
	for (int i = 0; i < maxsize; i++)
	{
		cout << Zi[i] << "\t";
	}

	//����ÿ���ַ�����Ӧ�ı���ֵ
	int* Enc_str;
	Enc_str = new int[n];//���ÿ���ַ�����Ӧ�ı���ֵ
	for (int i = 0; i < n; i++)//swp����--��
	{
		int Enc = 0; //��¼�����ַ����ı���ֵ
		for (int j = 0; j < maxsize; j++)//swp����--��
		{
			int low = 0;//��ʶ��Uli�е��±�ֵ
			int Enc_c = 0;//��¼�ַ����е����ַ��ı���ֵ
			string str = "0";
			str = swp[i][j];
			for (int swp2_j = 0; swp2_j != n; swp2_j++)
			{
				if (swp2[j][swp2_j] == str)
				{
					//�����ַ���ULi�е��±�
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
			//��Zi[j]��˼��㵥���ַ��ı���ֵEnc
			Enc_c = (low * Zi[j]);
			Enc += Enc_c;
		}
		Enc_str[i] = Enc;//����һ���ַ����ı���ֵ�洢
	}

	//��ӡÿ���ַ�����Ӧ�ı���ֵ
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

	/*cout << endl;
	cout << "�����ı���ֵ:" << endl;
	for (int i = 0; i < n; i++) {
		cout << Enc_sort[i] << "\t";
	}*/
	cout << endl;
	cout << "�������ַ���:" << endl;
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



