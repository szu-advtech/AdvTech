#include <time.h>
#include <iostream>
#include <queue>
#include <string>
#include <algorithm>
using namespace std;
#define M 2//m*2=key数，m*2+1=值；//插入满格灌，删除一个节点不能少于(M+1)/2,若zhi为5，不能少于2
string key_words[1001];//5阶树,最多4个关键字
class Inter_Node;

class Node
{
public:
    Node();
    virtual ~Node();
    Node* GetBrother(int& flag);
    Inter_Node* Parent;
    int key[M * 2];//4
    int count;
    int isLeaf;
    void Print();
};
Node::~Node() {}

class Inter_Node : public Node
{
public:
    Inter_Node();
    virtual ~Inter_Node();//虚函数声明定义
    bool Insert(int value, Node* pNode);
    bool Delete(int value);
    int Split(Inter_Node* pNode, int key);//分裂
    bool Merge(Inter_Node* pNode);
    bool Slib(Inter_Node* pNode);
    Node* Child[M * 2 + 1];//5个查找路径
};
Inter_Node::~Inter_Node()
{
    for (int i = 0; i < M * 2 + 1; i++)
        Child[i] = NULL;
}

//叶子结点
class Leaf_Node : public Node
{
public:
    Leaf_Node();
    virtual ~Leaf_Node();
    bool Insert(int value);
    bool Delete(int value);
    int Split(Leaf_Node* pNode);
    bool Merge(Leaf_Node* pNode);
    Leaf_Node* Pre_Node;
    Leaf_Node* Next_Node;
};
Leaf_Node::~Leaf_Node() {}

//B+树
class Bplus
{
public:
    Bplus();
    virtual ~Bplus();
    bool Search(int data, string& sPath);
    bool Insert(int data);
    bool Delete(int data);
    void Print();

protected:
    Leaf_Node* Find(int data);
    bool Add_Node(Inter_Node* pNode, int key, Node* New_Node);
    bool Remove_Node(Inter_Node* pNode, int key);
    Node* Root;
};
Bplus::~Bplus() {}

//结点创建对象
Node::Node()
{
    isLeaf = true;
    count = 0;
    Parent = NULL;
}
//叶子结点创建对象
Leaf_Node::Leaf_Node()
{
    isLeaf = true;
    Pre_Node = NULL;
    Next_Node = NULL;
}
//中间结点创建对象
Inter_Node::Inter_Node()
{
    isLeaf = false;
    for (int i = 0; i < M * 2 + 1; i++)
        Child[i] = NULL;
}
//Bplus创建对象
Bplus::Bplus()
{
    Root = NULL;
}

//结点查找兄弟结点
Node* Node::GetBrother(int& flag)
{
    if (NULL == Parent)
        return NULL;

    Node* p = NULL;
    for (int i = 0; i <= Parent->count; i++)
    {
        if (Parent->Child[i] == this)
        {
            if (i == Parent->count)
            {
                p = Parent->Child[i - 1];//左兄弟 flag=1
                flag = 1;
            }
            else
            {
                p = Parent->Child[i + 1];//右兄弟 flag=2
                flag = 2;
            }
        }
    }
    return p;
}

//结点输出
void Node::Print()
{
    for (int i = 0; i < count; i++)
    {
        cout << "(" << key[i] << "," << key_words[key[i]] << ")  ";
        if (i >= count - 1)
            cout << " | ";
    }
}

//叶子结点的分裂
int Leaf_Node::Split(Leaf_Node* p)
{
    int j = 0;
    for (int i = M; i < M * 2; i++, j++)//把值copy到新节点
        p->key[j] = this->key[i];//this为old node
    this->count = this->count - j;
    p->count = j;
    return p->key[0];
}

//叶子结点删除
bool Leaf_Node::Delete(int value)
{
    bool found = false;
    int i = 0;
    for (; i < count; i++)
    {
        if (value == key[i])
        {
            found = true;
            break;
        }
    }
    if (false == found)
        return false;
    int j = i;
    for (; j < count - 1; j++)
        key[j] = key[j + 1];
    key[j] = 0;
    count--;
    return true;
}

//叶子结点的插入
bool Leaf_Node::Insert(int value)
{
    int i = 0;
    for (; (value > key[i]) && (i < count); i++)//按顺序
    {
    }
    for (int j = count; j > i; j--)//移动，找到应该插入的关键字位置
        key[j] = key[j - 1];
    key[i] = value;//插入关键字
    count++;
    return true;
}

//叶结点查找
Leaf_Node* Bplus::Find(int data)//data为关键字
{
    int i = 0;
    Node* p = Root; //?????bplus的跟
    Inter_Node* q;  //?м???
    while (NULL != p)
    {
        if (p->isLeaf) //????????????
            break;
        for (i = 0; i < p->count; i++) //??????p??key?????p不是叶子，循环至count的节点
        {
            if (data < p->key[i])
                break;
        }
        q = (Inter_Node*)p;
        p = q->Child[i];
    }

    return (Leaf_Node*)p;//把根return,如果跟为空,第一个插入函数生成的节点即为根
}

//叶子结点,合并叶子结点
bool Leaf_Node::Merge(Leaf_Node* p)
{
    if (this->count + p->count > M * 2)//如果加在一起格满说明不需要合并
        return false;
    for (int i = 0; i < p->count; i++)//否则将oldnode的关键字都插入到bro里
        this->Insert(p->key[i]);
    return true;
}

//中间结点Merge合并
bool Inter_Node::Merge(Inter_Node* p)
{
    key[count] = p->Child[0]->key[0];
    count++;
    Child[count] = p->Child[0];
    for (int i = 0; i < p->count; i++)
    {
        key[count] = p->key[i];
        count++;
        Child[count] = p->Child[i + 1];
    }
    return true;
}

//中间结点插入
bool Inter_Node::Insert(int value, Node* New)
{
    int i = 0;
    for (; (i < count) && (value > key[i]); i++)//i指向key要插入的位置
    {
    }
    for (int j = count; j > i; j--)//挪动倒地方
        key[j] = key[j - 1];
    for (int j = count + 1; j > i + 1; j--)//父亲key值改变，孩子移动；
        Child[j] = Child[j - 1];
    key[i] = value;//关键字传到父亲节点
    Child[i + 1] = New;//newnode放到该放的位置
    New->Parent = this;
    count++;
    return true;
}

//中间结点分裂
int Inter_Node::Split(Inter_Node* p, int k)
{
    int i = 0, j = 0;
    if ((k > this->key[M - 1]) && (k < this->key[M]))//分裂的地方在中间
    {
        for (i = M; i < M * 2; i++, j++)
            p->key[j] = this->key[i];//拷贝后面值进brother
        j = 1;
        for (i = M + 1; i <= M * 2; i++, j++)
        {
            this->Child[i]->Parent = p;//孩子跟着往后移动
            p->Child[j] = this->Child[i];
        }
        this->count = M;//关键子数量各位一半
        p->count = M;
        return k;
    }
    int pos = k < this->key[M - 1] ? (M - 1) : M;//看k大小和中间-1比较，定位在前面还是在后面节点
    k = this->key[pos];//pos为分裂点,定位为前还是后分裂点,最后肯定为中间值
    j = 0;
    for (i = pos + 1; i < M * 2; i++, j++)//前节点考后节点,从插入的位置分，插入以后的放到新节点
        p->key[j] = this->key[i];
    j = 0;
    for (i = pos + 1; i <= M * 2; i++, j++)//将孩子送给兄弟
    {
        this->Child[i]->Parent = p;
        p->Child[j] = this->Child[i];
    }
    this->count = pos;
    p->count = M * 2 - pos - 1;
    return k;
}

//中间结点删除
bool Inter_Node::Delete(int k)
{
    int i = 0;
    for (; (k >= key[i]) && (i < count); i++)
    {
    }
    for (int j = i - 1; j < count - 1; j++)
        key[j] = key[j + 1];
    k = i;
    for (; k < count; k++)
    {
        Child[k] = Child[k + 1];
    }
    Child[k] = NULL;
    count--;
    return true;
}

//中间结点
bool Inter_Node::Slib(Inter_Node* p)
{
    int i, j;
    if (p->key[0] < this->key[0])
    {
        for (i = count; i > 0; i--)
            key[i] = key[i - 1];
        for (j = count + 1; j > 0; j--)
            Child[j] = Child[j - 1];
        key[0] = Child[0]->key[0];
        Child[0] = p->Child[p->count];
    }
    else
    {
        key[count] = p->Child[0]->key[0];
        Child[count + 1] = p->Child[0];
        for (i = 1; i < p->count - 1; i++)
            p->key[i - 1] = p->key[i];
        for (j = 0; j < p->count - 1; j++)
            p->Child[j] = p->Child[j + 1];
    }
    this->count++;
    p->count--;
    return true;
}

//B+树添加结点
bool Bplus::Add_Node(Inter_Node* p, int k, Node* New_Node)
{
    if (NULL == p || p->isLeaf)
        return false;
    if (p->count < M * 2)//父亲不满
        return p->Insert(k, New_Node);
    Inter_Node* Brother = new Inter_Node;
    //叶子节点满，父节点也满分裂情况
    int NewKey = p->Split(Brother, k);//NewKey为需要提取并插入到root节点的值

    //确定需要插入的关键字，是插入到分裂节点的哪个位置
    if (p->count < Brother->count)
    {
        p->Insert(k, New_Node);
    }
    else if (p->count > Brother->count)
    {
        Brother->Insert(k, New_Node);
    }
    else
    {
        Brother->Child[0] = New_Node;
        New_Node->Parent = Brother;
    }
    Inter_Node* parent = (Inter_Node*)(p->Parent);
    if (NULL == parent)
    {
        parent = new Inter_Node();
        parent->Child[0] = p;
        parent->key[0] = NewKey;//newkey为分裂传回，为插入的中间值
        parent->Child[1] = Brother;
        p->Parent = parent;
        Brother->Parent = parent;
        parent->count = 1;
        Root = parent;
        return true;
    }
    return Add_Node(parent, NewKey, Brother);
}

//B+树查找data
bool Bplus::Search(int data, string& sPath)
{
    int i = 0;
    sPath = "查找路径为: ";
    Node* p = Root;
    if (NULL == p)
        return false;
    Inter_Node* q;
    while (NULL != p)
    {
        if (p->isLeaf)
            break;
        for (i = 0; (i < p->count) && (data >= p->key[i]); i++)
        {
        }
        int k = i > 0 ? i - 1 : i;
        sPath += to_string(p->key[k]);
        sPath += "-->";
        q = (Inter_Node*)p;
        p = q->Child[i];
    }
    if (NULL == p)
        return false;
    sPath += to_string(p->key[0]);
    bool found = false;
    for (i = 0; i < p->count; i++)
    {
        if (data == p->key[i])
            found = true;
        //sPath += to_string(p->key[i]);
        //sPath += "-->";
    }
    if (found)
    {
        sPath += " OK";
    }
    else
    {
        sPath += " FAIL";
    }
    return found;
}

//B+树的插入
bool Bplus::Insert(int data) //data为插入的关键字
{

    string a;
    if (true == Search(data, a))//查找要插入的值
        return false;

    Leaf_Node* Old_Node = Find(data);//找到需要插入的叶子节点定义为oldnode

    if (NULL == Old_Node)
    {
        Old_Node = new Leaf_Node;//树为空
        Root = Old_Node;
    }

    if (Old_Node->count < M * 2) {//有空间插入，直接插进去并返回
        return Old_Node->Insert(data);

    }

    Leaf_Node* New_Node = new Leaf_Node;//即将分裂

    int k = Old_Node->Split(New_Node);//k为新节点第一个关键子

    Leaf_Node* OldNext = Old_Node->Next_Node;
    Old_Node->Next_Node = New_Node;//相邻叶子节点相连
    New_Node->Next_Node = OldNext;
    New_Node->Pre_Node = Old_Node;

    if (NULL != OldNext)
        OldNext->Pre_Node = New_Node;

    if (data < k)//小于newnode key[0]，插前面，否则插后面
    {
        Old_Node->Insert(data);
    }
    else
    {
        New_Node->Insert(data);
    }
    Inter_Node* parent = (Inter_Node*)(Old_Node->Parent);

    if (NULL == parent)//初始化parent，若没有父结点，新建一个
    {
        Inter_Node* New_Root = new Inter_Node;
        New_Root->Child[0] = Old_Node;
        New_Root->key[0] = k;
        New_Root->Child[1] = New_Node;
        Old_Node->Parent = New_Root;
        New_Node->Parent = New_Root;
        New_Root->count = 1;
        Root = New_Root;
        return true;
    }

    return Add_Node(parent, k, New_Node);//向父亲里插值或者分裂父亲建立新的节点
}

//B+树的删除
bool Bplus::Delete(int data)
{
    Leaf_Node* Old_Node = Find(data); //查找数据
    if (NULL == Old_Node)//树为空
        return false;
    if (false == Old_Node->Delete(data)) //删除
        return false;
    Inter_Node* parent = (Inter_Node*)(Old_Node->Parent);
    if (NULL == parent)
    {
        if (0 == Old_Node->count)//将整棵树删掉，没父亲没key
        {
            delete Old_Node;
            Root = NULL;
        }
        return true;
    }
    if (Old_Node->count >= M)
    {
        for (int i = 0; (i < parent->count) && (data >= parent->key[i]); i++)
        {
            if (parent->key[i] == data)//如果要删除的key比父亲索引他的值要大就直接删除，如果相等，就给父亲一个新索引
                parent->key[i] = Old_Node->key[0];
        }
        return true;
    }
    //不满足规格，要合并或借值
    int flag = 1;
    Leaf_Node* Brother = (Leaf_Node*)(Old_Node->GetBrother(flag));
    int NewData = 0;
    if (Brother->count > M)//借值
    {
        if (1 == flag)//左兄弟
        {
            NewData = Brother->key[Brother->count - 1];//要被借走的数据
        }
        else//右兄弟
        {
            NewData = Brother->key[0];
        }
        Old_Node->Insert(NewData);
        Brother->Delete(NewData);
        //替换parent中的key值
        if (1 == flag)
        {
            for (int i = 0; i <= parent->count; i++)//向左兄弟借值
            {
                if (parent->Child[i] == Old_Node && i > 0)
                    parent->key[i - 1] = Old_Node->key[0];
            }
        }
        else
        {
            for (int i = 0; i <= parent->count; i++)//向右兄弟借值
            {
                if (parent->Child[i] == Old_Node && i > 0)
                    parent->key[i - 1] = Old_Node->key[0];
                if (parent->Child[i] == Brother && i > 1)
                    parent->key[i - 1] = Brother->key[0];
            }
        }
        return true;
    }
    int NewKey = 0;
    if (1 == flag)//无法借值，合并
    {
        Brother->Merge(Old_Node);
        NewKey = Old_Node->key[0];//标记要删除的父亲里的key
        Leaf_Node* OldNext = Old_Node->Next_Node;//接入后面兄弟
        Brother->Next_Node = OldNext;
        if (NULL != OldNext)
            OldNext->Pre_Node = Brother;
        delete Old_Node;
    }
    else
    {
        Old_Node->Merge(Brother);
        NewKey = Brother->key[0];
        Leaf_Node* OldNext = Brother->Next_Node;
        Old_Node->Next_Node = OldNext;
        if (NULL != OldNext)
            OldNext->Pre_Node = Old_Node;
        delete Brother;
    }
    return Remove_Node(parent, NewKey);//移除parent或者移除parent中关键字；
}

//Bplus 移除结点
bool Bplus::Remove_Node(Inter_Node* p, int k)
{
    if (false == p->Delete(k))
    {
        return false;
    }
    Inter_Node* parent = (Inter_Node*)(p->Parent);
    if (NULL == parent)
    {
        if (0 == p->count)
        {
            Root = p->Child[0];
            delete p;
        }
        return true;
    }
    if (p->count >= M)//父亲不合并
    {
        //删掉parent中的关键字
        for (int i = 0; (i < parent->count) && (k >= parent->key[i]); i++)
        {
            if (parent->key[i] == k)//看父亲的parent里有没有要删除的关键字，有就更新索引
                parent->key[i] = p->key[0];
        }
        return true;
    }
    //父亲合并
    int flag = 1;
    Inter_Node* Brother = (Inter_Node*)(p->GetBrother(flag));
    if (Brother->count > M)//父亲借值
    {
        p->Slib(Brother);
        if (1 == flag)
        {
            for (int i = 0; i <= parent->count; i++)
            {
                if (parent->Child[i] == p && i > 0)
                    parent->key[i - 1] = p->key[0];
            }
        }
        else
        {
            for (int i = 0; i <= parent->count; i++)
            {
                if (parent->Child[i] == p && i > 0)
                    parent->key[i - 1] = p->key[0];
                if (parent->Child[i] == Brother && i > 0)
                    parent->key[i - 1] = Brother->key[0];
            }
        }
        return true;
    }
    //兄弟借值
    int NewKey = 0;
    if (1 == flag)
    {
        Brother->Merge(p);
        NewKey = p->key[0];
        delete p;
    }
    else
    {
        p->Merge(Brother);
        NewKey = Brother->key[0];
        delete Brother;
    }
    return Remove_Node(parent, NewKey);
}

//Bplus输出
void Bplus::Print()
{
    Node* p = Root;
    if (NULL == p)
        return;
    Inter_Node* a;
    int H = 0;
    queue<Node*> q;
    queue<int> h;
    q.push(p);
    h.push(1);
    while (!q.empty())
    {
        p = q.front();
        if (H != h.front())
        {
            cout << endl;
            cout << H << endl;
            H = h.front();
        }
        q.pop();
        h.pop();
        p->Print();
        if (NULL != p && !p->isLeaf)
        {
            a = (Inter_Node*)p;
            for (int i = 0; i <= p->count; i++)
            {
                q.push(a->Child[i]);
                h.push(H + 1);
            }
        }
    }
}

//测试：test1 建立B+tree
void test1(Bplus* bplus, int count)//count为想要几个节点，返回值建一个树
{
    srand((unsigned)time(NULL));
    //随机种子
    cout << " 插入数据： ";
    for (int i = 1; i <= count; i++)//
    {
        //int x = i;
        int x = rand() % 1000 + 1;//随机生成x
        key_words[x] = " ";
        for (int i = 0; i < 2; i++)
        {
            int y = rand() % 26 + 65;//生成一个随机数
            int z = rand() % 26 + 97;
            key_words[x] += char(y);
            key_words[x] += char(z);
        }//生成四个字符的数据
        cout << "[" << x << " "
            << "," << key_words[x] << "]" << " ";
        bplus->Insert(x);//x=key
    }
    cout << "B+树建立suceess!" << endl;
    cout << endl;
}

//测试：test2查询
void test2(Bplus* bplus, int data)
{
    string sPath;
    bplus->Search(data, sPath);
    cout << sPath;
    cout << endl;
}

//测试：test3插入
void test3(Bplus* bplus, int data, string s)
{
    bool success = bplus->Insert(data);
    key_words[data] = s;
    if (true == success)
    {
        cout << "OK" << endl;
    }
    else
    {
        cout << "FAIL" << endl;
    }
    cout << endl;
}

//测试：test4删除
void test4(Bplus* bplus, int data)
{
    bool success = bplus->Delete(data);
    if (true == success)
    {
        cout << "成功" << endl;
    }
    else
    {
        cout << "错误" << endl;
    }
    cout << endl;
}

//测试：test5打印
void test5(Bplus* bplus)
{
    bplus->Print();
    cout << endl;
}

//测试：test6修改
void test6(Bplus* bplus, int i, string j, int k, string l)
{
    if (i == k)
    {
        key_words[i] = l;
    }
    else
    {
        //i ！= j
        //删除(i , j );
        bplus->Delete(i);
        //插入(k , l )
        bplus->Insert(k);
        //初始化结点内容
        key_words[k] = l;
    }
    cout << endl;
}

//B+树主函数
int BPlusPlus()
{
    Bplus* bplus = new Bplus();
    int x = 1;
    int y = 0;
    string s = " ";
    int i, k;
    string j = " ";
    string l = " ";
    while (0 != x)
    {
        cout << "1.请问需要多少个节点的B+tree" << endl;
        cout << "2.需要查询 " << endl;
        cout << "3.需要插入内容 " << endl;
        cout << "4.需要删除内容" << endl;
        cout << "5.需要显示B+树形状" << endl;
        cout << "6.需要修改内容" << endl;
        cout << " \n";
        cin >> x;
        switch (x)
        {
        case 1:
            cout << "请输入需要多少个结点：";
            cin >> y;
            test1(bplus, y);
            break;
        case 2:
            cout << "请输入要查询的关键值key A：";
            cin >> y;
            test2(bplus, y);
            break;
        case 3:
            cout << "请输入要插入的关键值key A和B：";
            cin >> y >> s;
            test3(bplus, y, s);
            break;
        case 4:
            cout << "请输入要删除的关键值key A：";
            cin >> y;
            test4(bplus, y);
            break;
        case 5:
            test5(bplus);
            break;
        case 6:
            cout << "请输入要删除的A和B：";
            cin >> i >> j;
            cout << "请输入要插入的A和B：";
            cin >> k >> l;
            test6(bplus, i, j, k, l);
            break;
        default:
            delete bplus;
            return 0;
            break;
        }
    }
    return 0;
}

//————————————————
//版权声明：本文为CSDN博主「叫我老伯」的原创文章，遵循CC 4.0 BY - SA版权协议，转载请附上原文出处链接及本声明。
//原文链接：https://blog.csdn.net/weixin_51309915/article/details/122592023