#include<iostream>
#include<set>
#include<utility>
#include<queue>
#include<fstream>
#include<cstring>
#include<unordered_map>
#include<algorithm>
#include<map>
#include<time.h>
using namespace std;

#define KOSARAK_FILE    "kosarak.dat"
#define ENRON_FILE      ".\\enron.format(data+query)\\enron.format"
#define ORKUT_FILE      "orkut-ungraph.txt"
#define KOSARAK_CARD    990002
#define ENRON_CARD      517422
#define ORKUT_CARD      2723360
#define LINENUM         260000

// static results, 一跑就定
int candidate_num = 0, index_entries_num = 0, exceptional_cases_num = 0;

typedef pair<pair<int, int>, int> simpairs;

class sim_cmp
{
    public:
        bool operator()(const simpairs &a, const simpairs &b) 
        {
            return a.second > b.second; // 按照value从大到小排列
        }
};

typedef priority_queue<simpairs, vector<simpairs>, sim_cmp> simpair_queue;

enum dtype          // dataset_type
{
    kosarak,
    enron,
    dblp,
    orkut,
    test
};

enum etype          // element type
{
    int_t,
    string_t
};

class Element
{
    public:
        etype type;
        string str;
        int no;
    
        Element() {}
        ~Element() {}

        bool operator < (const Element &e) const {
            if(type == int_t) {
                return this->no < e.no;
            } else {
                return this->str < e.str;
            }
        }
        bool operator > (const Element &e) const {
            if(type == int_t) {
                return this->no > e.no;
            } else {
                return this->str > e.str;
            }
        }
        bool operator == (const Element &e) const {
            if(type != e.type) return false;
            else{
                if(type == int_t) {
                    return this->no == e.no;
                } else {
                    return this->str == e.str;
                }
            }
        }
        Element(const Element& e) {
            this->type = e.type;
            if(e.type == int_t)
                this->no = e.no;
            else{
                string s(e.str);
                this->str = s;
            }                
        }
};

// prefix event 需要按照t_{ub}自定义一个排序规则
class Prefix_event
{
    public:
        int set_no;
        Element element;     // 假设都按字符串处理 (不用假设了！)
        int pos;
        int t_ub;

        Prefix_event() {}
        ~Prefix_event() {}
        bool operator < (const Prefix_event &e) const {
            return this->t_ub < e.t_ub;
        }   
        Prefix_event(const Prefix_event& pe) {
            set_no = pe.set_no;
            element = pe.element;
            pos = pe.pos;
            t_ub = pe.t_ub;            
        }     
};


/*
 *  为Inverted_list自定义Hash Function
 */
class hash_func
{
    public:
        size_t operator()(const Element &e) const {
            if(e.type == int_t)
                return (hash<int>()(e.type)) ^ (hash<int>()(e.no));
            else
                return (hash<int>()(e.type)) ^ (hash<string>()(e.str));
        }
};
/*
 * 定义倒排列表
 * 元素ei对应的I(ei)含义：包含ei元素的所有集合
 */
unordered_map<Element, vector<int>, hash_func> Inverted_list;
// map<Element, vector<int>> Inverted_list;

// 存储经文件处理得到的原始数据
vector<Element> kodata[KOSARAK_CARD];
vector<Element> endata[ENRON_CARD];
vector<Element> ordata[ORKUT_CARD];
vector<Element> testdata[6];

vector<int> orkut_user_no;

/*
 *  click-stream data
 *  set:     recorded behavior of a user
 *  element: a link clicked by the user
 */
void process_kosarak()
{
    // 将kosarak.dat数据读到kodata中
    ifstream readfile;
    readfile.open(KOSARAK_FILE);
    char line[LINENUM];
    char *ptr = NULL;
    char delim[] = " \t";
    int i = 0;
    line[0] = '\0';

    while(!readfile.eof()) {
        readfile.getline(line, LINENUM);
        ptr = strtok(line, delim);

        // 读出来，放到集合里
		while(ptr != NULL)
		{
            Element e;
            e.type = int_t;
            e.no = stoi(ptr);
            vector<int> vec;
            Inverted_list.insert(make_pair(e, vec));
            kodata[i].push_back(e);
			ptr = strtok(NULL, delim);
		}
        sort(kodata[i].begin(), kodata[i].end());
        i++;
    }

    readfile.close();
}

/*
 *  Real e-mail data
 *  set:     an email
 *  element: a word from the subject or the body field
 */
void process_enron()
{
    // 将enron.format数据读到endata中
    ifstream readfile;
    readfile.open(ENRON_FILE);
    char line[LINENUM];
    char *ptr = NULL;
    char delim[] = " \t";
    int i = 0;
    line[0] = '\0';

    while(!readfile.eof()) {
        readfile.getline(line, LINENUM);
        ptr = strtok(line, delim);

        // 读出来，放到集合里
		while(ptr != NULL)
		{
            // Element e;
            // e.type = string_t;
            // string s(ptr);
            // e.str = s;
            // vector<int> vec;
            // Inverted_list.insert(make_pair(e, vec));
            // endata[i].push_back(e);
			// ptr = strtok(NULL, delim);
            Element e;
            e.type = int_t;
            string s(ptr);
            e.no = hash<string>()(s);
            vector<int> vec;
            Inverted_list.insert(make_pair(e, vec));
            endata[i].push_back(e);
			ptr = strtok(NULL, delim);		
		}
        sort(endata[i].begin(), endata[i].end());
        i++;
    }

    readfile.close();
}

/*
 *  ORKUT social network data
 *  set:     a user
 *  element: a group membership of the user
 */
void process_orkut()
{
    // 将com-orkut.ungraph数据读到ordata中
    ifstream readfile;
    readfile.open(ORKUT_FILE);
    char line[LINENUM];
    char *ptr = NULL;
    char delim[] = " \t";
    int i = 0, line_first_ele = 0;
    line[0] = '\0';

    while(!readfile.eof()) {
        readfile.getline(line, LINENUM);
        ptr = strtok(line, delim);

        // 读出来，放到集合里
		while(ptr != NULL)
		{
            if(line_first_ele == 0) {
                orkut_user_no.push_back(stoi(ptr));
                line_first_ele = 1;
                ptr = strtok(NULL, delim);
            }     
            else{
                Element e;
                e.type = int_t;
                e.no = stoi(ptr);
                vector<int> vec;
                Inverted_list.insert(make_pair(e, vec));
                ordata[i].push_back(e);
			    ptr = strtok(NULL, delim);
            }
		}
        sort(ordata[i].begin(), ordata[i].end());
        i++;
        line_first_ele = 0;
    }

      readfile.close();
}

void process_test()
{
    Element e[11];
    for(int i = 0; i < 11; i++) {
        e[i].type = string_t;
        e[i].str = "e" + to_string(i+1);

        vector<int> vec;
        Inverted_list.insert(make_pair(e[i], vec));
        if(i == 0) {
            testdata[0].push_back(e[i]);
            testdata[1].push_back(e[i]);
        } else if(i == 1) {
            testdata[0].push_back(e[i]);
            testdata[1].push_back(e[i]);
        } else if(i == 2) {
            testdata[0].push_back(e[i]);
            testdata[2].push_back(e[i]);
        } else if(i == 3) {
            testdata[0].push_back(e[i]);
            testdata[3].push_back(e[i]);
        } else if(i == 4) {
            testdata[1].push_back(e[i]);
            testdata[2].push_back(e[i]);            
        } else if(i == 5) {
            testdata[0].push_back(e[i]);
            testdata[1].push_back(e[i]);            
        } else if(i == 6) {
            testdata[0].push_back(e[i]);
            testdata[1].push_back(e[i]);            
            testdata[2].push_back(e[i]);
            testdata[3].push_back(e[i]);           
        } else if(i == 7) {
            testdata[0].push_back(e[i]);
            testdata[1].push_back(e[i]);            
            testdata[2].push_back(e[i]);
            testdata[4].push_back(e[i]);            
        } else if(i == 8) {
            testdata[0].push_back(e[i]);
            testdata[1].push_back(e[i]);            
            testdata[2].push_back(e[i]);
            testdata[4].push_back(e[i]);            
        } else if(i == 9) {
            testdata[0].push_back(e[i]);
            testdata[1].push_back(e[i]);            
            testdata[2].push_back(e[i]);
            testdata[5].push_back(e[i]);            
        } else if(i == 10) {
            testdata[1].push_back(e[i]);
            testdata[2].push_back(e[i]);            
            testdata[3].push_back(e[i]);
            testdata[4].push_back(e[i]); 
            testdata[5].push_back(e[i]);            
        }
    }
}

void process_data(dtype data_type)
{
    if(data_type == test)
        process_test();

    else if(data_type == kosarak)
        process_kosarak();  
    
    else if(data_type == enron) 
        process_enron();
    
    else if(data_type == orkut)
        process_orkut();

}

priority_queue<Prefix_event> Initialize_events(dtype data_type)
{
    priority_queue<Prefix_event> events;

    switch (data_type)
    {
    case kosarak:
    {  
        vector<Element>::iterator it;    
        // 用collection初始化events：每个集合挑出第一个元素
        for(int i = 0; i < KOSARAK_CARD; i++) {
            it = kodata[i].begin();
            Prefix_event pre;
            pre.set_no = i;
            pre.element = *it;
            pre.pos = 1;
            pre.t_ub = kodata[i].size();
            events.push(pre);
        }
        break;
    }
    case enron:
    {  
        vector<Element>::iterator it;    
        // 用collection初始化events：每个集合挑出第一个元素
        for(int i = 0; i < ENRON_CARD; i++) {
            it = endata[i].begin();
            Prefix_event pre;
            pre.set_no = i;
            pre.element = *it;
            pre.pos = 1;
            pre.t_ub = endata[i].size();
            events.push(pre);
        }
        break;
    }
    case dblp:
        break;    
    case orkut:
    {  
        vector<Element>::iterator it;    
        // 用collection初始化events：每个集合挑出第一个元素
        for(int i = 0; i < ORKUT_CARD; i++) {
            it = ordata[i].begin();
            Prefix_event pre;
            pre.set_no = i;
            pre.element = *it;
            pre.pos = 1;
            pre.t_ub = ordata[i].size();
            events.push(pre);
        }
        break;
    }
    case test:
    {
        // 小规模测试
        vector<Element>::iterator it;    
        // 用collection初始化events：每个集合挑出第一个元素
        for(int i = 0; i < 6; i++) {
            it = testdata[i].begin();
            Prefix_event pre;
            pre.set_no = i;
            pre.element = *it;
            pre.pos = 1;
            pre.t_ub = testdata[i].size();
            events.push(pre);
        }
        break;
    }
    default:
        break;
    }

    return events;
}

/*
 *  init top-k pairs with value 0
 */
void init_pairs(simpair_queue &Q, int k)
{
    for(int i = 0; i < k; i++) {
        simpairs p(make_pair(-1, -1), 0);
        Q.push(p);
    }
}

Element get_element(dtype data_type, int set_no, int pos)
{
    Element e;
    switch (data_type)
    {
    case kosarak:
        e = kodata[set_no][pos];
        break;
    case test:
        e = testdata[set_no][pos];
        break;
    case enron:
        e = endata[set_no][pos];
        break;
    case orkut:
        e = ordata[set_no][pos];
        break;
    default:
        break;
    }

    return e;
}

/*
 *  Verify similarity between 2 sets
 */
int Verification(dtype data_type, int set1_no, int pos1, int set2_no, int pos2)
{
    int common_element_num = 0;
    switch (data_type)
    {
    case kosarak:
    {
        // 从set1的pos1，和set2的pos2开始向后查找，前面的已经不用查找
        int i = pos1 - 1, j = pos2 - 1, len1 = kodata[set1_no].size(), len2 = kodata[set2_no].size();
        while(i < len1 && j < len2) {
            if(kodata[set1_no][i] == kodata[set2_no][j]) {
                common_element_num++;
                i++; j++;
            }
            else if(kodata[set1_no][i] < kodata[set2_no][j]) i++;
            else j++;    
        }
        break;
    }
    case test:
    {
        // 从set1的pos1，和set2的pos2开始向后查找，前面的已经不用查找
        int i = pos1 - 1, j = pos2 - 1, len1 = testdata[set1_no].size(), len2 = testdata[set2_no].size();
        while(i < len1 && j < len2) {
            if(testdata[set1_no][i] == testdata[set2_no][j]) {
                common_element_num++;
                i++; j++;
            }
            else if(testdata[set1_no][i] < testdata[set2_no][j]) i++;
            else j++;    
        }
        break;
    }   
    case enron:
    {
        int i = pos1 - 1, j = pos2 - 1, len1 = endata[set1_no].size(), len2 = endata[set2_no].size();
        while(i < len1 && j < len2) {
            if(endata[set1_no][i] == endata[set2_no][j]) {
                common_element_num++;
                i++; j++;
            }
            else if(endata[set1_no][i] < endata[set2_no][j]) i++;
            else j++;    
        }
        break;
    }
    case orkut:
    {
        int i = pos1 - 1, j = pos2 - 1, len1 = ordata[set1_no].size(), len2 = ordata[set2_no].size();
        while(i < len1 && j < len2) {
            if(ordata[set1_no][i] == ordata[set2_no][j]) {
                common_element_num++;
                i++; j++;
            }
            else if(ordata[set1_no][i] < ordata[set2_no][j]) i++;
            else j++;    
        }
        break;
    }
    default:
        break;
    }
    return common_element_num;
}

int get_element_pos(dtype data_type, int set_no, Element e)
{
    int pos = -1;
    switch (data_type)
    {
    case kosarak:
    {
        // 其实直接用下标访问就可以了，哈哈哈
        for(vector<Element>::iterator it = kodata[set_no].begin(); it != kodata[set_no].end(); it++) {
            if((*it) == e) {
                pos = it - kodata[set_no].begin() + 1;
                break;
            }
        }
        break;
    }
    case test:
    {
        // 其实直接用下标访问就可以了，哈哈哈
        for(vector<Element>::iterator it = testdata[set_no].begin(); it != testdata[set_no].end(); it++) {
            if((*it) == e) {
                pos = it - testdata[set_no].begin() + 1;
                break;
            }
        }
        break;
    }   
    case enron:
    {
        // 其实直接用下标访问就可以了，哈哈哈
        for(vector<Element>::iterator it = endata[set_no].begin(); it != endata[set_no].end(); it++) {
            if((*it) == e) {
                pos = it - endata[set_no].begin() + 1;
                break;
            }
        }
        break;
    }
    case orkut:
    {
        // 其实直接用下标访问就可以了，哈哈哈
        for(vector<Element>::iterator it = ordata[set_no].begin(); it != ordata[set_no].end(); it++) {
            if((*it) == e) {
                pos = it - ordata[set_no].begin() + 1;
                break;
            }
        }
        break;
    }
    default:
        break;  
    }
    return pos;
}

set<pair<int, int>> H;

/*
 *  Algorithm 1: Topk-Join
 *  Input:  collection of sets (different datasets)
 *  Output: Top-k set pairs R [(set1, set2),(sim)]
 */
void Topk_Join(dtype data_type, int k, simpair_queue &R)
{
    init_pairs(R, k);
    // 潜在风险：vector 可能会过大，产生段错误
    priority_queue<Prefix_event> PQ;
    PQ = Initialize_events(data_type);      // 如果不将collection按集合大小，提前从大到小排序，再插入，是否会增加优先队列的排序工作量？
    int t_k;

    while(!PQ.empty()) {
        Prefix_event ev = PQ.top();
        PQ.pop();   // xs
        t_k = R.top().second;
        
        if(ev.t_ub <= t_k)
            break;
        
        vector<int> Ix;
        unordered_map<Element, vector<int>, hash_func>::iterator it = Inverted_list.find(ev.element);
        pair<Element, vector<int>> pev = *it;
        Ix = pev.second; // segfault
        for(auto set_no: Ix) {
            pair<int, int> p1 = make_pair(ev.set_no, set_no),
                           p2 = make_pair(set_no, ev.set_no);

            set<pair<int, int>>::iterator it1 = H.find(p1);
            if(it1 != H.end()) {
                continue;
            }     
            set<pair<int, int>>::iterator it2 = H.find(p2);
            if(it2 != H.end()) {
                continue;
            }            
            // verify the similarity of 2 sets: x & y (set_no = ev.set_no, set_no = set_no)
            // 首先取得ev.element在集合set_no中的位置，从该位置开始往后遍历找相似，避免重复
            int pos = get_element_pos(data_type, set_no, ev.element);
            if(pos == -1){  // which is impossible to reach
                cout << "ERROR in getting pos!" << endl;
            }
            int sim = Verification(data_type, ev.set_no, ev.pos, set_no, pos); // To-Do...
            H.insert(p2);
            // update similarity pairs
            simpairs sp = make_pair(p2, sim);
            if(R.size() < k) {
                R.push(sp); 
            } else if(R.top().second < sim){
                R.pop();
                R.push(sp); 
            }        
        }

        (*it).second.push_back(ev.set_no);
        int t_ub_n = ev.t_ub - 1;
        if(t_ub_n > 0) {
            Prefix_event ev1;
            ev1.set_no = ev.set_no;          
            // get next element
            ev1.element = get_element(data_type, ev.set_no, ev.pos);    // pos从1开始

            ev1.pos = ev.pos + 1;
            ev1.t_ub = t_ub_n;
            PQ.push(ev1);
        }
    }
    for(auto x: Inverted_list) {
        index_entries_num += x.second.size();
    }
    cout << "Number of candidates: " << candidate_num << endl;
    cout << "Number of Index Entries: " << index_entries_num << endl;
}

/*
 *  Algorithm 2: l-ssjoin
 *  Input:  collection of sets (different datasets)
 *  Output: Top-k set pairs R [(set1, set2),(sim)]
 */
void l_ssjoin(dtype data_type, int k, simpair_queue &R, int l) // l:fixed step size
{    
    init_pairs(R, k); 
    // 潜在风险：vector 可能会过大，产生段错误
    priority_queue<Prefix_event> PQ;
    PQ = Initialize_events(data_type);      // 如果不将collection按集合大小，提前从大到小排序，再插入，是否会增加优先队列的排序工作量？
    int t_k;
    // int t_k_star = 1917;
    
    while(!PQ.empty()) {
        Prefix_event ev = PQ.top();
        PQ.pop();   // xs
        t_k = R.top().second;
        if(ev.t_ub <= t_k)
            break;
 
        int t_ub = ev.t_ub;
        for(int j = 0; j < l; j++) {
            if(t_ub - j <= t_k)
                break;
            
            // int size = endata[ev.set_no].size();
            // if(ev.pos+j >= (size - t_k_star + 1) && ev.pos+j <= size - ev.t_ub + l) 
            //     exceptional_cases_num++; 

            Element ele = get_element(data_type, ev.set_no, ev.pos+j-1);
            vector<int> Ix;
            unordered_map<Element, vector<int>, hash_func>::iterator it = Inverted_list.find(ele);
            pair<Element, vector<int>> pev = *it; 
            Ix = pev.second; // segfault

            for(auto set_no: Ix) {
                pair<int, int> p1 = make_pair(ev.set_no, set_no),
                               p2 = make_pair(set_no, ev.set_no);
    
                set<pair<int, int>>::iterator it1 = H.find(p1);
                if(it1 != H.end()) {
                    continue;
                }         
                set<pair<int, int>>::iterator it2 = H.find(p2);
                if(it2 != H.end()) {
                    continue;
                }
                
                // verify the similarity of 2 sets: x & y (set_no = ev.set_no, set_no = set_no)
                // 首先取得ev.element在集合set_no中的位置，从该位置开始往后遍历找相似，避免重复
                int pos = get_element_pos(data_type, set_no, ele);
                if(pos == -1){  // which is impossible to reach
                    cout << "ERROR in getting pos!" << endl;
                }
                int sim = Verification(data_type, ev.set_no, ev.pos, set_no, pos); // To-Do...
                H.insert(p2);
                // update similarity pairs
                simpairs sp = make_pair(p2, sim);
                if(R.size() < k) {
                    R.push(sp);
                } else if(R.top().second < sim){
                    R.pop();
                    R.push(sp);
                }       
            }
            (*it).second.push_back(ev.set_no);
            // t_ub -= 1;           // why?
        }
        int t_ub_n = ev.t_ub - l;
        if(t_ub_n > t_k) {
            Prefix_event ev1;
            ev1.set_no = ev.set_no;       
            // get next element
            ev1.element = get_element(data_type, ev.set_no, ev.pos+l-1);    // pos从1开始
            ev1.pos = ev.pos + l;
            ev1.t_ub = t_ub_n;
            PQ.push(ev1);
        }
    }
    // for(auto x: Inverted_list) {
    //     index_entries_num += x.second.size();
    // }
    // cout << "Number of candidates: " << H.size() << endl;
    // cout << "Number of Index Entries: " << index_entries_num << endl;
    // cout << "Number of Exceptional Cases: " << exceptional_cases_num << endl;
}

/*
 *  Algorithm 3: Adaptive Step Size Join
 *  Input:  collection of sets (different datasets)
 *  Output: Top-k set pairs R [(set1, set2),(sim)]
 */
void adaptive_ssjoin(dtype data_type, int k, simpair_queue &R)
{
    init_pairs(R, k);
    // 潜在风险：vector 可能会过大，产生段错误
    priority_queue<Prefix_event> PQ;
    PQ = Initialize_events(data_type);      // 如果不将collection按集合大小，提前从大到小排序，再插入，是否会增加优先队列的排序工作量？
    int t_k;

    int l = 1;
    while(!PQ.empty()) {
        Prefix_event ev = PQ.top();
        PQ.pop();   // xs
        t_k = R.top().second;
        
        if(l < ev.t_ub - t_k) {
            l++;
        } else {
            l = ev.t_ub - t_k;
        }
        
        if(ev.t_ub <= t_k)
            break;
 
        int t_ub = ev.t_ub;
        for(int j = 0; j < l; j++) {
            if(t_ub - j <= t_k) 
                break;

            Element ele = get_element(data_type, ev.set_no, ev.pos+j-1);
            vector<int> Ix;
            unordered_map<Element, vector<int>, hash_func>::iterator it = Inverted_list.find(ele);
            pair<Element, vector<int>> pev = *it; 
            Ix = pev.second; // segfault
            for(auto set_no: Ix) {
                pair<int, int> p1 = make_pair(ev.set_no, set_no),
                               p2 = make_pair(set_no, ev.set_no);
    
                set<pair<int, int>>::iterator it1 = H.find(p1);
                if(it1 != H.end()) {
                    continue;
                }         
                set<pair<int, int>>::iterator it2 = H.find(p2);
                if(it2 != H.end()) {
                    continue;
                }
                
                // verify the similarity of 2 sets: x & y (set_no = ev.set_no, set_no = set_no)
                // 首先取得ev.element在集合set_no中的位置，从该位置开始往后遍历找相似，避免重复
                int pos = get_element_pos(data_type, set_no, ele);
                if(pos == -1){  // which is impossible to reach
                    cout << "ERROR in getting pos!" << endl;
                }
                int sim = Verification(data_type, ev.set_no, ev.pos, set_no, pos); // To-Do...
                H.insert(p1);
                // update similarity pairs
                simpairs sp = make_pair(p2, sim);
                if(R.size() < k) {
                    R.push(sp);
                } else if(R.top().second < sim){
                    R.pop();
                    R.push(sp);
                }       
            }
            (*it).second.push_back(ev.set_no);
            // t_ub -= 1;           // why?
        }
        int t_ub_n = ev.t_ub - l;
        if(t_ub_n > t_k) {
            Prefix_event ev1;
            ev1.set_no = ev.set_no;       
            // get next element
            ev1.element = get_element(data_type, ev.set_no, ev.pos+l-1);    // pos从1开始
            ev1.pos = ev.pos + l;
            ev1.t_ub = t_ub_n;
            PQ.push(ev1);
        }
    }
    // for(auto x: Inverted_list) {
    //     index_entries_num += x.second.size();
    // }
    // cout << "Number of candidates: " << H.size() << endl;
    // cout << "Number of Index Entries: " << index_entries_num << endl;
}

void topk_join(dtype type, int k) 
{
    simpair_queue sim;
    clock_t start, end;
    
    if(type == kosarak)
        cout << "TopK-Join(k=" << k << ",data=kosarak)" << endl;
    else if(type == enron)
        cout << "TopK-Join(k=" << k << ",data=enron)" << endl;
    else if(type == test)
        cout << "TopK-Join(k=" << k << ",data=test)" << endl;
    else if(type == orkut)
        cout << "TopK-Join(k=" << k << ",data=orkut)" << endl;
        
    // top-k join
    start = clock();
    Topk_Join(type, k, sim);
    end = clock();
    
    // while(!sim.empty()) {
    //     simpairs p = sim.top();
    //     sim.pop();
    //     cout << "(" << p.first.first+1 << ", " << p.first.second+1 << ") :" << p.second << endl;
    // }
    simpairs p = sim.top();
    cout << k << "th: (" << p.first.first+1 << "," << p.first.second+1 << "):" << p.second << endl;

    cout << "running time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "----------------------------" << endl;
}

void l_join(dtype type, int k, int l)
{
    if(type == kosarak)
        cout << "l-ssjoin(k=" << k << ",l=" << l << ",data=kosarak)" << endl;
    else if(type == enron)
        cout << "l-ssjoin(k=" << k << ",l=" << l << ",data=enron)" << endl;
    else if(type == test)
        cout << "l-ssjoin(k=" << k << ",l=" << l << ",data=test)" << endl;
    else if(type == orkut)
        cout << "l-ssjoin(k=" << k << ",l=" << l << ",data=orkut)" << endl;

    simpair_queue sim;
    clock_t start, end;
    start = clock();
    l_ssjoin(type, k, sim, l);
    end = clock();
    
    // while(!sim1.empty()) {
    //     simpairs p = sim1.top();
    //     sim1.pop();
    //     cout << "(" << p.first.first+1 << ", " << p.first.second+1 << ") :" << p.second << endl;
    // }
    simpairs p = sim.top();
    cout << k << "th: (" << p.first.first+1 << "," << p.first.second+1 << "):" << p.second << endl;

    cout << "running time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "----------------------------" << endl;
}

void adaptive_join(dtype type, int k)
{
    simpair_queue sim;
    clock_t start, end;
    
    if(type == kosarak)
        cout << "adaptive-Join(k=" << k << ",data=kosarak)" << endl;
    else if(type == enron)
        cout << "adaptive-Join(k=" << k << ",data=enron)" << endl;
    else if(type == test)
        cout << "adaptive-Join(k=" << k << ",data=test)" << endl;
    else if(type == orkut)
        cout << "adaptive-Join(k=" << k << ",data=orkut)" << endl;

    // top-k join
    start = clock();
    adaptive_ssjoin(type, k, sim);
    end = clock();
    
    // while(!sim.empty()) {
    //     simpairs p = sim.top();
    //     sim.pop();
    //     cout << "(" << p.first.first+1 << ", " << p.first.second+1 << ") :" << p.second << endl;
    // }
    simpairs p = sim.top();
    cout << k << "th: (" << p.first.first+1 << "," << p.first.second+1 << "):" << p.second << endl;

    cout << "running time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;
    cout << "----------------------------" << endl;
}

int main()
{
    dtype type = orkut;
    int k = 500, l = 2;
    clock_t start, end;

    start = clock();
    process_data(type);
    end = clock();
    cout << "process data time: " << (double)(end - start) / CLOCKS_PER_SEC << endl;


    // Inverted_list会变化，需要注释
    topk_join(type, k);
    // l_join(type, k, l);
    // adaptive_join(type, k);
    
    return 0;
}