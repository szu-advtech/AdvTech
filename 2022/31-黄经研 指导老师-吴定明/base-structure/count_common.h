#include<string>
#include<vector>
#include<algorithm>

template<typename T>
//计算两个vector中的相同节点数量
int count_common(vector<T> &s, vector<T> &d){ 
  if(s.empty() || d.empty())
    return 0;
  int count = 0;
  if(s.size() > d.size()){
    std::sort(d.begin(), d.end());  //对d(较短)中的元素升序排列
    for(int i = 0; i < s.size(); i++){
      // 使用二分查找
      int left = 0;
      int right = d.size() - 1;
      while(left <= right){
        int mid = left + (right - left)/2;
        if(s[i] == d[mid]){
          count++;
          break;
        }else if(s[i] < d[mid]){
          right = mid - 1;
        }else{
          left = mid + 1;
        }
      }
    }
  }else{
    std::sort(s.begin(), s.end()); 
    for(int i = 0; i < d.size(); i++){
      // 使用二分查找
      int left = 0;
      int right = s.size() - 1;
      while(left <= right){
        int mid = left + (right - left)/2;
        if(d[i] == s[mid]){
          count++;
          break;
        }else if(d[i] < s[mid]){
          right = mid - 1;
        }else{
          left = mid + 1;
        }
      }
    }
  }
  return count;
}

template<typename T>
//计算s和d中相同节点的数量，并返回这些节点，即两个节点的共同邻居（当输入的s和d存储的是邻居时）
int count_common(vector<T> &s, vector<T> &d, vector<T> &common_neighbor){
  if(s.empty() || d.empty())
    return 0;
  if(s.size() > d.size()){
    std::sort(d.begin(), d.end());
    for(int i = 0; i < s.size(); i++){
      int left = 0;
      int right = d.size() - 1;
      while(left <= right){
        int mid = left + (right - left)/2;
        if(s[i] == d[mid]){
          common_neighbor.push_back(s[i]);
          break;
        }else if(s[i] < d[mid]){
          right = mid - 1;
        }else{
          left = mid + 1;
        }
      }
    }
  }else{
    std::sort(s.begin(), s.end());
    for(int i = 0; i < d.size(); i++){
      int left = 0;
      int right = s.size() - 1;
      while(left <= right){
        int mid = left + (right - left)/2;
        if(d[i] == s[mid]){
          common_neighbor.push_back(d[i]);
          break;
        }else if(d[i] < s[mid]){
          right = mid - 1;
        }else{
          left = mid + 1;
        }
      }
    }
  }
  return common_neighbor.size();
};