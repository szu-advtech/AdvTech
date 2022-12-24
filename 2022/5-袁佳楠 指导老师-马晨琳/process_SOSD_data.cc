// 处理SOSD 二进制文件的脚本
// 转化为csv文件, 归档文件备份

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

std::vector<ll> load_data(const std::string &filename) {
    /* Open file. */
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        exit(EXIT_FAILURE);

    /* Read number of keys. */
    uint64_t n_keys;
    in.read(reinterpret_cast<char*>(&n_keys), sizeof(uint64_t));

    /* Initialize vector. */
    std::vector<ll> data;
    data.resize(n_keys);

    /* Read keys. */
    in.read(reinterpret_cast<char*>(data.data()), n_keys * sizeof(ll));
    in.close();

    return data;
}

int main() {
    vector<ll> vec = load_data("./books_200M_uint64");
    std::ofstream outFile;
    outFile.open("books_200M_uint64.csv", std::ios::out | std::ios::trunc);
    for (int i = 0; i < 100000; i += 1) {
        outFile << vec[i] << std::endl;
    }
    outFile.close();
    return 0;
}
