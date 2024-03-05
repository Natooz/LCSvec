#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <string>
#include <iostream>
#include <string.h>
#include <sstream>
#include <algorithm>
using namespace std;

// TODO Remove when not necessary anymore
int add(int a, int b) { return a + b; }


vector<string> utf8_split(const string &str){
    vector<string> split;
    int len = str.length();
    int left = 0;
    int right = 1;

    for (int i = 0; i < len; i++){
        if (right >= len || ((str[right] & 0xc0) != 0x80)){
            string s = str.substr(left, right - left);
            split.push_back(s);
            // printf("%s %d %d\n", s.c_str(), left, right);
            left = right;
        }
        right ++;
    }
    return split;
}


// 最长公共子序列（不连续）
int lcs_sequence_length(const string &str1, const string &str2) {
    if (str1 == "" || str2 == "")
        return 0;
    vector<string> s1 = utf8_split(str1);
    vector<string> s2 = utf8_split(str2);
    int m = s1.size();
    int n = s2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    int i, j;
    // printf("%d %d\n", m, n);

    for (i = 0; i <= m; i++) {
        dp[i][0] = 0;
    }
    for (j = 0; j <= n; j++) {
        dp[0][j] = 0;
    }
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                if (dp[i - 1][j] >= dp[i][j - 1])
                    dp[i][j] = dp[i - 1][j];
                else
                    dp[i][j] = dp[i][j-1];
            }
        }
    }
    return dp[m][n];
}


vector<int> lcs_sequence_idx(const string &str, const string &ref) {
    vector<string> s1 = utf8_split(str);
    vector<string> s2 = utf8_split(ref);
    int m = s1.size();
    int n = s2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    vector<vector<char>> direct(m + 1, vector<char>(n + 1));
    vector<int> res(m, -1);
    if (m == 0 || n == 0)
        return res;

    int i, j;
    for (i = 0; i <= m; i++) dp[i][0] = 0;
    for (j = 0; j <= n; j++) dp[0][j] = 0;
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                direct[i][j] = 'm';     // match
            } else {
                if (dp[i - 1][j] >= dp[i][j - 1]) {
                    dp[i][j] = dp[i - 1][j];
                    direct[i][j] = 's';     // str+1
                }
                else {
                    dp[i][j] = dp[i][j-1];
                    direct[i][j] = 'r';     // ref+1
                }
            }
        }
    }
    for (i = m, j = n; i > 0 && j > 0; ){
        if (direct[i][j] == 'm') {
            res[i-1] = j-1;
            i--; j--;
        }
        else if (direct[i][j] == 's') i--;
        else if (direct[i][j] == 'r') j--;
    }
    return res;
}


// 最长公共子串（连续）
int lcs_string_length(const string &str1, const string &str2) {
    if (str1 == "" || str2 == "")
        return 0;
    vector<string> s1 = utf8_split(str1);
    vector<string> s2 = utf8_split(str2);
    int m = s1.size();
    int n = s2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    int i, j;
    int max = 0;

    for (i = 0; i <= m; i++) {
        dp[i][0] = 0;
    }
    for (j = 0; j <= n; j++) {
        dp[0][j] = 0;
    }
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > max){
                    max = dp[i][j];
                }
            }
            else {
                dp[i][j] = 0;
            }
        }
    }
    return max;
}


vector<int> lcs_string_idx(const string &str, const string &ref) {
    vector<string> s1 = utf8_split(str);
    vector<string> s2 = utf8_split(ref);
    int m = s1.size();
    int n = s2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1));
    vector<int> res(m, -1);
    if (m == 0 || n == 0)
        return res;

    int i, j;
    int max_i = 0, max_j = 0;
    for (i = 0; i <= m; i++) {
        dp[i][0] = 0;
    }
    for (j = 0; j <= n; j++) {
        dp[0][j] = 0;
    }
    for (i = 1; i <= m; i++) {
        for (j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                if (dp[i][j] > dp[max_i][max_j]){
                    max_i = i; max_j = j;
                }
            }
            else {
                dp[i][j] = 0;
            }
        }
    }
    for (i = 0; i < dp[max_i][max_j]; i++) {
        res[max_i-i-1] = max_j-i-1;
    }
    return res;
}


vector<int> lcs_sequence_of_list(const string &str1, vector<string> &str_list){
    int size = str_list.size();
    vector<int> ls(size);
    for (int i = 0; i < size; i++){
        int l = lcs_sequence_length(str1, str_list[i]);
        ls[i] = l;
    }
    return ls;
}


vector<int> lcs_string_of_list(const string &str1, vector<string> &str_list){
    int size = str_list.size();
    vector<int> ls(size);
    for (int i = 0; i < size; i++){
        int l = lcs_string_length(str1, str_list[i]);
        ls[i] = l;
    }
    return ls;
}


namespace py = nanobind;

NB_MODULE(lctorch, m) {
    m.def("add", &add);

    m.def("lcs_sequence_length", &lcs_sequence_length, R"pbdoc(Longest common subsequence)pbdoc");
    m.def("lcs_sequence_idx", &lcs_sequence_idx, R"pbdoc(Longest common subsequence indices mapping from str to ref)pbdoc",
        py::arg("s"), py::arg("ref"));
    m.def("lcs_sequence_of_list", &lcs_sequence_of_list, R"pbdoc(Longest common subsequence of list)pbdoc");

    m.def("lcs_string_length", &lcs_string_length, R"pbdoc(Longest common substring)pbdoc");
    m.def("lcs_string_idx", &lcs_string_idx, R"pbdoc(Longest common substring indices mapping from str to ref)pbdoc",
        py::arg("s"), py::arg("ref"));
    m.def("lcs_string_of_list", &lcs_string_of_list, R"pbdoc(Longest common substring of list)pbdoc");
}