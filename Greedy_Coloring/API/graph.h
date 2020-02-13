#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <numeric> // for accumulate
#include <string>
#include <dirent.h>
#include <climits>
#include <cmath>
#include <omp.h>
#include <map>
#include <iostream>

using namespace std;

bool descending(const pair<int, float> &left, const pair<int, float> &right);
bool ascending(const pair<int, float> &left, const pair<int, float> &right);
pair<int, float> add(const float &left, const pair<int, float> &right);
bool isValid(const vector<int> &color_arr, const vector<int> &row_ptr, const vector<int> &col_ind, int num_nodes);
int graph_1d_coloring(const vector<int> &row_ptr, const vector<int> &col_ind, const vector<pair<int, float>> &ordering);
int graph_2d_coloring(const vector<int> &row_ptr, const vector<int> &col_ind, const vector<pair<int, float>> &ordering);
void clustering_coeff(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
bool bfstest(vector<int> &lhs, vector<int> &rhs);
void bfs(int start_node, int num_nodes, vector<int> row_ptr, vector<int> col_ind, vector<int> &distance_arr, int step_size = INT_MAX);
void closeness_centrality_approx(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering, int size = 100);
void closeness_centrality(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void degree_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void degree_2_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void degree_3_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void page_rank(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering, int iter = 20, float alpha = 0.85);
int read_graphs(string &fname, int &num_nodes, int &num_edges, vector<int> &row_ptr, vector<int> &col_ind);
void write_to_csv(vector<string> labels, vector<vector<pair<int, float>>> &data, string filename, string path = "./");
void normal_params(vector<pair<int, float>> &order, float &mean, float &stdev);
void normalize(vector<vector<pair<int, float>>> &orders, int num = -1);
vector<pair<vector<int>, vector<int>>> random_ugraphs_generator(int graph_cnt, int node_cnt, int edge_cnt);

#endif
