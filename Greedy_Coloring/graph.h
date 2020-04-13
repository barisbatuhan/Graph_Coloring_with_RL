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

/* HELPER FUNCTIONS FOR PAIR OPERATIONS */
bool descending(const pair<int, float> &left, const pair<int, float> &right);
bool ascending(const pair<int, float> &left, const pair<int, float> &right);
pair<int, float> add(const float &left, const pair<int, float> &right);

/* READ & WRITE OPERATIONS */
void get_filenames(vector<string> &filenames, const vector<string> &locations); // gets <path>/<filenames> from folders specified with locations parameter
int read_graphs(string &fname, int &num_nodes, int &num_edges, vector<int> &row_ptr, vector<int> &col_ind);
string read_family(string &fname);

/* COLORING FUNCTIONS */
int graph_1d_coloring(const vector<int> &row_ptr, const vector<int> &col_ind, const vector<pair<int, float>> &ordering);
int graph_2d_coloring(const vector<int> &row_ptr, const vector<int> &col_ind, const vector<pair<int, float>> &ordering);
int saturation_1d_coloring(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & spare_order);
int saturation_2d_coloring(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & spare_order);

/* COLORING VALIDITY CHECKERS */
bool isValid(const vector<int> &color_arr, const vector<int> &row_ptr, const vector<int> &col_ind, int num_nodes);
bool isValid2d(const vector<int> &color_arr, const vector<int> &row_ptr, const vector<int> &col_ind, int num_nodes);

/* ORDERING ALGORITHMS */
void degree_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void degree_2_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void degree_3_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void closeness_centrality(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void closeness_centrality_approx(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering, int size = 100);
void clustering_coeff(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering);
void page_rank(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering, int iter = 20, float alpha = 0.85);

/* HELPER FUNCTIONS FOR ORDERING ALGORITHMS */
void bfs(int start_node, int num_nodes, vector<int> row_ptr, vector<int> col_ind, vector<int> &distance_arr, int step_size = INT_MAX);
void normal_params(vector<pair<int, float>> &order, float &mean, float &stdev);
void normalize(vector<vector<pair<int, float>>> &orders, int num = -1);

/* DATASET ENLARGING METHODS */
vector<pair<vector<int>, vector<int>>> random_ugraphs_generator(int graph_cnt, int node_cnt, int edge_cnt);

#endif
