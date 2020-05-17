/**
 * Undirected and unweighted graph class implementation along with vertex ordering heuristics.
 * Detailed explanation of each method can be viewed in graph.cpp folder
 * */

#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <numeric> // for accumulate
#include <string>
#include <dirent.h>
#include <climits>
#include <cmath>
#include <omp.h>
#include <map>
#include <vector>
#include <iostream>
#include "heap.h"

using namespace std;

class Graph
{
public:
    // constructor
    Graph();
    Graph(std::string fname);
    Graph(int node_cnt, int edge_cnt); // random graph generation


    // coloring methods
    int color_1d(const std::vector<std::pair<int, float>> &ordering);
    int color_2d(const std::vector<std::pair<int, float>> &ordering);
    int color_dynamic_1d(bool random_start = false);
    int color_dynamic_2d(bool random_start = false);
    int color_saturation_1d(std::vector<std::pair<int, float>> &spare_order);
    int color_saturation_2d(std::vector<std::pair<int, float>> &spare_order);


    // coloring validity checkers
    bool is_valid_1d(const std::vector<int> &color_arr);
    bool is_valid_2d(const std::vector<int> &color_arr);


    // ordering methods
    void degree_order(std::vector<std::pair<int, float>> &ordering);
    void degree_order(std::vector<float> &ordering);

    void degree_2_order(std::vector<std::pair<int, float>> &ordering);
    void degree_2_order(std::vector<float> &ordering);

    void degree_3_order(std::vector<std::pair<int, float>> &ordering);
    void degree_3_order(std::vector<float> &ordering);

    void closeness_centrality(std::vector<std::pair<int, float>> &ordering);
    void closeness_centrality(std::vector<float> &ordering);

    void closeness_centrality_approx(std::vector<std::pair<int, float>> &ordering, int size = 100);
    void closeness_centrality_approx(std::vector<float> &ordering, int size = 100);

    void clustering_coeff(std::vector<std::pair<int, float>> &ordering);
    void clustering_coeff(std::vector<float> &ordering);

    void page_rank(std::vector<std::pair<int, float>> &ordering, int iter = 20, float alpha = 0.85);
    void page_rank(std::vector<float> &ordering, int iter = 20, float alpha = 0.85);

    // helper methods
    void bfs(int start_node, std::vector<int> &distance_arr, int step_size = INT_MAX);

    // printing graph
    void print_graph();

    // static methods
    static void normalize(std::vector<std::vector<std::pair<int, float>>> &orders, int num = -1);
    static bool descending(const std::pair<int, float> &left, const std::pair<int, float> &right);
    static bool ascending(const std::pair<int, float> &left, const std::pair<int, float> &right);

    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::string relative_path;
    int num_nodes;
    int num_edges;
    int max_degree;

private:
    std::string family;

    // static helper methods
    static void normal_params(std::vector<std::pair<int, float>> &ordering, float &mean, float &stdev);
    static std::pair<int, float> add(const float &left, const std::pair<int, float> &right);
};

#endif
