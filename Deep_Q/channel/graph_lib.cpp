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
//#include "heap.h"

using namespace std;

class Heap{
public:
    Heap();
    Heap(int cap);
    
    bool isEmpty(){return currentSize==0;}
    bool isFull(){return currentSize == (int)arr.size()-1;}
    pair<int, float> findMax(){return arr[1];}

    void insert(const pair<int,float> & item);
    pair<int, float> deleteMax();
    void makeEmpty();

    void percolateDown(int hole);
    void percolateUp(int hole);
    void buildHeap(vector<pair<int,float> > & vec);

    
    int currentSize;
    vector<pair<int, float> > arr;
    vector<int> locArr;
};

Heap::Heap(){
    currentSize = 0;
}

Heap::Heap(int cap){
    currentSize = 0;
    arr.resize(cap+1);
    for(int i=0; i<cap; i++){
        arr[i].first = i;
        arr[i].second = 0;
    }
    locArr.resize(cap+1,-1);
}

void Heap::insert(const pair<int,float> & item){

    if(isFull()){
        cout << "Heap is full" << endl;
    }
    int hole = ++currentSize;
    arr[hole] = item;
    locArr[item.first] = hole;
    percolateUp(hole);
}

pair<int, float> Heap::deleteMax(){
    
    if(isEmpty()){
        cout << "Heap is empty" << endl;
    }
    auto res = arr[1];
    arr[1] = arr[currentSize--];
    locArr[res.first] = -1;
    locArr[arr[1].first] = 1;
    
    percolateDown(1);
    
    return res;
}

void Heap::makeEmpty(){
    locArr.clear();
    locArr.resize(0);
    arr.clear();
    arr.resize(0);
    currentSize = 0;
}

void Heap::percolateDown(int hole){
    int child;
    auto temp = arr[hole];
    for(; hole*2 <= currentSize; hole = child){
        child = hole*2;
        if(child != currentSize && (arr[child+1].second > arr[child].second || (arr[child+1].second == arr[child].second && arr[child+1].first < arr[child].first))){
            child++;
        }
        if(arr[child].second > temp.second || (arr[child].second == temp.second && arr[child].first < temp.first)){
            arr[hole] = arr[child];
            locArr[arr[child].first] = hole;
        }
        else{
            break;
        }
    }
    arr[hole] = temp;
    locArr[temp.first] = hole;
}

void Heap::percolateUp(int hole){
    auto temp = arr[hole];
    for(; hole>1 && (temp.second > arr[hole/2].second || (temp.second == arr[hole/2].second && temp.first<arr[hole/2].first)); hole/=2){
       
        arr[hole] = arr[hole/2];
        locArr[arr[hole/2].first] = hole;
    }

    arr[hole] = temp;
    locArr[temp.first] = hole;
}

void Heap::buildHeap(vector<pair<int,float> > & vec){
    makeEmpty();
    arr.resize(vec.size()+1);
    locArr = vector<int>(vec.size()+1, -1);
    for(int i=1; i<(int)arr.size(); i++){
        arr[i] = vec[i-1];
        locArr[vec[i-1].first] = i;
    }
    currentSize = vec.size();


    for(int i=currentSize/2; i>0; i--){
        percolateDown(i);
    }
    
}


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
    int num_nodes;
    int num_edges;

private:
    std::string family;
    std::string relative_path;

    // static helper methods
    static void normal_params(std::vector<std::pair<int, float>> &ordering, float &mean, float &stdev);
    static std::pair<int, float> add(const float &left, const std::pair<int, float> &right);
};

// CONSTRUCTORS
Graph::Graph()
{
}

Graph::Graph(std::string fname)
{
    std::ifstream input(fname.c_str());
    if (input.fail())
    {
        throw "No file is found in the current path!";
    }
    else
    {
        relative_path = fname;
    }
    // read graph
    std::string line = "%";
    family = "%";
    while (line.find("%") != std::string::npos)
    {
        getline(input, line);
        if (family == "%" && line.find("kind:") != std::string::npos)
        {
            family = line.substr(8);
            family = family.substr(0, family.length() - 1);
        }
    }

    std::istringstream ss(line);
    ss >> num_nodes >> num_nodes >> num_edges;
    int v1, v2;

    std::vector<int> renameArr(num_nodes, -1);
    int counter = 0;
    bool eliminateUnused = true;

    std::vector<std::vector<int>> adj_list(num_nodes);
    for (int i = 0; i < num_edges; i++)
    {
        getline(input, line);
        std::istringstream inp(line);
        inp >> v1 >> v2;
        v1--; // make it 0 based
        v2--;

        //for detecting vetices that are unused
        if (renameArr[v1] == -1 && eliminateUnused)
        {
            renameArr[v1] = counter;
            v1 = counter;
            counter++;
        }
        else if (eliminateUnused)
        {
            v1 = renameArr[v1];
        }
        if (renameArr[v2] == -1 && eliminateUnused)
        {
            renameArr[v2] = counter;
            v2 = counter;
            counter++;
        }
        else if (eliminateUnused)
        {
            v2 = renameArr[v2];
        }

        if (v1 != v2)
        {
            adj_list[v1].push_back(v2); // add the edge v1->v2
            adj_list[v2].push_back(v1); // add the edge v2->v1
        }
    }
    if (eliminateUnused)
    {
        num_nodes = counter;
    }

    row_ptr = std::vector<int>(num_nodes + 1);
    col_ind = std::vector<int>(2 * num_edges);
    row_ptr[0] = 0;
    int index = 0;
    for (int v = 0; v < num_nodes; v++)
    {
        row_ptr[v + 1] = adj_list[v].size(); // assign number of edges going from node v
        for (int i = 0; i < (int)adj_list[v].size(); i++)
        {
            col_ind[index] = adj_list[v][i]; // put all edges in order wrt row_ptr
            index++;
        }
    }
    for (int v = 1; v < num_nodes + 1; v++)
    { // cumulative sum
        row_ptr[v] += row_ptr[v - 1];
    }
}

Graph::Graph(int node_cnt, int edge_cnt)
{   
    srand(112);
    std::vector<std::vector<bool>> adj_list(node_cnt, std::vector<bool>(node_cnt, false));
    num_nodes = node_cnt;
    num_edges = edge_cnt;
    int edge_num = edge_cnt;
    while (edge_num > 0)
    {
        int node1 = rand() % node_cnt;
        int node2 = rand() % node_cnt;
        if (node1 == node2)
            continue;
        if (adj_list[node1][node2] == false)
        {
            adj_list[node1][node2] = true;
            adj_list[node2][node1] = true;
            edge_num--;
        }
    }
    row_ptr = std::vector<int>(node_cnt + 1);
    col_ind = std::vector<int>(2 * edge_cnt);
    row_ptr[0] = 0;
    int index = 0;
    for (int v = 0; v < node_cnt; v++)
    {
        int adj_cnt = 0;
        for (int i = 0; i < (int)adj_list[v].size(); i++)
        {
            if (adj_list[v][i] == true)
            {
                col_ind[index] = i; // put all edges in order wrt row_ptr
                index++;
                adj_cnt++;
            }
        }
        row_ptr[v + 1] = row_ptr[v] + adj_cnt; // assign number of edges going from node v
    }
}

// COLORING METHODS 
int Graph::color_1d(const std::vector<std::pair<int, float>> &ordering)
{
    std::vector<int> color_arr(ordering.size(), -1);
    int nofcolors = 0;
    std::vector<int> forbid_arr(row_ptr.size() - 1, -1);
    bool hasEdge = false;
    for (int i = 0; i < (int)ordering.size(); i++)
    {
        const int &node = ordering[i].first; //for each node in ordering
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++)
        {
            hasEdge = true;
            const int &adj = col_ind[edge]; //for each adjacent node
            if (color_arr[adj] != -1)
            {                                      // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
        }
        for (int color = 0; color < (int)ordering.size(); color++)
        { // greedily choose the smallest possible color
            if (forbid_arr[color] != node)
            {
                color_arr[node] = color;
                if (nofcolors < color)
                {
                    nofcolors = color;
                }
                break;
            }
        }
    }
    if (hasEdge)
    {
        nofcolors++;
    }
    // if (!is_valid_1d(color_arr) == true)
    // {
    // 	cout << "ERROR" << endl;
    // }
    return nofcolors;
}

int Graph::color_2d(const std::vector<std::pair<int, float>> &ordering)
{
    std::vector<int> color_arr(ordering.size(), -1);
    int nofcolors = 0;
    std::vector<int> forbid_arr(row_ptr.size() - 1, -1);
    bool hasEdge = false;
    for (int i = 0; i < (int)ordering.size(); i++)
    {
        const int &node = ordering[i].first; //for each node in ordering
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++)
        {
            hasEdge = true;
            const int &adj = col_ind[edge]; //for each adjacent node
            if (color_arr[adj] != -1)
            {                                      // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
            for (int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj + 1]; edge_2++)
            {
                const int &adj_neigh = col_ind[edge_2];
                if (color_arr[adj_neigh] != -1)
                {                                            // if it is already colored
                    forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
                }
            }
        }
        for (int color = 0; color < (int)ordering.size(); color++)
        { // greedily choose the smallest possible color
            if (forbid_arr[color] != node)
            {
                color_arr[node] = color;
                if (nofcolors < color)
                {
                    nofcolors = color;
                }
                break;
            }
        }
    }
    if (hasEdge)
    {
        nofcolors++;
    }
    // if (!is_valid_2d(color_arr) == true) {
    // 	cout << "ERROR" << endl;
    // }
    return nofcolors;
}

int Graph::color_dynamic_1d(bool random_start){
    int start=-1;
    if(random_start){
        start = rand()%num_nodes;
    }
    else{ // highest degree1
        int max_degree = -1;
        for(int v=0; v<num_nodes; v++){
            int degree = row_ptr[v+1]-row_ptr[v];
            if(max_degree < degree){
                start = v;
                max_degree = degree;
            }
        }
    }

    Heap orderHeap;
    vector<pair<int, float>> zero(num_nodes, pair<int,float>(0,0));
    for(int i=0; i<(int)zero.size(); i++){
        zero[i].first = i;
    }
    zero[start].second = 1;
    orderHeap.buildHeap(zero);
    int nofcolors = 0, colored = 0;
    vector<int> forbid_arr(num_nodes-1, -1);
    vector<int> color_arr(num_nodes, -1);

    while(colored < num_nodes){
        auto [node, val] = orderHeap.deleteMax(); // structured binding
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++) {
            const int & adj = col_ind[edge]; //for each adjacent node
            int index = orderHeap.locArr[adj];
            if(index != -1){ // if it is not colored
                orderHeap.arr[index].second++;
                orderHeap.percolateUp(index);  
            }
            if (color_arr[adj] != -1) { // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
        }
        for (int color = 0; color < num_nodes; color++) { // greedily choose the smallest possible color
            if (forbid_arr[color] != node) {
                color_arr[node] = color;
                if (nofcolors < color) {
                    nofcolors = color;
                }
                break;
            }
        }
        colored++;
    }
			
    nofcolors++; // coloring was zero based so increment
    if (!is_valid_1d(color_arr)){
        cout << "ERROR" << endl;
    }
    return nofcolors;
}

int Graph::color_dynamic_2d(bool random_start){
    int start=-1;
    if(random_start){
        start = rand()%num_nodes;
    }
    else{ // highest degree1
        int max_degree = -1;
        for(int v=0; v<num_nodes; v++){
            int degree = row_ptr[v+1]-row_ptr[v];
            if(max_degree < degree){
                start = v;
                max_degree = degree;
            }
        }
    }

    Heap orderHeap;
    vector<pair<int, float>> zero(num_nodes, pair<int,float>(0,0));
    for(int i=0; i<(int)zero.size(); i++){
        zero[i].first = i;
    }
    zero[start].second = 1;
    orderHeap.buildHeap(zero);
    int nofcolors = 0, colored = 0;
    vector<int> forbid_arr(num_nodes-1, -1);
    vector<int> color_arr(num_nodes, -1);

    while(colored < num_nodes){
        auto [node, val] = orderHeap.deleteMax(); // structured binding
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++) {
            const int & adj = col_ind[edge]; //for each adjacent node
            int index = orderHeap.locArr[adj];
            if(index != -1){ // if it is not colored
                orderHeap.arr[index].second++;
                orderHeap.percolateUp(index);  
            }
            if (color_arr[adj] != -1) { // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
            for(int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj+1]; edge_2++){
                const int & adj_neigh = col_ind[edge_2];
                index = orderHeap.locArr[adj_neigh];
                if(index != -1){ // if it is not colored
                    orderHeap.arr[index].second ++;
                    orderHeap.percolateUp(index);  
                }
                if (color_arr[adj_neigh] != -1) { // if it is already colored
                    forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
                }
            }
        }
        for (int color = 0; color < num_nodes; color++) { // greedily choose the smallest possible color
            if (forbid_arr[color] != node) {
                color_arr[node] = color;
                if (nofcolors < color) {
                    nofcolors = color;
                }
                break;
            }
        }
        colored++;
    }
			
    nofcolors++; // coloring was zero based so increment
    if (!is_valid_2d(color_arr)){
        cout << "ERROR" << endl;
    }
    return nofcolors;
}

int Graph::color_saturation_1d(std::vector<std::pair<int, float>> &spare_order)
{
    std::vector<std::unordered_set<int>> color_infos(num_nodes);
    std::vector<std::pair<int, int>> node_values(2, {1, 0});
    std::vector<bool> node_colored(num_nodes, false);

    std::vector<int> color_arr(num_nodes, -1);
    int nofcolors = 0;
    bool hasEdge = false;
    int colored_num = 0;

    do
    {
        int i = 0;
        std::pair<int, int> &node = node_values[i];
        while (node_colored[node.first] == true)
        {
            node = node_values[++i];
        }
        std::pair<int, int> &next_node = node_values[i + 1];
        if (next_node.first != -1 && next_node.second == node.second)
        {
            if (node_colored[next_node.first] != true)
            {
                if (spare_order[next_node.first] > spare_order[node.first])
                {
                    node = next_node;
                }
            }
        }

        int color = 0;
        for (; color < num_nodes; color++)
        { // greedily choose the smallest possible color
            if (color_infos[node.first].find(color) == color_infos[node.first].end())
            {
                color_arr[node.first] = color;
                if (nofcolors < color)
                {
                    nofcolors = color;
                }
                break;
            }
        }

        node_colored[node.first] = true;
        colored_num++;

        // set of color of neighbors are arranged
        for (int edge = row_ptr[node.first]; edge < row_ptr[node.first + 1]; edge++)
        {
            hasEdge = true;
            const int &adj = col_ind[edge];
            color_infos[adj].insert(color);
        }

        node_values[0].first = -1;
        node_values[0].second = -9999;
        node_values[1].first = -1;
        node_values[1].second = -9999;

        for (int i = 0; i < num_nodes; i++)
        {
            if (node_colored[i])
            {
                continue;
            }
            int curr_colors = color_infos[i].size();
            if (curr_colors > node_values[0].second)
            {
                node_values[1].second = node_values[0].second;
                node_values[1].first = node_values[0].first;
                node_values[0].second = curr_colors;
                node_values[0].first = i;
            }
            else if (curr_colors > node_values[1].second)
            {
                node_values[1].second = curr_colors;
                node_values[1].first = i;
            }
        }
    } while (colored_num < num_nodes);

    if (hasEdge)
    {
        nofcolors++;
    }
    // if (!is_valid_1d(color_arr) == true)
    // {
    // 	cout << "ERROR" << endl;
    // }
    return nofcolors;
}

int Graph::color_saturation_2d(std::vector<std::pair<int, float>> &spare_order)
{
    std::vector<std::unordered_set<int>> color_infos(num_nodes);
    std::vector<std::pair<int, int>> node_values(2, {1, 0});
    std::vector<bool> node_colored(num_nodes, false);

    std::vector<int> color_arr(num_nodes, -1);
    int nofcolors = 0;
    bool hasEdge = false;
    int colored_num = 0;

    do
    {
        int i = 0;
        std::pair<int, int> &node = node_values[i];
        while (node_colored[node.first] == true)
        {
            node = node_values[++i];
        }
        std::pair<int, int> &next_node = node_values[i + 1];
        if (next_node.first != -1 && next_node.second == node.second)
        {
            if (node_colored[next_node.first] != true)
            {
                if (spare_order[next_node.first] > spare_order[node.first])
                {
                    node = next_node;
                }
            }
        }

        int color = 0;
        for (; color < num_nodes; color++)
        { // greedily choose the smallest possible color
            if (color_infos[node.first].find(color) == color_infos[node.first].end())
            {
                color_arr[node.first] = color;
                if (nofcolors < color)
                {
                    nofcolors = color;
                }
                break;
            }
        }

        node_colored[node.first] = true;
        colored_num++;

        // set of color of neighbors are arranged
        for (int edge = row_ptr[node.first]; edge < row_ptr[node.first + 1]; edge++)
        {
            hasEdge = true;
            const int &adj = col_ind[edge];
            color_infos[adj].insert(color);

            for (int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj + 1]; edge_2++)
            {
                const int &adj_neigh = col_ind[edge_2];
                color_infos[adj_neigh].insert(color);
            }
        }

        node_values[0].first = -1;
        node_values[0].second = -9999;
        node_values[1].first = -1;
        node_values[1].second = -9999;

        for (int i = 0; i < num_nodes; i++)
        {
            if (node_colored[i])
            {
                continue;
            }
            int curr_colors = color_infos[i].size();
            if (curr_colors > node_values[0].second)
            {
                node_values[1].second = node_values[0].second;
                node_values[1].first = node_values[0].first;
                node_values[0].second = curr_colors;
                node_values[0].first = i;
            }
            else if (curr_colors > node_values[1].second)
            {
                node_values[1].second = curr_colors;
                node_values[1].first = i;
            }
        }
    } while (colored_num < num_nodes);

    if (hasEdge)
    {
        nofcolors++;
    }
    // if (!is_valid_2d(color_arr) == true)
    // {
    // 	cout << "ERROR" << endl;
    // }
    return nofcolors;
}

// COLORING VALIDITY CHECKERS 
bool Graph::is_valid_1d(const std::vector<int> &color_arr)
{
    for (int v = 0; v < num_nodes; v++)
    { // for each node v
        if (color_arr[v] == -1)
        {
            return false;
        }
        for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++)
        {
            const int &adj = col_ind[e]; // for each adjacent of v
            if (color_arr[adj] == color_arr[v])
            { // if color of v equals its adjacent's color
                return false;
            }
        }
    }
    return true;
}

bool Graph::is_valid_2d(const std::vector<int> &color_arr)
{
    for (int v = 0; v < num_nodes; v++)
    { // for each node v
        if (color_arr[v] == -1)
        {
            return false;
        }
        for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++)
        {
            const int &adj = col_ind[e]; // for each adjacent of v
            if (color_arr[adj] == color_arr[v])
            { // if color of v equals its adjacent's color
                return false;
            }

            for (int e2 = row_ptr[adj]; e2 < row_ptr[adj + 1]; e2++)
            {
                const int &adj_neigh = col_ind[e2]; // for each adjacent of v
                if (adj_neigh == v)
                    continue;
                if (color_arr[adj_neigh] == color_arr[v])
                { // if color of v equals its adjacent's neighbor's color
                    return false;
                }
            }
        }
    }
    return true;
}

// ORDERING METHODS
void Graph::degree_order(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        ordering[v] = std::make_pair(v, row_ptr[v + 1] - row_ptr[v]);
    }
}

void Graph::degree_order(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        ordering[v] = row_ptr[v + 1] - row_ptr[v];
    }
}

void Graph::degree_2_order(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 2); // take distance array for node v
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2)
            {
                count++;
            }
        }
        ordering[v] = std::make_pair(v, count);
    }
}

void Graph::degree_2_order(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 2); // take distance array for node v
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2)
            {
                count++;
            }
        }
        ordering[v] = count;
    }
}

void Graph::degree_3_order(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 3); // take distance array for node v
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2 || val == 3)
            {
                count++;
            }
        }
        ordering[v] = std::make_pair(v, count);
    }
}

void Graph::degree_3_order(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 3); // take distance array for node v
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2 || val == 3)
            {
                count++;
            }
        }
        ordering[v] = count;
    }
}

void Graph::closeness_centrality(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    std::vector<int> dist_arr;
    for (int v = 0; v < num_nodes; v++)
    {
        bfs(v, dist_arr);                                                       // take distance array for node v
        int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
        float coeff = sum_of_dist > 0 ? (float)num_nodes / sum_of_dist : 0;     // if coefficient is negative(meaning that graph is not connected) assign to 0
        ordering[v] = std::make_pair(v, coeff);
    }
}

void Graph::closeness_centrality(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    std::vector<int> dist_arr;
    for (int v = 0; v < num_nodes; v++)
    {
        bfs(v, dist_arr);                                                       // take distance array for node v
        int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
        float coeff = sum_of_dist > 0 ? (float)num_nodes / sum_of_dist : 0;     // if coefficient is negative(meaning that graph is not connected) assign to 0
        ordering[v] = coeff;
    }
}

void Graph::closeness_centrality_approx(std::vector<std::pair<int, float>> &ordering, int size)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    if (size > num_nodes)
    {
        size = num_nodes;
    }
    std::vector<std::vector<int>> dist_arr(size, std::vector<int>(num_nodes));

    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < size; v++)
    {
        //srand(1);
        int start_node = rand() % num_nodes;
        bfs(start_node, dist_arr[v]); // take distance array for node v
    }
    #pragma omp barrier

    #pragma omp parallel for num_threads(32)
    for (int v = 0; v < num_nodes; v++)
    {
        int sum_of_dist = 0;
        for (int i = 0; i < size; i++)
        {
            sum_of_dist += dist_arr[i][v];
        }
        float coeff = sum_of_dist > 0 ? (float)size / sum_of_dist : 0; // if coefficient is negative(meaning that graph is not connected) assign to 0
        ordering[v] = std::make_pair(v, coeff);
    }
}

void Graph::closeness_centrality_approx(std::vector<float> &ordering, int size)
{
    ordering = std::vector<float>(num_nodes);
    if (size > num_nodes)
    {
        size = num_nodes;
    }
    std::vector<std::vector<int>> dist_arr(size, std::vector<int>(num_nodes));

    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < size; v++)
    {
        //srand(1);
        int start_node = rand() % num_nodes;
        bfs(start_node, dist_arr[v]); // take distance array for node v
    }
    #pragma omp barrier

    #pragma omp parallel for num_threads(32)
    for (int v = 0; v < num_nodes; v++)
    {
        int sum_of_dist = 0;
        for (int i = 0; i < size; i++)
        {
            sum_of_dist += dist_arr[i][v];
        }
        float coeff = sum_of_dist > 0 ? (float)size / sum_of_dist : 0; // if coefficient is negative(meaning that graph is not connected) assign to 0
        ordering[v] = coeff;
    }
}

void Graph::clustering_coeff(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    { // for each node
        int degree = row_ptr[v + 1] - row_ptr[v];
        int possiblelinks = degree * (degree - 1) / 2;
        int noflinks = 0;
        std::unordered_set<int> set;
        for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++)
        { // insert all of its neighbors
            int adj = col_ind[edge];
            set.insert(adj);
        }
        for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++)
        {
            const int &adj = col_ind[edge]; // for each neighbor of v
            for (int link = row_ptr[adj]; link < row_ptr[adj + 1]; link++)
            {
                int adj_neigh = col_ind[link]; // for each neighbor of neighbor
                if (set.find(adj_neigh) != set.end())
                { // count number of links
                    noflinks++;
                }
            }
        }
        float coeff = noflinks != 0 ? (float)possiblelinks / noflinks : possiblelinks + 1;
        ordering[v] = std::make_pair(v, coeff);
    }
}

void Graph::clustering_coeff(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    { // for each node
        int degree = row_ptr[v + 1] - row_ptr[v];
        int possiblelinks = degree * (degree - 1) / 2;
        int noflinks = 0;
        std::unordered_set<int> set;
        for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++)
        { // insert all of its neighbors
            int adj = col_ind[edge];
            set.insert(adj);
        }
        for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++)
        {
            const int &adj = col_ind[edge]; // for each neighbor of v
            for (int link = row_ptr[adj]; link < row_ptr[adj + 1]; link++)
            {
                int adj_neigh = col_ind[link]; // for each neighbor of neighbor
                if (set.find(adj_neigh) != set.end())
                { // count number of links
                    noflinks++;
                }
            }
        }
        float coeff = noflinks != 0 ? (float)possiblelinks / noflinks : possiblelinks + 1;
        ordering[v] = coeff;
    }
}

void Graph::page_rank(std::vector<std::pair<int, float>> &ordering, int iter, float alpha)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    // distribute %15 evenly
    float dist_value = (1 - alpha) / num_nodes;
    for (int i = 0; i < num_nodes; i++)
    {
        ordering[i] = std::make_pair(i, (float)1 / num_nodes);
    }
    // initially likelyhoods are uniformly distributed
    for (int i = 0; i < iter; i++)
    {
        // on each iteration (required for convergence)
        std::vector<std::pair<int, float>> copy_ordering(num_nodes, std::pair<int, float>(0, dist_value));
        for (int j = 0; j < num_nodes; j++)
        {
            copy_ordering[j].first = j;
        }
        for (int v = 0; v < num_nodes; v++)
        { // for each node v
            // assign total page ranks %85
            float &pr_v = copy_ordering[v].second; // update page rank of v by looking its in-degree nodes
            for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++)
            {
                // for each in degree neighbor of v (since graph is symmetric in or out degree does not matter)
                const int &adj = col_ind[edge];
                float &pr_adj = ordering[adj].second;
                int degree_adj = row_ptr[adj + 1] - row_ptr[adj];
                pr_v += alpha * (pr_adj / degree_adj); // page_rank_of_v <- (page_rank_of_v + page_rank_of_neighbor/out_degree_of_neighbor)
            }
        }
        ordering = copy_ordering;
    }
}

void Graph::page_rank(std::vector<float> &ordering, int iter, float alpha)
{
    ordering = std::vector<float>(num_nodes);
    // distribute %15 evenly
    float dist_value = (1 - alpha) / num_nodes;
    for (int i = 0; i < num_nodes; i++)
    {
        ordering[i] = 1 / num_nodes;
    }
    // initially likelyhoods are uniformly distributed
    for (int i = 0; i < iter; i++)
    {
        // on each iteration (required for convergence)
        std::vector<float> copy_ordering(num_nodes, dist_value);
        for (int v = 0; v < num_nodes; v++)
        { // for each node v
            // assign total page ranks %85
            float &pr_v = copy_ordering[v]; // update page rank of v by looking its in-degree nodes
            for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++)
            {
                // for each in degree neighbor of v (since graph is symmetric in or out degree does not matter)
                const int &adj = col_ind[edge];
                float &pr_adj = ordering[adj];
                int degree_adj = row_ptr[adj + 1] - row_ptr[adj];
                pr_v += alpha * (pr_adj / degree_adj); // page_rank_of_v <- (page_rank_of_v + page_rank_of_neighbor/out_degree_of_neighbor)
            }
        }
        ordering = copy_ordering;
    }
}

// HELPER METHODS
void Graph::bfs(int start_node, std::vector<int> &distance_arr, int step_size)
{
    std::vector<int> frontier(num_nodes, -1);
    frontier[0] = start_node;
    int queuestart = 0, queueend = 1, frontsize = 0;
    distance_arr.assign(num_nodes, -1); // every node is unvisited
    distance_arr[start_node] = 0;       // distance from a node to itself is 0
    int dist = 1;                       // initial distance
    bool improvement = true;
    while (improvement && dist <= step_size)
    {
        improvement = false;
        do
        {
            int front = frontier[queuestart++];
            for (int edge = row_ptr[front]; edge < row_ptr[front + 1]; edge++)
            { // for each adjacent of front
                int adj = col_ind[edge];
                if (distance_arr[adj] == -1)
                { // if it is not visited
                    improvement = true;
                    frontier[queueend + frontsize++] = adj; // place it into next location (new frontier)
                    distance_arr[adj] = dist;               // assign corresponding distance
                }
            }
        } while (queuestart < queueend);
        queueend += frontsize; // add the offset
        frontsize = 0;         // reset the offset
        dist++;                // next frontier will be further
    }
}

// PRINTER METHODS 
void Graph::print_graph()
{
    std::cout << "Path: " << relative_path << std::endl
              << "Number of Nodes: " << num_nodes << std::endl
              << "Number of Edges: " << num_edges << std::endl
              << "Graph Family: " << family << std::endl;
}

// STATIC METHODS 
void Graph::normal_params(std::vector<std::pair<int, float>> &order, float &mean, float &stdev)
{
    float sum = 0;
    float sq_sum = 0;
    for (auto &x : order)
    {
        sum += x.second;
        sq_sum += x.second * x.second;
    }
    mean = sum / order.size();
    stdev = sqrt((sq_sum / order.size()) - (mean * mean));
    if (std::isnan(stdev) || stdev == 0)
        stdev = 0.001;
}

void Graph::normalize(std::vector<std::vector<std::pair<int, float>>> &orders, int num)
{
    if (num == -1)
    {
        num = orders.size();
    }
    float mean, stdev;
    for (int i = 0; i < num; i++)
    {
        normal_params(orders[i], mean, stdev);
        for_each(orders[i].begin(), orders[i].end(), [mean, stdev](std::pair<int, float> &x) { x.second = (x.second - mean) / stdev; });
    }
}

bool Graph::descending(const std::pair<int, float> &left, const std::pair<int, float> &right)
{
	if ((left.second > right.second) || (left.second == right.second && left.first < right.first))
	{
		return true;
	}
	return false;
}

bool Graph::ascending(const std::pair<int, float> &left, const std::pair<int, float> &right)
{
	if ((left.second < right.second) || (left.second == right.second && left.first < right.first))
	{
		return true;
	}
	return false;
}

std::pair<int, float> Graph::add(const float &left, const std::pair<int, float> &right)
{
	return std::pair<int, float>(right.first, left + right.second);
}

/* ---------------------------------------------
COMMUNICATING FUNCTIONS
--------------------------------------------- */

std::vector<Graph> graphs;

Graph curr_graph;
float** node_embed;
float* graph_embed;

// helper methods
extern "C" void normalize(vector<float> & g_s);
extern "C" void transpose(vector<vector<float>> & node_embeddings);
extern "C" vector<float> concatenate(vector<float> & first, vector<float> & second);

// initialization
extern "C" void initialize_node_embeddings(Graph & g, vector<vector<float>> & node_embeddings);
extern "C" void initialize_graph_state(Graph & g, vector<float> & g_s);

// update
extern "C" void update_node_embeddings(Graph & g, vector<vector<float>> & node_embeddings, int latest_colored_node, int latest_color);
extern "C" void update_graph_state(Graph & g, vector<float> & graph_state, unordered_set<int> & sol_set);

// pseudo normalization
void normalize(vector<float> & g_s) {
	double sq_sum = 0;
	for (auto x : g_s)
		sq_sum += x*x;
	double norm = sqrt(sq_sum); // norm of the vec
	for (auto & x : g_s)
		x /= norm;
}

void transpose(vector<vector<float>> & node_embeddings){
    int rows = (int)node_embeddings.size();
    int cols = (int)node_embeddings[0].size();
    vector<vector<float>> res(cols, vector<float>(rows));

    for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			res[j][i] = node_embeddings[i][j];
		}
	}
    node_embeddings = res;
}

vector<float> concatenate(vector<float> & first, vector<float> & second) {
	vector<float> res = first;
	for (int i = 0; i < (int)second.size(); i++) {
		res.push_back(second[i]);
	}
	return res;
}

extern "C" float** init_node_embeddings() {
    int size = curr_graph.num_nodes;
    vector<vector<float>> node_embeddings = vector<vector<float>>(9, vector<float>(size, 0));
    curr_graph.degree_order(node_embeddings[0]);
    curr_graph.degree_2_order(node_embeddings[1]);
    curr_graph.degree_3_order(node_embeddings[2]);
    curr_graph.closeness_centrality_approx(node_embeddings[3]);
    curr_graph.clustering_coeff(node_embeddings[4]);
    curr_graph.page_rank(node_embeddings[5]);
    // dynamic coeffs
    // 6th index -> nof colored neighbors
    // 7th index -> nof different colored neighbors
    // 8th index -> is node itself colored
    // dynamic coefficients are automatically 0 in the beginning
    // transpose(node_embeds);
    node_embed = new float*[9];
    for(int i = 0; i < 9; i++) {
        node_embed[i] = new float[size];
        for(int j = 0; j < size; j++) {
            node_embed[i][j] = node_embeddings[i][j];
        }
    }
    return node_embed;
}

extern "C" float* init_graph_embeddings() {
    auto & row_ptr = curr_graph.row_ptr;
    int size = curr_graph.num_nodes;
    vector<float> closeness;
    curr_graph.closeness_centrality_approx(closeness);
	graph_embed = new float[16];

	graph_embed[0] = size;                                              // num nodes
	graph_embed[1] = curr_graph.num_edges;                                       // num edges
	graph_embed[2] = 0;                                                 // nof adjacents of colored nodes
	graph_embed[3] = 0;                                                 // nof colored adjacents of colored nodes
	graph_embed[4] = 0;                                                 // closeness sum of colored nodes
	graph_embed[5] = 0;                                                 // nof colored nodes
    for(int i=0; i<(int)closeness.size(); i++){                      // closeness sum of uncolored nodes
        graph_embed[6] += closeness[i];
    }
	graph_embed[7] = size;                                              // nof uncolored nodes
	graph_embed[8] = 0;                                                 // degree above mean (among colored nodes)
	graph_embed[9] = 0;                                                 // degree below mean (among colored nodes)
	graph_embed[10] = 0;                                                // degree above mean (among uncolored nodes)
	graph_embed[11] = 0;                                                // degree below mean (among uncolored nodes)
	double degree_mean = (double)row_ptr.back() / size;
	double degree_sq_sum = 0;
	for (int v = 0; v < (int)row_ptr.size() - 1; v++) {
		int degree = row_ptr[v + 1] - row_ptr[v];
		if (degree > degree_mean) {
			graph_embed[10]++;
		}
		else {
			graph_embed[11]++;
		}
		degree_sq_sum += degree * degree;
	}
	double variance = degree_sq_sum / size - (degree_mean*degree_mean);
	graph_embed[12] = 0;                                                // mean degree of colored nodes
	graph_embed[13] = degree_mean;                                      // mean degree of uncolored nodes
	graph_embed[14] = 0;                                                // variance of degrees (colored nodes)
	graph_embed[15] = variance;                                         // variance of degrees (uncolored nodes)
	//normalize(g_s);
    return graph_embed;
}

// smallest distance to colored/uncolored node is missing
void initialize_node_embeddings(Graph & g, vector<vector<float>> & node_embeddings){
    int size = g.num_nodes;
	
    node_embeddings = vector<vector<float>>(9, vector<float>(size, 0));

    g.degree_order(node_embeddings[0]);
    g.degree_2_order(node_embeddings[1]);
    g.degree_3_order(node_embeddings[2]);
    g.closeness_centrality_approx(node_embeddings[3]);
    g.clustering_coeff(node_embeddings[4]);
    g.page_rank(node_embeddings[5]);
    // dynamic coeffs
    // 6th index -> nof colored neighbors
    // 7th index -> nof different colored neighbors
    // 8th index -> is node itself colored
    // dynamic coefficients are automatically 0 in the beginning

    transpose(node_embeddings);
}

void initialize_graph_state(Graph & g, vector<float> & g_s) {
	auto & row_ptr = g.row_ptr;
    int size = g.num_nodes;
    vector<float> closeness;
    g.closeness_centrality_approx(closeness);
	g_s.resize(16);

	g_s[0] = size;                                              // num nodes
	g_s[1] = g.num_edges;                                       // num edges
	g_s[2] = 0;                                                 // nof adjacents of colored nodes
	g_s[3] = 0;                                                 // nof colored adjacents of colored nodes
	g_s[4] = 0;                                                 // closeness sum of colored nodes
	g_s[5] = 0;                                                 // nof colored nodes
    for(int i=0; i<(int)closeness.size(); i++){                      // closeness sum of uncolored nodes
        g_s[6] += closeness[i];
    }
	g_s[7] = size;                                              // nof uncolored nodes
	g_s[8] = 0;                                                 // degree above mean (among colored nodes)
	g_s[9] = 0;                                                 // degree below mean (among colored nodes)
	g_s[10] = 0;                                                // degree above mean (among uncolored nodes)
	g_s[11] = 0;                                                // degree below mean (among uncolored nodes)
	double degree_mean = (double)row_ptr.back() / size;
	double degree_sq_sum = 0;
	for (int v = 0; v < (int)row_ptr.size() - 1; v++) {
		int degree = row_ptr[v + 1] - row_ptr[v];
		if (degree > degree_mean) {
			g_s[10]++;
		}
		else {
			g_s[11]++;
		}
		degree_sq_sum += degree * degree;
	}
	double variance = degree_sq_sum / size - (degree_mean*degree_mean);
	g_s[12] = 0;                                                // mean degree of colored nodes
	g_s[13] = degree_mean;                                      // mean degree of uncolored nodes
	g_s[14] = 0;                                                // variance of degrees (colored nodes)
	g_s[15] = variance;                                         // variance of degrees (uncolored nodes)
	//normalize(g_s);
}

// immediate update (not expensive)
void update_node_embeddings(Graph & g, vector<vector<float>> & node_embeddings, int latest_colored_node, int latest_color){
    auto & row_ptr = g.row_ptr;
    auto & col_ind = g.col_ind; 
	for(int edge = row_ptr[latest_colored_node]; edge < row_ptr[latest_colored_node+1]; edge++){
        int adj = col_ind[edge];
        node_embeddings[adj][6]++;
        if(latest_color > node_embeddings[adj][7]){
            node_embeddings[adj][7] = latest_color;
        }
    }
    node_embeddings[latest_colored_node][8] = 1;
}

// diameter is missing (we need two bfs to calculate it)
void update_graph_state(Graph & g, vector<float> & graph_state, unordered_set<int> & sol_set) {
	auto & row_ptr = g.row_ptr;
	auto & col_ind = g.col_ind;
    int size = g.num_nodes;
    vector<float> closeness;
    g.closeness_centrality_approx(closeness);

	double degree_mean = (double)row_ptr.back() / size;
	double colored_degree_sum = 0;
	double colored_degree_sq_sum = 0;
	double uncolored_degree_sum = 0;
	double uncolored_degree_sq_sum = 0;
	graph_state[0] = size;                                              // num nodes
	graph_state[1] = g.num_edges;                                       // num edges
	graph_state[2] = 0;                                                 // nof adjacents of colored nodes
	graph_state[3] = 0;                                                 // nof colored adjacents of colored nodes
	graph_state[4] = 0;                                                 // closeness sum of colored nodes
	graph_state[6] = accumulate(closeness.begin(), closeness.end(), 0); // closeness sum of uncolored nodes
	graph_state[8] = 0;                                                 // degree above mean (among colored nodes)
	graph_state[9] = 0;                                                 // degree below mean (among colored nodes)
	graph_state[10] = 0;                                                // degree above mean (among uncolored nodes)
	graph_state[11] = 0;                                                // degree below mean (among uncolored nodes)
	for (int v = 0; v < size; v++) {
		int degree = row_ptr[v + 1] - row_ptr[v];
		if (sol_set.find(v) != sol_set.end()) {
			colored_degree_sum += degree;
			colored_degree_sq_sum += degree*degree;
			graph_state[2] += degree;
			if (degree > degree_mean) {
				graph_state[8] ++;
				graph_state[10] --;
			}
			else {
				graph_state[9] ++;
				graph_state[11] --;
			}

			for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++) {
				int adj = col_ind[edge];
				if (sol_set.find(adj) != sol_set.end()) {
					graph_state[3]++;
				}
			}
			graph_state[4] += closeness[v];
			graph_state[6] -= closeness[v];
		}
		else {
			uncolored_degree_sum += degree;
			uncolored_degree_sq_sum += degree*degree;
		}
	}
	graph_state[5] = sol_set.size();
	graph_state[7] = size - sol_set.size();
	graph_state[12] = colored_degree_sum / (sol_set.size() + 1);
	graph_state[13] = uncolored_degree_sum / (size - sol_set.size());
	graph_state[14] = colored_degree_sq_sum / sol_set.size() - (graph_state[12] * graph_state[12]);
	graph_state[15] = uncolored_degree_sq_sum / (size - sol_set.size()) - (graph_state[13] * graph_state[13]);
	normalize(graph_state);
}


extern "C" void insert_graph(int n, int e){
    Graph g(n, e);
    graphs.push_back(g);
}

extern "C" int read_graph(char* graph_name){
    Graph g(graph_name);
    curr_graph = g;
    graphs.push_back(g);
    return g.num_nodes;
}

extern "C" void print_graph_features(){
    for(int i=0; i<(int)graphs.size(); i++){
        cout << "nof nodes " << graphs[i].num_nodes << " nof edges " << graphs[i].num_edges << endl;  
    }
}

extern "C" float** initialize_graph_embeddings_for_batch(int * rows, int * cols){ 
    
    *rows = graphs.size();

    float** res = new float*[graphs.size()];
    for(int i=0; i<(int)graphs.size(); i++){
        vector<float> vec;
        initialize_graph_state(graphs[i], vec);
        res[i] = new float[vec.size()];
        for(int j=0; j < (int)vec.size(); j++){
            res[i][j] = vec[j];
        }
        *cols = vec.size();
    }

    /*
    for(int i=0; i<*rows; i++){
        for(int j=0; j < *cols; j++){
            cout << res[i][j] << " ";
        }
        cout << endl;
    }
    */
    return res;
}

#endif
