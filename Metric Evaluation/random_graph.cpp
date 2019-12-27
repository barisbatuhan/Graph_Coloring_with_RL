#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <numeric> // for accumulate
#include <string>
#include<dirent.h>
#include <climits>
#include <cmath>
#include <omp.h>
#include <map>
// for random integer
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

bool descending(const pair<int, float> & left, const pair<int, float> & right) {
	return left.second>right.second;
}
bool ascending(const pair<int, float> & left, const pair<int, float> & right) {
	return left.second<right.second;
}

pair<int, float> add(const float & left, const pair<int, float> & right) {
	return pair<int, float>(right.first, left + right.second);
}

bool isValid(const vector<int> & color_arr, const vector<int>& row_ptr, const vector<int> & col_ind, int num_nodes) {
	for (int v = 0; v < num_nodes; v++) { // for each node v
		for (int e = row_ptr[v]; e < row_ptr[v + 1]; e++) {
			const int & adj = col_ind[e]; // for each adjacent of v
			if (color_arr[adj] == color_arr[v]) { // if color of v equals its adjacent's color
				return false;
			}
		}
	}
	return true;
}

int graph_coloring(const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & ordering, vector<int> & color_arr, int maxdegree) {
	color_arr.resize(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(row_ptr.size()-1, -1);
	bool hasEdge = false;
	for (int i = 0; i<ordering.size(); i++) {
		const int & node = ordering[i].first; //for each node in ordering
		for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++) {
			hasEdge = true;
			const int & adj = col_ind[edge];//for each adjacent node
			if (color_arr[adj] != -1) { // if it is already colored
				forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
			}
      /*for(int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj+1]; edge_2++){
        const int & adj_neigh = col_ind[edge_2];
        if (color_arr[adj_neigh] != -1) { // if it is already colored
  				forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
  			}
      }*/
		}
		for (int color = 0; color < maxdegree; color++) { // greedily choose the smallest possible color
			if (forbid_arr[color] != node) {
				color_arr[node] = color;
				if (nofcolors < color) {
					nofcolors = color;
				}
				break;
			}
		}
	}
	if(hasEdge){
		nofcolors++;
	}
	/*if (!isValid(color_arr, row_ptr, col_ind, ordering.size()) == true) {
		cout << "ERROR" << endl;
	}*/
	return nofcolors;
}

void degree_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	for (int v = 0; v<num_nodes; v++) {
		ordering[v] = make_pair(v, row_ptr[v + 1] - row_ptr[v]);
	}
}

void clustering_coeff(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	for (int v = 0; v<num_nodes; v++) { // for each node
		int degree = row_ptr[v + 1] - row_ptr[v];
		int possiblelinks = degree*(degree - 1) / 2;
		int noflinks = 0;
		unordered_set<int> set;
		for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++) { // insert all of its neighbors
			int adj = col_ind[edge];
			set.insert(adj);
		}
		for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++) {
			const int & adj = col_ind[edge]; // for each neighbor of v
			for (int link = row_ptr[adj]; link < row_ptr[adj + 1]; link++) {
				int adj_neigh = col_ind[link];// for each neighbor of neighbor
				if (set.find(adj_neigh) != set.end()) { // count number of links
					noflinks++;
				}
			}
		}
		float coeff = noflinks != 0 ? (float)possiblelinks / noflinks : possiblelinks+1;
		ordering[v] = make_pair(v, coeff);
	}
}

vector<pair<vector<int>, vector<int>>> random_ugraphs_generator(const int graph_cnt, const int node_cnt, const int edge_cnt) {
	srand (time(NULL));
	vector<pair<vector<int>, vector<int>>> graphs;
	for(int i = 0; i < graph_cnt; i++) {
		vector<vector<bool>> adj_list(node_cnt, vector<bool>(node_cnt, false));
		int edge_num = edge_cnt;
		while(edge_num > 0) {
			int node1 = rand() % node_cnt;
			int node2 = rand() % node_cnt;
			if(node1 == node2) continue;
			if(adj_list[node1][node2] == false) {
				adj_list[node1][node2] = true;
				adj_list[node2][node1] = true;
				edge_num--;
			}
		}
		vector<int> row_ptr(node_cnt + 1);
		vector<int>col_ind(2 * edge_cnt);
		row_ptr[0] = 0;
		int index = 0;
		for (int v = 0; v < node_cnt; v++) {
			int adj_cnt = 0;
			for (int i = 0; i < (int)adj_list[v].size(); i++) {
				if(adj_list[v][i] == true) {
					col_ind[index] = i; // put all edges in order wrt row_ptr
					index++;
					adj_cnt++;
				}
			}
			row_ptr[v + 1] = row_ptr[v] + adj_cnt; // assign number of edges going from node v
		}
		pair<vector<int>, vector<int>> col_row_pair = make_pair(row_ptr, col_ind);
		graphs.push_back(col_row_pair);
	}
	return graphs;
}

int main() {
	int num_graphs = 10;
	int num_nodes = 10000;
	int num_edges = 100000;
	vector<pair<vector<int>, vector<int>>> graphs = random_ugraphs_generator(num_graphs, num_nodes, num_edges);
	for(int i = 0; i < num_graphs; i++) {
		vector <pair<int, float>> orders1(num_nodes);
		degree_order(num_nodes, graphs[i].first, graphs[i].second, orders1);
		sort(orders1.begin(), orders1.end(), descending);
        vector<int> color_arr1;
		int k1 = graph_coloring(graphs[i].first, graphs[i].second, orders1, color_arr1, (int) orders1[0].second);
        
        vector <pair<int, float>> orders2(num_nodes);
        clustering_coeff(num_nodes, graphs[i].first, graphs[i].second, orders2);
        sort(orders2.begin(), orders2.end(), descending);
        vector<int> color_arr2;
		int k2 = graph_coloring(graphs[i].first, graphs[i].second, orders2, color_arr2, (int) orders1[0].second);
		
		cout << "\nRandom Graph No: " << i << " - color result deg1: " << k1 <<  " - color result ClCo: "  << k2 << endl << endl;
	}
	return 0;
}