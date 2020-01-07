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


int graph_coloring(const vector<int> & row_ptr, const vector<int> & col_ind, const vector<pair<int, float>> & ordering) {
 	vector<int> color_arr(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(row_ptr.size()-1, -1);
	bool hasEdge = false;
	for (int i = 0; i<ordering.size(); i++) {
		const int & node = ordering[i].first; //for each node in ordering
		for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++) {
			hasEdge = true;
			const int & adj = col_ind[edge];//for each adjacent node

      for(int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj+1]; edge_2++){
        const int & adj_neigh = col_ind[edge_2];
        if (color_arr[adj_neigh] != -1) { // if it is already colored
  				forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
  			}
      }
		}
		for (int color = 0; color < ordering.size(); color++) { // greedily choose the smallest possible color
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

void clustering_coeff(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	omp_set_num_threads(32);
	#pragma omp parallel for num_threads(32)
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

bool bfstest(vector<int> & lhs, vector<int> & rhs){
	for(int i=0; i<lhs.size(); i++){
		if(lhs[i]!=rhs[i]){
				return false;
		}
	}
	return true;
}


void bfs(int start_node, int num_nodes, vector<int> row_ptr, vector<int> col_ind, vector<int> & distance_arr, int step_size = INT_MAX) {
	vector<int> frontier(num_nodes, -1);
	frontier[0] = start_node;
	int queuestart = 0, queueend=1, frontsize = 0;
	distance_arr.assign(num_nodes, -1); // every node is unvisited
	distance_arr[start_node] = 0; // distance from a node to itself is 0
	int dist = 1; // initial distance
	bool improvement = true;
	while (improvement && dist <= step_size) {
	  improvement = false;
	  do {
	    int front = frontier[queuestart++];
	    for (int edge = row_ptr[front]; edge < row_ptr[front + 1]; edge++) { // for each adjacent of front
	      int adj = col_ind[edge];
	      if (distance_arr[adj] == -1) { // if it is not visited
					improvement = true;
					frontier[queueend + frontsize++] = adj; // place it into next location (new frontier)
					distance_arr[adj] = dist; // assign corresponding distance
	      }
	    }
	  } while (queuestart < queueend);
	  queueend += frontsize; // add the offset
		frontsize = 0; // reset the offset
	  dist++; // next frontier will be further
	}
}


void closeness_centrality_approx(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering, int size = 100) {
	vector<vector<int> > dist_arr(size, vector<int>(num_nodes));
	omp_set_num_threads(32);
#pragma omp parallel for num_threads(32)
	for (int v = 0; v<size; v++) {
	  bfs(v, num_nodes, row_ptr, col_ind, dist_arr[v]); // take distance array for node v
	}
	#pragma omp barrier

	#pragma omp parallel for num_threads(32)
	for(int v=0; v<num_nodes; v++){
		int sum_of_dist = 0;
		for(int i = 0; i<size; i++){
			sum_of_dist += dist_arr[i][v];
		}
		float coeff = sum_of_dist > 0 ? (float)size / sum_of_dist : 0; // if coefficient is negative(meaning that graph is not connected) assign to 0
		ordering[v] = make_pair(v, coeff);
	}
}

void closeness_centrality(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	vector<int> dist_arr;
	for (int v = 0; v<num_nodes; v++) {
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr); // take distance array for node v
		int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
		float coeff = sum_of_dist > 0 ? (float)num_nodes / sum_of_dist : 0; // if coefficient is negative(meaning that graph is not connected) assign to 0
		ordering[v] = make_pair(v, coeff);
	}
}


void degree_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	omp_set_num_threads(32);
	#pragma omp parallel for num_threads(32)
	for (int v = 0; v<num_nodes; v++) {
		ordering[v] = make_pair(v, row_ptr[v + 1] - row_ptr[v]);
	}
}

void degree_2_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	omp_set_num_threads(32);
	#pragma omp parallel for num_threads(32)
	for (int v = 0; v<num_nodes; v++) {
		vector<int> dist_arr(num_nodes);
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr, 2); // take distance array for node v
		ordering[v] = make_pair(v, count(dist_arr.begin(), dist_arr.end(), 2));
	}
}

void degree_3_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
  omp_set_num_threads(32);
#pragma omp parallel for num_threads(32)
  for (int v = 0; v<num_nodes; v++) {
    vector<int> dist_arr(num_nodes);
    bfs(v, num_nodes, row_ptr, col_ind, dist_arr, 3); // take distance array for node v
    ordering[v] = make_pair(v, count(dist_arr.begin(), dist_arr.end(), 3));
  }
}

void page_rank(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering, int iter = 20, float alpha = 0.85) {
	for (int i = 0; i<num_nodes; i++) { ordering[i] = make_pair(i, (float)1 / num_nodes); } // initially likelyhoods are uniformly distributed
	vector<float> offset(num_nodes, 0.0);
	for (int i = 0; i<iter; i++) { // on each iteration (required for convergence)
		vector<pair<int, float> > copy_ordering(num_nodes, pair<int, float>(0, 0.0));
		for (int j = 0; j < num_nodes; j++) {
			copy_ordering[j].first = j;
		}
		for (int v = 0; v<num_nodes; v++) { // for each node v
											// assign total page ranks %85
			float & pr_v = copy_ordering[v].second; // update page rank of v by looking its in-degree nodes
			for (int edge = row_ptr[v]; edge<row_ptr[v + 1]; edge++) { // for each in degree neighbor of v (since graph is symmetric in or out degree does not matter)
				const int & adj = col_ind[edge];
				float & pr_adj = ordering[adj].second;
				int degree_adj = row_ptr[adj + 1] - row_ptr[adj];
				pr_v += pr_adj / degree_adj; // page_rank_of_v <- (page_rank_of_v + page_rank_of_neighbor/out_degree_of_neighbor)
			}
			// distribute %15 evenly
			float dist_value = pr_v * (1 - alpha) / (num_nodes - 1);
			pr_v *= alpha;
			for_each(offset.begin(), offset.end(), [dist_value](float& d) { d += dist_value; });
			pr_v -= dist_value;
		}
		transform(offset.begin(), offset.end(), copy_ordering.begin(), ordering.begin(), add);
		offset.assign(num_nodes, 0);
	}
}

int read_graphs(string & fname, int & num_nodes, int &num_edges, vector<int> & row_ptr, vector<int> & col_ind, vector<pair<vector<int>, vector<int> > > &graphs) {
	ifstream input(fname.c_str());
	if (input.fail()) {
		return -1;
	}
	// read graph
	string line = "%";
	while (line.find("%") != string::npos) {
		getline(input, line);
	}
	istringstream ss(line);
	ss >> num_nodes >> num_nodes >> num_edges;
	int v1, v2;
	double weight;

  vector<int> renameArr(num_nodes, -1);
  int counter = 0;
	bool eliminateUnused = true;

	vector<vector<int> > adj_list(num_nodes);
	for (int i = 0; i<num_edges; i++) {
		getline(input, line);
		istringstream inp(line);
		inp >> v1 >> v2;
		v1--; // make it 0 based
		v2--;

		//for detecting vetices that are unused
    if(renameArr[v1] == -1 && eliminateUnused) {
        renameArr[v1] = counter;
        v1 = counter;
        counter++;
    } else if(eliminateUnused){
        v1 = renameArr[v1];
    }
    if(renameArr[v2] == -1 && eliminateUnused) {
        renameArr[v2] = counter;
        v2 = counter;
        counter++;
    } else if(eliminateUnused) {
        v2 = renameArr[v2];
    }

		if (v1 != v2) {
			adj_list[v1].push_back(v2); // add the edge v1->v2
			adj_list[v2].push_back(v1); // add the edge v2->v1
		}
	}
	if(eliminateUnused) {
		num_nodes = counter;
	}

	row_ptr = vector<int>(num_nodes + 1);
	col_ind = vector<int>(2 * num_edges);
	row_ptr[0] = 0;
	int index = 0;
	for (int v = 0; v<num_nodes; v++) {
		row_ptr[v + 1] = adj_list[v].size(); // assign number of edges going from node v
		for (int i = 0; i<(int)adj_list[v].size(); i++) {
			col_ind[index] = adj_list[v][i]; // put all edges in order wrt row_ptr
			index++;
		}
	}
	for (int v = 1; v<num_nodes + 1; v++) {  // cumulative sum
		row_ptr[v] += row_ptr[v - 1];
	}
	//cout << "nof nodes " << num_nodes << endl;
	//cout << "nof edges " << num_edges << endl;
	pair<vector<int>, vector<int> > graph = make_pair(row_ptr, col_ind);
	graphs.push_back(graph);
	return 0;
}

void normal_params(vector <pair<int, float> > & order, float & mean, float & stdev){
	float sum = 0;
	float sq_sum = 0;
  for(auto & x: order) {
     sum += x.second;
     sq_sum += x.second * x.second;
  }
  mean = sum / order.size();
  stdev = sqrt(sq_sum / order.size() - mean * mean);
}


void normalize(vector<vector <pair<int, float> > > & orders){
	float mean, stdev;
	for(auto & order: orders){
		normal_params(order, mean, stdev);
		if(stdev==0){
			stdev = 1;
		}
		for_each(order.begin(), order.end(), [mean, stdev](pair<int, float> & x){ x.second = (x.second-mean)/stdev;});
	}
}
vector<pair<vector<int>, vector<int> > > random_ugraphs_generator(int graph_cnt, int node_cnt, int edge_cnt) {
	srand (112);
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

int main(int argc, char** argv) {
	const char* path = argv[1];
	DIR *pDIR;
	struct dirent *entry;
	map<string, int> ht;

	vector<string> gnames;
	vector<pair<vector<int>, vector<int> > > graphs;//s = random_ugraphs_generator(10,100,9000);

	// read graphs and store baseline in graphname-color map
	if (pDIR = opendir(path)) {
		while (entry = readdir(pDIR)) {
			string g = entry->d_name;
			string fname = path + ((string)entry->d_name);
			if (fname.at(fname.length() - 1) == '.') {
				continue;
			}
			gnames.push_back(g);

			int num_nodes, num_edges;
			vector<int> row_ptr, col_ind;
			if (read_graphs(fname, num_nodes, num_edges, row_ptr, col_ind, graphs) == -1) {
				cerr << "error reading graph" << endl;
				return 0;
			}
			cout << g << " " << num_nodes << " " << num_edges <<endl;

			//baseline
			vector <pair<int, float> > orders(num_nodes);
			vector<int> color_arr;

			degree_2_order(num_nodes, row_ptr, col_ind, orders);
			sort(orders.begin(), orders.end(), descending);
			int k = graph_coloring(row_ptr, col_ind, orders);
			ht.insert(pair<string,int>(g,k));

    }
		closedir(pDIR);
	}

	vector<float> results(8, 0.0);
	int k= 0;
	for(auto & graph: graphs){
		// initializations
		auto & row_ptr = graph.first;
		auto & col_ind = graph.second;
		int num_nodes = row_ptr.size()-1;
		vector<vector <pair<int, float> > > orders(6, vector<pair<int,float> >(num_nodes));

		// orderings
		degree_order(num_nodes, row_ptr, col_ind, orders[0]);
		degree_2_order(num_nodes, row_ptr, col_ind, orders[1]);
		degree_3_order(num_nodes, row_ptr, col_ind, orders[2]);
		//closeness_centrality(num_nodes, row_ptr, col_ind, orders[3]);
		closeness_centrality_approx(num_nodes, row_ptr, col_ind, orders[3]);
		clustering_coeff(num_nodes, row_ptr, col_ind, orders[4]);
		page_rank(num_nodes, row_ptr, col_ind, orders[5]);

		// normalization
		normalize(orders);

		vector<pair<int, float> > new_order(num_nodes);
		vector<pair<int, float> > uniform_order(num_nodes);
		for(int i=0; i<num_nodes; i++){
			uniform_order[i] = make_pair(i,1/6*orders[0][i].second + 1/6*orders[1][i].second + 1/6*orders[2][i].second + 1/6*orders[3][i].second + 1/6*orders[4][i].second + 1/6*orders[5][i].second);
			new_order[i] = make_pair(i,0.15*orders[0][i].second + 0.1*orders[1][i].second + 0.05*orders[2][i].second + 0.7*orders[3][i].second);
		}

		float d = ht[gnames[k]]; // 2d coloring baseline

		// coloring in that permutation
		sort(new_order.begin(), new_order.end(), descending);
		sort(uniform_order.begin(), uniform_order.end(), descending);
		sort(orders[0].begin(), orders[0].end(), descending);
		sort(orders[1].begin(), orders[1].end(), descending);
		sort(orders[2].begin(), orders[2].end(), descending);
		sort(orders[3].begin(), orders[3].end(), descending);
		sort(orders[4].begin(), orders[4].end(), descending);
		sort(orders[5].begin(), orders[5].end(), descending);

		float totalcolors = graph_coloring(row_ptr, col_ind, new_order);
		results[0] += totalcolors/d;
		cout << gnames[k] << " weighted: " << totalcolors << " degreetwo:" << d << endl;

		totalcolors = graph_coloring(row_ptr, col_ind, uniform_order);
		results[1] += totalcolors/d;
		cout << gnames[k] << " uniform: " << totalcolors << " degreetwo:" << d << endl;

		totalcolors = graph_coloring(row_ptr, col_ind, orders[0]);
		results[2] += totalcolors/d;
		cout << gnames[k] << " degree1: " << totalcolors << " degreetwo:" << d << endl;

		totalcolors = graph_coloring(row_ptr, col_ind, orders[1]);
		results[3] += totalcolors/d;
		cout << gnames[k] << " degree2: " << totalcolors << " degreetwo:" << d << endl;

		totalcolors = graph_coloring(row_ptr, col_ind, orders[2]);
		results[4] += totalcolors/d;
		cout << gnames[k] << " degree3: " << totalcolors << " degreetwo:" << d << endl;

		totalcolors = graph_coloring(row_ptr, col_ind, orders[3]);
		results[5] += totalcolors/d;
		cout << gnames[k] << " closeness: " << totalcolors << " degreetwo:" << d << endl;

		totalcolors = graph_coloring(row_ptr, col_ind, orders[4]);
		results[6] += totalcolors/d;
		cout << gnames[k] << " clustering: " << totalcolors << " degreetwo:" << d << endl;

		totalcolors = graph_coloring(row_ptr, col_ind, orders[5]);
		results[7] += totalcolors/d;
		cout << gnames[k] << " pagerank: " << totalcolors << " degreetwo:" << d << endl;

		k++;
	}

	for(auto & x: results){
		x /= k;
	}

	cout << "weighted order result: " << results[0] << endl;
	cout << "uniform order result: " << results[1] << endl;
	cout << "degree order result: " << results[2] << endl;
	cout << "degree2 order result: " << results[3] << endl;
	cout << "degree3 order result: " << results[4] << endl;
	cout << "closeness order result: " << results[5] << endl;
	cout << "clustering order result: " << results[6] << endl;
	cout << "pagerank order result: " << results[7] << endl;


	return 0;
}
