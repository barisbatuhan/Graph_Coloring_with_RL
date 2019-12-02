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

int graph_coloring(const vector<int> & row_ptr, const vector<int> & col_ind, const vector<pair<int, float> > & ordering, vector<int> & color_arr, int maxdegree, string type) {
	color_arr.resize(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(maxdegree + 1, -1);
	bool hasEdge = false;
	for (int i = 0; i<ordering.size(); i++) {
		const int & node = ordering[i].first; //for each node in ordering
		for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++) {
			hasEdge = true;
			const int & adj = col_ind[edge];//for each adjacent node
			if (color_arr[adj] != -1) { // if it is already colored
				forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
			}
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
	//cout << "with " << type << " nof colors: " << nofcolors << endl;
	if (!isValid(color_arr, row_ptr, col_ind, ordering.size()) == true) {
		//cout << "row ptr size " << row_ptr.size() << " colind size" << col_ind.size() << " ordering size " << ordering.size() << endl;
		cout << "ERROR" << endl;
		for(auto & x: color_arr){
			cout << x << " ";
		}
		cout << endl;
	}
	return nofcolors;
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

void bfs(int start_node, int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<int> & distance_arr, int step_size = INT_MAX) {
	distance_arr.assign(num_nodes, -1); // every node is unvisited
	distance_arr[start_node] = 0; // distance from a node to itself is 0
	queue<int> frontier; // FIFO queue
	frontier.push(start_node);
	int dist = 1; // initial distance
	bool improvement = true;
	while (improvement && dist <= step_size) {
		improvement = false;
		queue<int> new_frontier; // FIFO queue
		do {
			int & front = frontier.front();
			frontier.pop();
			for (int edge = row_ptr[front]; edge < row_ptr[front + 1]; edge++) { // for each adjacent of front
				const int & adj = col_ind[edge];
				if (distance_arr[adj] == -1) { // if it is not visited
					improvement = true;
					distance_arr[adj] = dist; // assign corresponding distance
					new_frontier.push(adj); // add it to the frontier
				}
			}
		} while (!frontier.empty());
		frontier = new_frontier;
		dist++; // next frontier will be further
	}
	/*
	for(auto & x: distance_arr){
	cout << x << endl;
	}
	*/
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
	for (int v = 0; v<num_nodes; v++) {
		ordering[v] = make_pair(v, row_ptr[v + 1] - row_ptr[v]);
	}
}

void degree_2_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	vector<int> dist_arr;
	for (int v = 0; v<num_nodes; v++) {
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr, 2); // take distance array for node v
		ordering[v] = make_pair(v, count(dist_arr.begin(), dist_arr.end(), 2));
	}
}

void degree_3_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering) {
	vector<int> dist_arr;
	for (int v = 0; v<num_nodes; v++) {
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr, 3); // take distance array for node v
		ordering[v] = make_pair(v, count(dist_arr.begin(), dist_arr.end(), 3));
	}
}

void page_rank(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float> > & ordering, int iter = 100, float alpha = 0.85) {
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
	cout << "nof nodes " << num_nodes << endl;
	cout << "nof edges " << num_edges << endl;
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


int main(int argc, char** argv) {

	const char* path = argv[1];
	DIR *pDIR;
	struct dirent *entry;
	vector<pair<vector<int>, vector<int> > > graphs;
	if (pDIR = opendir(path)) {
		while (entry = readdir(pDIR)) {
			cout << entry->d_name << endl;
			string fname = path + ((string)entry->d_name);
			if (fname.at(fname.length() - 1) == '.') {
				continue;
			}
			cout << fname << endl;
			int num_nodes, num_edges;
			vector<int> row_ptr, col_ind;
			if (read_graphs(fname, num_nodes, num_edges, row_ptr, col_ind, graphs) == -1) {
				cerr << "error reading graph" << endl;
				return 0;
			}
    }
		closedir(pDIR);
	}
	float sum;
	int train_size = graphs.size()*8/10;

	bool flag = true;
	omp_set_num_threads(32);
	vector<vector<float> > coefficients(32,vector<float>(6));
	vector<int> min_colors(32, INT_MAX);
	vector<float> coef_list(21);
	for(int i=0; i<=20; i++){coef_list[i] = i*0.05;}
	#pragma omp parallel for collapse(6)
	for(int i1 = 0; i1 < 21; i1 ++){
		for(int i2 = 0; i2 < 21; i2++){
			for(int i3 = 0; i3 < 21; i3++){
				for(int i4 = 0; i4 < 21; i4 ++){
					for(int i5 = 0; i5 < 21; i5++){
						for(int i6 = 0; i6 < 21; i6++){
								float c1,c2,c3,c4,c5,c6;
								c1 = coef_list[i1];
								c2 = coef_list[i2];
								c3 = coef_list[i3];
								c4 = coef_list[i4];
								c5 = coef_list[i5];
								c6 = coef_list[i6];
								sum = c1+c2+c3+c4+c5+c6;
								if((sum== 1) && flag){
									printf("%f %f %f %f %f %f\n", c1 ,c2 ,c3, c4, c5, c6);
									int total_colors = 0;
									int tid = omp_get_thread_num();
									for(auto & graph: graphs){
										auto & row_ptr = graph.first;
										auto & col_ind = graph.second;
										int num_nodes = row_ptr.size()-1;
										vector<vector <pair<int, float> > > orders(6, vector<pair<int,float> >(num_nodes));
										degree_order(num_nodes, row_ptr, col_ind, orders[0]);
										int maxdegree = INT_MIN;
										for(auto & x: orders[0]){
											if(x.second>maxdegree){
												maxdegree = x.second;
											}
										}
										degree_2_order(num_nodes, row_ptr, col_ind, orders[1]);
										degree_3_order(num_nodes, row_ptr, col_ind, orders[2]);
										closeness_centrality(num_nodes, row_ptr, col_ind, orders[3]);
										clustering_coeff(num_nodes, row_ptr, col_ind, orders[4]);
										page_rank(num_nodes, row_ptr, col_ind, orders[5]);

										normalize(orders);

										vector<int> color_arr;
										vector<pair<int, float> > new_order(num_nodes);
										for(int i=0; i<num_nodes; i++){
											new_order[i] = make_pair(i,c1*orders[2][i].second + c2*orders[1][i].second + c3*orders[0][i].second + c4*orders[3][i].second + c5*orders[4][i].second + c6*orders[5][i].second);
										}

										sort(new_order.begin(), new_order.end(), descending);
										total_colors += graph_coloring(row_ptr, col_ind, new_order, color_arr, maxdegree, "selam");

									}

									if(total_colors < min_colors[tid]){
										coefficients[tid][0] = c1;
										coefficients[tid][1] = c2;
										coefficients[tid][2] = c3;
										coefficients[tid][3] = c4;
										coefficients[tid][4] = c5;
										coefficients[tid][5] = c6;
										printf("Total colors : %d\n", total_colors);
										min_colors[tid] = total_colors;
									}
								}
						}
					}
				}
			}
		}
	}
	int mincolors = INT_MAX;
	int tid =-1;
	for(int i=0; i<min_colors.size(); i++){
		if(min_colors[i]<mincolors){
			mincolors = min_colors[i];
			tid = i;
		}
	}
	cout << "best result is " << mincolors << endl;
	cout << "Coefficients are ";
	for(auto & x: coefficients[tid]){
		cout << x << " ";
	}
	cout << endl;

	return 0;
}
