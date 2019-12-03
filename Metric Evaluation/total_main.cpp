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
	if (!isValid(color_arr, row_ptr, col_ind, ordering.size()) == true) {
		cout << "ERROR" << endl;
	}
	return nofcolors;
}

void clustering_coeff(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & ordering) {
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

void closeness_centrality(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & ordering) {
	vector<int> dist_arr;
	for (int v = 0; v<num_nodes; v++) {
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr); // take distance array for node v
		int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
		float coeff = sum_of_dist > 0 ? (float)num_nodes / sum_of_dist : 0; // if coefficient is negative(meaning that graph is not connected) assign to 0
		ordering[v] = make_pair(v, coeff);
	}
}

void degree_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & ordering) {
	for (int v = 0; v<num_nodes; v++) {
		ordering[v] = make_pair(v, row_ptr[v + 1] - row_ptr[v]);
	}
}

void degree_2_order(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & ordering) {
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

void page_rank(int num_nodes, const vector<int> & row_ptr, const vector<int> & col_ind, vector<pair<int, float>> & ordering, int iter = 100, float alpha = 0.85) {
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

int read_graph(string & fname, int & num_nodes, int &num_edges, vector<int> & row_ptr, vector<int> & col_ind) {
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

	int totalRegression = 0;
	// best version:
	vector<float> regressionParams = {0.15, 0.0, 0.1, 0.05, 0.7, 0.0};
	// for other trials:
	// vector<float> regressionParams = {0.636879, 0.599466, 0.585907, 0.568834, 0.610054, 0.541399};

	const char* path = argv[1];
	ofstream out;
	out.open("metric_evaluation1.csv");
	out << "Graph Name, Num_nodes, Num_edges, DegreeOrderDesc, Degree2OrderDesc,  Degree3OrderDesc,"
		<< "Random, ClusteringCoeffDesc, ClosenessCentralityDesc, PageRankDesc, Regression "
		<< endl;

	DIR *pDIR;
	struct dirent *entry;
	if (pDIR = opendir(path)) {
		while (entry = readdir(pDIR)) {
			//cout << entry->d_name << endl;
			string fname = path + ((string)entry->d_name);
			if (fname.at(fname.length() - 1) == '.') {
				continue;
			}
			//cout << fname << endl;
			int num_nodes, num_edges;
			vector<int> row_ptr, col_ind;
			if (read_graph(fname, num_nodes, num_edges, row_ptr, col_ind) == -1) {
				cerr << "error reading graph" << endl;
				return 0;
			}
			// initializations
			out << entry->d_name << "," << num_nodes << "," << num_edges << ",";
			int maxDegree = -1;
			for(int i=0; i<num_nodes; i++){
				if((row_ptr[i+1]-row_ptr[i]) > maxDegree){
						maxDegree = row_ptr[i+1]-row_ptr[i];
					}
			}

			vector<vector<pair<int, float> > > order(8,vector<pair<int,float> >(num_nodes));

			degree_order(num_nodes, row_ptr, col_ind, order[0]);
			degree_2_order(num_nodes, row_ptr, col_ind, order[1]);
			degree_3_order(num_nodes, row_ptr, col_ind, order[2]);
			clustering_coeff(num_nodes, row_ptr, col_ind, order[3]);
			closeness_centrality(num_nodes, row_ptr, col_ind, order[4]);
			page_rank(num_nodes, row_ptr, col_ind, order[5]);

			normalize(order);

			for(int i=0; i<num_nodes; i++){
				order[6][i] = make_pair<int, float>((int)i, (float)(regressionParams[0]*order[0][i].second + regressionParams[1]*order[1][i].second + regressionParams[2]*order[2][i].second + regressionParams[3] * order[3][i].second + regressionParams[4]*order[4][i].second + regressionParams[5]*order[5][i].second));
				order[7][i] = order[0][i];
			}
			vector<vector<int> > color_arr(8);
			sort(order[0].begin(), order[0].end(), descending);
			sort(order[1].begin(), order[1].end(), descending);
			sort(order[2].begin(), order[2].end(), descending);
			sort(order[3].begin(), order[3].end(), descending);
			sort(order[4].begin(), order[4].end(), descending);
			sort(order[5].begin(), order[5].end(), descending);
			sort(order[6].begin(), order[6].end(), descending);
			random_shuffle(order[7].begin(), order[7].end());
			/*for(int i=0; i<num_nodes; i++){
				cout << order[0][i].first << "  " << order[1][i].first<< "  " << order[2][i].first<< "  " << order[3][i].first << "  " << order[4][i].first<< "  " << order[5][i].first<< endl;
			}*/

			//cout << graph_coloring(row_ptr, col_ind, dorder, color_arr, maxDegree) <<endl;
			//cout << graph_coloring(row_ptr, col_ind, regress, color_arr, maxDegree) <<endl;
			int deg1Desc = graph_coloring(row_ptr, col_ind, order[0], color_arr[0], maxDegree);
			int deg2Desc = graph_coloring(row_ptr, col_ind, order[1], color_arr[1], maxDegree);
			int deg3Desc = graph_coloring(row_ptr, col_ind, order[2], color_arr[2], maxDegree);
			int clusteringCoeffDesc = graph_coloring(row_ptr, col_ind, order[3], color_arr[3], maxDegree);
			int closenessCentralityDesc = graph_coloring(row_ptr, col_ind, order[4], color_arr[4], maxDegree);
			int pageRankDesc = graph_coloring(row_ptr, col_ind, order[5], color_arr[5], maxDegree);
			int regressionDesc = graph_coloring(row_ptr, col_ind, order[6], color_arr[6], maxDegree);
			int random = graph_coloring(row_ptr, col_ind, order[7], color_arr[7], maxDegree);

			totalRegression += regressionDesc;

			out << deg1Desc << "," << deg2Desc << "," << deg3Desc << "," << random << "," << clusteringCoeffDesc << "," << closenessCentralityDesc << "," << pageRankDesc << "," << regressionDesc << endl;
			cout << deg1Desc << "," << deg2Desc << "," << deg3Desc << "," << random << "," << clusteringCoeffDesc << "," << closenessCentralityDesc << "," << pageRankDesc << "," << regressionDesc << endl;
		}
		closedir(pDIR);
	}
	out.close();
	cout << "In total: " << totalRegression << endl;
	return 0;
}
