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

using namespace std;

bool descending(const pair<int, float> & left, const pair<int, float> & right) {
	return left.second>right.second;
}
bool ascending(const pair<int, float> & left, const pair<int, float> & right) {
	return left.second<right.second;
}

bool descendingFirst(const pair<int, float> & left, const pair<int, float> & right) {
	return left.first>right.first;
}
bool ascendingFirst(const pair<int, float> & left, const pair<int, float> & right) {
	return left.first<right.first;
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

int graph_coloring(const vector<int> & row_ptr, const vector<int> & col_ind, const vector<pair<int, float>> & ordering, vector<int> & color_arr, int maxdegree, string type) {
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
	if (hasEdge) {
		nofcolors++;
	}
	cout << "with " << type << " nof colors: " << nofcolors << endl;
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
		float coeff = possiblelinks != 0 ? (float)noflinks / possiblelinks : 0;
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
		if (renameArr[v1] == -1 && eliminateUnused) {
			renameArr[v1] = counter;
			v1 = counter;
			counter++;
		}
		else if (eliminateUnused) {
			v1 = renameArr[v1];
		}
		if (renameArr[v2] == -1 && eliminateUnused) {
			renameArr[v2] = counter;
			v2 = counter;
			counter++;
		}
		else if (eliminateUnused) {
			v2 = renameArr[v2];
		}

		if (v1 != v2) {
			adj_list[v1].push_back(v2); // add the edge v1->v2
			adj_list[v2].push_back(v1); // add the edge v2->v1
		}
	}
	if (eliminateUnused) {
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

	return 0;
}

int runAlgorithm(string name, bool isAscending, int num_nodes, const vector<int> & row_ptr,
	const vector<int> & col_ind, int maxDegree, int iter = 100, float alpha = 0.85) {
	vector<pair<int, float> > ordering(num_nodes);
	vector<int> color_arr;
	int result = 0;

	if (name == "Degree1") {
		degree_order(num_nodes, row_ptr, col_ind, ordering);
	}
	else if (name == "Degree2") {
		degree_2_order(num_nodes, row_ptr, col_ind, ordering);
	}
	else if (name == "Degree3") {
		degree_3_order(num_nodes, row_ptr, col_ind, ordering);
	}
	else if (name == "ClusteringCoeff") {
		clustering_coeff(num_nodes, row_ptr, col_ind, ordering);
	}
	else if (name == "ClosenessCentrality") {
		closeness_centrality(num_nodes, row_ptr, col_ind, ordering);
	}
	else if (name == "PageRank") {
		page_rank(num_nodes, row_ptr, col_ind, ordering);
	}
	else {
		cerr << "Wrong algorithm name is entered: " << name << endl;
		return -1;
	}
	if (isAscending) {
		sort(ordering.begin(), ordering.end(), ascending);
		name += " Ascending";
	}
	else {
		sort(ordering.begin(), ordering.end(), descending);
		name += " Descending";
	}
	return graph_coloring(row_ptr, col_ind, ordering, color_arr, maxDegree, name);
}


void findDifferentOrderings(const char* path) {

	ofstream out;
	out.open("metric_evaluation1.csv");
	out << "Graph Name, Num_nodes, Num_edges, DegreeOrderDesc, DegreeOrderAsc, Degree2OrderDesc,  Degree2OrderAsc,  Degree3OrderDesc, Degree3OrderAsc,"
		<< "Random, ClusteringCoeffDesc,ClusteringCoeffAsc, ClosenessCentralityDesc, ClosenessCentralityAsc, PageRankDesc, PageRankDesc "
		<< endl;

	DIR *pDIR;
	struct dirent *entry;
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
			if (read_graph(fname, num_nodes, num_edges, row_ptr, col_ind) == -1) {
				cerr << "error reading graph" << endl;
				return;
			}
			// initializations
			out << entry->d_name << "," << num_nodes << "," << num_edges << ",";

			// in order to initialize maxDegree
			vector<pair<int, float> > order(num_nodes);
			degree_order(num_nodes, row_ptr, col_ind, order);
			sort(order.begin(), order.end(), descending);
			int maxDegree = order[0].second;

			// for randomization
			vector<int> color_arr;
			random_shuffle(order.begin(), order.end());

			// test metrics
			int deg1Desc = runAlgorithm("Degree1", false, num_nodes, row_ptr, col_ind, maxDegree);
			int deg1Asc = runAlgorithm("Degree1", true, num_nodes, row_ptr, col_ind, maxDegree);
			int deg2Desc = runAlgorithm("Degree2", false, num_nodes, row_ptr, col_ind, maxDegree);
			int deg2Asc = runAlgorithm("Degree2", true, num_nodes, row_ptr, col_ind, maxDegree);
			int deg3Desc = runAlgorithm("Degree3", false, num_nodes, row_ptr, col_ind, maxDegree);
			int deg3Asc = runAlgorithm("Degree3", true, num_nodes, row_ptr, col_ind, maxDegree);
			int clusteringCoeffDesc = runAlgorithm("ClusteringCoeff", false, num_nodes, row_ptr, col_ind, maxDegree);
			int clusteringCoeffAsc = runAlgorithm("ClusteringCoeff", true, num_nodes, row_ptr, col_ind, maxDegree);
			int closenessCentralityDesc = runAlgorithm("ClosenessCentrality", false, num_nodes, row_ptr, col_ind, maxDegree);
			int closenessCentralityAsc = runAlgorithm("ClosenessCentrality", true, num_nodes, row_ptr, col_ind, maxDegree);
			int pageRankDesc = runAlgorithm("PageRank", false, num_nodes, row_ptr, col_ind, maxDegree);
			int pageRankAsc = runAlgorithm("PageRank", true, num_nodes, row_ptr, col_ind, maxDegree);
			int random = graph_coloring(row_ptr, col_ind, order, color_arr, maxDegree, "RandomOrder");

			out << deg1Desc << "," << deg1Asc << "," << deg2Desc << "," << deg2Asc << "," << deg3Desc << "," << deg3Asc
				<< "," << random << "," << clusteringCoeffDesc << "," << clusteringCoeffAsc << ","
				<< closenessCentralityDesc << "," << closenessCentralityAsc << "," << pageRankDesc << "," << pageRankAsc << endl;
    	}
		closedir(pDIR);
	}
	out.close();
}

/** The order of the arguments in the weightList has to be as follows:
 * 1 - Order 1 weight
 * 2 - Order 2 weight
 * 3 - Order 3 weight
 * 4 - Clustering Coefficient weight
 * 5 - Closeness Centrality weight
 * 6 - Page Rank weight
 * 
 *  !!! if you want using directly the weights that are given to you, then pass second parameter as NULL
*/
void findWeightedAnalysis(const char* path, const char* csvPath, long totalOptimal, vector<float> weightList = vector<float>(6, 0.0)) {

	if(csvPath != NULL) {
		// weights 
		long totalDeg1 = 0, totalDeg2 = 0, totalDeg3 = 0, totalClusCoeff = 0, totalClosenessCentrality = 0, totalPageRank = 0;

		// get Order Colorings from the csv file created
		ifstream input(csvPath);
		if (input.fail()) {
			cerr << "The csv path is not correct!!" << endl;
			return;
		}
	
		// read graph
		string line = "";

		// discard first row since it includes labels
		getline(input, line);
		// read other rows
		while(getline(input, line)) {
			istringstream subInput(line);
			int counter = 0;
			string element;
			while(getline(subInput, element, ',')){
				if(counter == 3) totalDeg1 += stoi(element);
				else if(counter == 5) totalDeg2 += stoi(element);
				else if(counter == 7) totalDeg3 += stoi(element);
				else if(counter == 9) totalClusCoeff += stoi(element);
				else if(counter == 11) totalClosenessCentrality += stoi(element);
				else if(counter == 13) totalPageRank += stoi(element);
				counter++;
			}
		}
		// weight calculation
		weightList[0] = (float) totalOptimal / totalDeg1;
		weightList[1] = (float) totalOptimal / totalDeg2;
		weightList[2] = (float) totalOptimal / totalDeg3;
		weightList[3] = (float) totalOptimal / totalClusCoeff;
		weightList[4] = (float) totalOptimal / totalClosenessCentrality;
		weightList[5] = (float) totalOptimal / totalPageRank;
	}

	ofstream out;
	out.open("weighted_metric1.csv");
	out << "Graph Name, Num_nodes, Num_edges, WeightedResult" << endl;

	DIR *pDIR;
	struct dirent *entry;
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
			if (read_graph(fname, num_nodes, num_edges, row_ptr, col_ind) == -1) {
				cerr << "error reading graph" << endl;
				return;
			}
			// initializations
			out << entry->d_name << "," << num_nodes << "," << num_edges << ",";

			// getting orders for each algorithm
			vector<pair<int, float>> deg1Order(num_nodes);
			degree_order(num_nodes, row_ptr, col_ind, deg1Order);
			sort(deg1Order.begin(), deg1Order.end(), ascendingFirst);
			vector<pair<int, float>> deg2Order(num_nodes);
			degree_2_order(num_nodes, row_ptr, col_ind, deg2Order);
			sort(deg2Order.begin(), deg2Order.end(), ascendingFirst);
			vector<pair<int, float>> deg3Order(num_nodes);
			degree_3_order(num_nodes, row_ptr, col_ind, deg3Order);
			sort(deg3Order.begin(), deg3Order.end(), ascendingFirst);
			vector<pair<int, float>> clussCoeffOrder(num_nodes);
			clustering_coeff(num_nodes, row_ptr, col_ind, clussCoeffOrder);
			sort(clussCoeffOrder.begin(), clussCoeffOrder.end(), ascendingFirst);
			vector<pair<int, float>> closenessCentralityOrder(num_nodes);
			closeness_centrality(num_nodes, row_ptr, col_ind, closenessCentralityOrder);
			sort(closenessCentralityOrder.begin(), closenessCentralityOrder.end(), ascendingFirst);
			vector<pair<int, float>> pageRankOrder(num_nodes);
			page_rank(num_nodes, row_ptr, col_ind, pageRankOrder);
			sort(pageRankOrder.begin(), pageRankOrder.end(), ascendingFirst);
			
			vector<pair<int, float>> finalOrder(num_nodes);
			for(int i = 0; i < num_nodes; i++) {
				float value = deg1Order[i].second * weightList[0] + deg2Order[i].second * weightList[1] 
							+ deg3Order[i].second * weightList[2] + clussCoeffOrder[i].second * weightList[3] 
							+ closenessCentralityOrder[i].second * weightList[4] + pageRankOrder[i].second * weightList[5];
				
				finalOrder[i]= make_pair(i, value);
			}
			vector<int> color_arr;
			sort(deg1Order.begin(), deg1Order.end(), descending);
			int maxDegree = deg1Order[0].second;

			sort(finalOrder.begin(), finalOrder.end(), descending);
			int result = graph_coloring(row_ptr, col_ind, finalOrder, color_arr, maxDegree, "Weighted Result");
			out << result << endl;
    	}
		closedir(pDIR);
	}
	out.close();
}

int main(int argc, char** argv) {
	// findDifferentOrderings(argv[1]);
	findWeightedAnalysis(argv[1], "./ordering_greedy_results.csv", 1347);
	return 0;
}
