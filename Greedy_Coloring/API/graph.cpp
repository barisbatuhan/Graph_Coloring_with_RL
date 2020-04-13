#include "./graph.h"

/* HELPER FUNCTIONS FOR PAIR OPERATIONS */

bool descending(const pair<int, float> &left, const pair<int, float> &right)
{
	if ((left.second > right.second) || (left.second == right.second && left.first < right.first))
	{
		return true;
	}
	return false;
}

bool ascending(const pair<int, float> &left, const pair<int, float> &right)
{
	if ((left.second < right.second) || (left.second == right.second && left.first < right.first))
	{
		return true;
	}
	return false;
}

pair<int, float> add(const float &left, const pair<int, float> &right)
{
	return pair<int, float>(right.first, left + right.second);
}

/* ---------------------------------------- */

/* READ & WRITE OPERATIONS */

void get_filenames(vector<string> &filenames, const vector<string> &locations)
{
    for (int i = 0; i < locations.size(); i++)
    {
        if (auto dir = opendir(locations[i].c_str()))
        {
            while (auto f = readdir(dir))
            {
                if (!f->d_name || f->d_name[0] == '.')
                    continue;

                string path = locations[i] + f->d_name;
                filenames.push_back(path);
            }
        }
    }
}

int read_graphs(string &fname, int &num_nodes, int &num_edges, vector<int> &row_ptr, vector<int> &col_ind)
{
	ifstream input(fname.c_str());
	if (input.fail())
	{
		return -1;
	}
	// read graph
	string line = "%";
	while (line.find("%") != string::npos)
	{
		getline(input, line);
	}
	istringstream ss(line);
	ss >> num_nodes >> num_nodes >> num_edges;
	int v1, v2;
	double weight;

	vector<int> renameArr(num_nodes, -1);
	int counter = 0;
	bool eliminateUnused = true;

	vector<vector<int>> adj_list(num_nodes);
	for (int i = 0; i < num_edges; i++)
	{
		getline(input, line);
		istringstream inp(line);
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

	row_ptr = vector<int>(num_nodes + 1);
	col_ind = vector<int>(2 * num_edges);
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
	//cout << "nof nodes " << num_nodes << endl;
	//cout << "nof edges " << num_edges << endl;
	return 0;
}

string read_family(string &fname)
{
	string family = "not available";
	ifstream input(fname.c_str());
	if (input.fail())
	{
		return "-1";
	}
	// read graph
	string line = "%";
	while (line.find("%") != string::npos)
	{
		getline(input, line);
		if (line.find("kind:") != string::npos)
		{
			family = line.substr(8);
			family = family.substr(0, family.length() - 1); // newline is excluded
			// cout << fname << " - " << family << endl;
			break;
		}
	}
	input.close();
	return family;
}

/* ---------------------------------------- */

/* COLORING FUNCTIONS */

int graph_1d_coloring(const vector<int> &row_ptr, const vector<int> &col_ind, const vector<pair<int, float>> &ordering)
{
	vector<int> color_arr(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(row_ptr.size() - 1, -1);
	bool hasEdge = false;
	for (int i = 0; i < ordering.size(); i++)
	{
		const int &node = ordering[i].first; //for each node in ordering
		for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++)
		{
			hasEdge = true;
			const int &adj = col_ind[edge]; //for each adjacent node
			if (color_arr[adj] != -1)
			{									   // if it is already colored
				forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
			}
		}
		for (int color = 0; color < ordering.size(); color++)
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
	// if (!isValid(color_arr, row_ptr, col_ind, ordering.size()) == true)
	// {
	// 	cout << "ERROR" << endl;
	// }
	return nofcolors;
}

int graph_2d_coloring(const vector<int> &row_ptr, const vector<int> &col_ind, const vector<pair<int, float>> &ordering)
{
	vector<int> color_arr(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(row_ptr.size() - 1, -1);
	bool hasEdge = false;
	for (int i = 0; i < ordering.size(); i++)
	{
		const int &node = ordering[i].first; //for each node in ordering
		for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++)
		{
			hasEdge = true;
			const int &adj = col_ind[edge]; //for each adjacent node
			if (color_arr[adj] != -1)
			{									   // if it is already colored
				forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
			}
			for (int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj + 1]; edge_2++)
			{
				const int &adj_neigh = col_ind[edge_2];
				if (color_arr[adj_neigh] != -1)
				{											 // if it is already colored
					forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
				}
			}
		}
		for (int color = 0; color < ordering.size(); color++)
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
	// if (!isValid2d(color_arr, row_ptr, col_ind, ordering.size()) == true) {
	// 	cout << "ERROR" << endl;
	// }
	return nofcolors;
}

int saturation_1d_coloring(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &spare_order)
{
	vector<unordered_set<int>> color_infos(num_nodes);
	vector<pair<int, int>> node_values(2, {1, 0});
	vector<bool> node_colored(num_nodes, false);
	// vector<int> node_colored_cnt(num_nodes, 0);

	vector<int> color_arr(num_nodes, -1);
	int nofcolors = 0;
	bool hasEdge = false;
	int colored_num = 0;

	do
	{
		int i = 0;
		pair<int, int> &node = node_values[i];
		while (node_colored[node.first] == true)
		{
			node = node_values[++i];
		}
		pair<int, int> &next_node = node_values[i + 1];
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
			// node_colored_cnt[adj]++;
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
			if(curr_colors > node_values[0].second) {
				node_values[1].second = node_values[0].second;
				node_values[1].first = node_values[0].first;
				node_values[0].second = curr_colors;
				node_values[0].first = i;
			}
			else if(curr_colors > node_values[1].second) {
				node_values[1].second = curr_colors;
				node_values[1].first = i;
			}
		}
	} while (colored_num < num_nodes);

	if (hasEdge)
	{
		nofcolors++;
	}
	// if (!isValid(color_arr, row_ptr, col_ind, spare_order.size()) == true)
	// {
	// 	cout << "ERROR" << endl;
	// }
	return nofcolors;
}

int saturation_2d_coloring(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &spare_order)
{
	vector<unordered_set<int>> color_infos(num_nodes);
	vector<pair<int, int>> node_values(2, {1, 0});
	vector<bool> node_colored(num_nodes, false);

	vector<int> color_arr(num_nodes, -1);
	int nofcolors = 0;
	bool hasEdge = false;
	int colored_num = 0;

	do
	{
		int i = 0;
		pair<int, int> &node = node_values[i];
		while (node_colored[node.first] == true)
		{
			node = node_values[++i];
		}
		pair<int, int> &next_node = node_values[i + 1];
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
			if(curr_colors > node_values[0].second) {
				node_values[1].second = node_values[0].second;
				node_values[1].first = node_values[0].first;
				node_values[0].second = curr_colors;
				node_values[0].first = i;
			}
			else if(curr_colors > node_values[1].second) {
				node_values[1].second = curr_colors;
				node_values[1].first = i;
			}
		}
	} while (colored_num < num_nodes);

	if (hasEdge)
	{
		nofcolors++;
	}
	// if (!isValid2d(color_arr, row_ptr, col_ind, spare_order.size()) == true)
	// {
	// 	cout << "ERROR" << endl;
	// }
	return nofcolors;
}

/* ---------------------------------------- */

/* COLORING VALIDITY CHECKERS */

bool isValid(const vector<int> &color_arr, const vector<int> &row_ptr, const vector<int> &col_ind, int num_nodes)
{
	for (int v = 0; v < num_nodes; v++)
	{ // for each node v
		if(color_arr[v] == -1) {
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

bool isValid2d(const vector<int> &color_arr, const vector<int> &row_ptr, const vector<int> &col_ind, int num_nodes)
{
	for (int v = 0; v < num_nodes; v++)
	{ // for each node v
		if(color_arr[v] == -1) {
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
				if(adj_neigh == v) continue;
				if (color_arr[adj_neigh] == color_arr[v])
				{ // if color of v equals its adjacent's neighbor's color
					return false;
				}
			}
		}
	}
	return true;
}

/* ---------------------------------------- */

/* ORDERING ALGORITHMS */

void degree_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering)
{
	#pragma omp parallel for num_threads(32) schedule(dynamic)
	for (int v = 0; v < num_nodes; v++)
	{
		ordering[v] = make_pair(v, row_ptr[v + 1] - row_ptr[v]);
	}
}

void degree_2_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering)
{
	#pragma omp parallel for num_threads(32) schedule(dynamic)
	for (int v = 0; v < num_nodes; v++)
	{
		vector<int> dist_arr(num_nodes);
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr, 2); // take distance array for node v
		int count = 0, val;
		for (int i = 0; i < dist_arr.size(); i++)
		{
			val = dist_arr[i];
			if (val == 1 || val == 2)
			{
				count++;
			}
		}
		ordering[v] = make_pair(v, count);
	}
}

void degree_3_order(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering)
{
	#pragma omp parallel for num_threads(32) schedule(dynamic)
	for (int v = 0; v < num_nodes; v++)
	{
		vector<int> dist_arr(num_nodes);
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr, 3); // take distance array for node v
		int count = 0, val;
		for (int i = 0; i < dist_arr.size(); i++)
		{
			val = dist_arr[i];
			if (val == 1 || val == 2 || val == 3)
			{
				count++;
			}
		}
		ordering[v] = make_pair(v, count);
	}
}

void closeness_centrality(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering)
{
	vector<int> dist_arr;
	for (int v = 0; v < num_nodes; v++)
	{
		bfs(v, num_nodes, row_ptr, col_ind, dist_arr);							// take distance array for node v
		int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
		float coeff = sum_of_dist > 0 ? (float)num_nodes / sum_of_dist : 0;		// if coefficient is negative(meaning that graph is not connected) assign to 0
		ordering[v] = make_pair(v, coeff);
	}
}

void closeness_centrality_approx(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering, int size)
{
	if (size > num_nodes)
	{
		size = num_nodes;
	}
	vector<vector<int>> dist_arr(size, vector<int>(num_nodes));

	#pragma omp parallel for num_threads(32) schedule(dynamic)
	for (int v = 0; v < size; v++)
	{
		//srand(1);
		int start_node = rand() % num_nodes;
		bfs(start_node, num_nodes, row_ptr, col_ind, dist_arr[v]); // take distance array for node v
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
		ordering[v] = make_pair(v, coeff);
	}
}

void clustering_coeff(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering)
{

	#pragma omp parallel for num_threads(32) schedule(dynamic)
	for (int v = 0; v < num_nodes; v++)
	{ // for each node
		int degree = row_ptr[v + 1] - row_ptr[v];
		int possiblelinks = degree * (degree - 1) / 2;
		int noflinks = 0;
		unordered_set<int> set;
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
		ordering[v] = make_pair(v, coeff);
	}
}

void page_rank(int num_nodes, const vector<int> &row_ptr, const vector<int> &col_ind, vector<pair<int, float>> &ordering, int iter, float alpha)
{
	// distribute %15 evenly
	float dist_value = (1 - alpha) / num_nodes;
	for (int i = 0; i < num_nodes; i++)
	{
		ordering[i] = make_pair(i, (float)1 / num_nodes);
	}
	// initially likelyhoods are uniformly distributed
	for (int i = 0; i < iter; i++)
	{
		// on each iteration (required for convergence)
		vector<pair<int, float>> copy_ordering(num_nodes, pair<int, float>(0, dist_value));
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

/* ---------------------------------------- */

/* HELPER FUNCTIONS FOR ORDERING ALGORITHMS */

/* BFS */

void bfs(int start_node, int num_nodes, vector<int> row_ptr, vector<int> col_ind, vector<int> &distance_arr, int step_size)
{
	vector<int> frontier(num_nodes, -1);
	frontier[0] = start_node;
	int queuestart = 0, queueend = 1, frontsize = 0;
	distance_arr.assign(num_nodes, -1); // every node is unvisited
	distance_arr[start_node] = 0;		// distance from a node to itself is 0
	int dist = 1;						// initial distance
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
					distance_arr[adj] = dist;				// assign corresponding distance
				}
			}
		} while (queuestart < queueend);
		queueend += frontsize; // add the offset
		frontsize = 0;		   // reset the offset
		dist++;				   // next frontier will be further
	}
}

void normal_params(vector<pair<int, float>> &order, float &mean, float &stdev)
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
	if (isnan(stdev) || stdev == 0)
		stdev = 0.001;
}

void normalize(vector<vector<pair<int, float>>> &orders, int num)
{
	if (num == -1)
	{
		num = orders.size();
	}
	float mean, stdev;
	for (int i = 0; i < num; i++)
	{
		normal_params(orders[i], mean, stdev);
		for_each(orders[i].begin(), orders[i].end(), [mean, stdev](pair<int, float> &x) { x.second = (x.second - mean) / stdev; });
	}
}

/* ---------------------------------------- */

/* DATASET ENLARGING METHODS */

vector<pair<vector<int>, vector<int>>> random_ugraphs_generator(int graph_cnt, int node_cnt, int edge_cnt)
{
	srand(112);
	vector<pair<vector<int>, vector<int>>> graphs;
	for (int i = 0; i < graph_cnt; i++)
	{
		vector<vector<bool>> adj_list(node_cnt, vector<bool>(node_cnt, false));
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
		vector<int> row_ptr(node_cnt + 1);
		vector<int> col_ind(2 * edge_cnt);
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
		pair<vector<int>, vector<int>> col_row_pair = make_pair(row_ptr, col_ind);
		graphs.push_back(col_row_pair);
	}
	return graphs;
}

/* ---------------------------------------- */

