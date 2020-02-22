#include "./../API/graph.h"

int main(int argc, char **argv)
{
	const char *path = argv[1];
	DIR *pDIR;
	struct dirent *entry;
	map<string, int> ht;

	const char *chrom = "./../../Optimal_Coloring/chromatic_numbers.csv";
	ifstream f(chrom);
	string graphname;
	int chronum;
	getline(f, graphname);

	while (!f.eof())
	{
		getline(f, graphname);
		if (graphname.length() < 2)
		{
			break;
		}
		graphname = graphname.substr(0, graphname.rfind(","));
		string d = graphname.substr(graphname.rfind(",") + 1, graphname.length());
		graphname = graphname.substr(0, graphname.find(","));
		chronum = stoi(d);
		ht.insert(pair<string, int>(graphname, chronum));
	}

	vector<string> gnames;
	vector<pair<vector<int>, vector<int>>> graphs; //s = random_ugraphs_generator(10,100,9000);

	// read graphs and store baseline in graphname-color map
	if (pDIR = opendir(path))
	{
		while (entry = readdir(pDIR))
		{
			string g = entry->d_name;
			string fname = path + ((string)entry->d_name);
			if (fname.at(fname.length() - 1) == '.')
			{
				continue;
			}
			gnames.push_back(g);

			int num_nodes, num_edges;
			vector<int> row_ptr, col_ind;
			if (read_graphs(fname, num_nodes, num_edges, row_ptr, col_ind) == -1)
			{
				cerr << "error reading graph" << endl;
				return 0;
			}
			graphs.push_back({row_ptr, col_ind});
			//cout << g << " " << num_nodes << " " << num_edges <<endl;
		}
		closedir(pDIR);
	}

	ofstream out;
	out.open("1dsmallresults.csv");
	out << "gname,family,nodes,edges,degree1,degree2,degree3,closeness,clustering,pagerank,weighted,uniform,random" << endl;

	vector<float> results(9, 0.0);
	int k = 0;
	for (auto &graph : graphs)
	{
		// initializations
		auto &row_ptr = graph.first;
		auto &col_ind = graph.second;
		int num_nodes = row_ptr.size() - 1;
		int num_edges = row_ptr[row_ptr.size() - 1];

		vector<vector<pair<int, float>>> orders(8, vector<pair<int, float>>(num_nodes));
		// orderings
		degree_order(num_nodes, row_ptr, col_ind, orders[0]);
		degree_2_order(num_nodes, row_ptr, col_ind, orders[1]);
		degree_3_order(num_nodes, row_ptr, col_ind, orders[2]);
		closeness_centrality(num_nodes, row_ptr, col_ind, orders[3]);
		clustering_coeff(num_nodes, row_ptr, col_ind, orders[4]);
		page_rank(num_nodes, row_ptr, col_ind, orders[5]);

		// normalization
		normalize(orders);

		for (int i = 0; i < num_nodes; i++)
		{
			// weighted
			orders[6][i] = make_pair(i, 0.15 * orders[0][i].second + 0.05 * orders[1][i].second + 0.55 * orders[2][i].second + 0.2 * orders[3][i].second + 0.05 * orders[4][i].second);
			//uniform
			orders[7][i] = make_pair(i, 1 / 6 * orders[0][i].second + 1 / 6 * orders[1][i].second + 1 / 6 * orders[2][i].second + 1 / 6 * orders[3][i].second + 1 / 6 * orders[4][i].second + 1 / 6 * orders[5][i].second);
		}

		string path_to_graph = string(path) + gnames[k];
		string family = read_family(path_to_graph);

		float d = ht[gnames[k]];
		out << gnames[k] << "," << family << "," << num_nodes << "," << num_edges << ",";

		// sort values to create permutations
		for (int i = 0; i < orders.size(); i++)
		{
			sort(orders[i].begin(), orders[i].end(), descending);
		}
		// color the graph for each permutation
		int random_size = 5;
		float totalcolors, randomsum = 0;
		for (int i = 0; i < results.size() - 1; i++)
		{
			totalcolors = graph_1d_coloring(row_ptr, col_ind, orders[i]);
			if (i == 0)
			{
				out << totalcolors;
			}
			else
			{
				out << "," << totalcolors;
			}
			results[i] += totalcolors / d;
		}
		// color the graph randomly
		for (int i = 0; i < random_size; i++)
		{
			random_shuffle(orders[0].begin(), orders[0].end());
			randomsum += graph_1d_coloring(row_ptr, col_ind, orders[0]);
		}
		randomsum /= random_size; // average
		out << "," << randomsum << endl;
		results[8] += randomsum / d;
		k++;
	}
	out.close();

	for (auto &x : results)
	{
		x /= k;
	}
	for (int i = 0; i < results.size(); i++)
	{
		cout << "order " << i << " result: " << results[i] << endl;
	}
	return 0;
}
