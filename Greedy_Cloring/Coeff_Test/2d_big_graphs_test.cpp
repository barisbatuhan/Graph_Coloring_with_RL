#include "./../API/graph.h"

int main(int argc, char **argv)
{
	const char *path = argv[1];
	DIR *pDIR;
	struct dirent *entry;
	vector<string> gnames;
	vector<pair<vector<int>, vector<int>>> graphs; //s = random_ugraphs_generator(10,100,9000);
	int k = 0;
	vector<float> results(9, 0.0);
	if (pDIR = opendir(path))
	{
		while (entry = readdir(pDIR))
		{
			// get graph name
			string g = entry->d_name;
			string fname = path + ((string)entry->d_name);
			if (fname.at(fname.length() - 1) == '.')
			{
				continue;
			}
			gnames.push_back(g);
			// read graph
			int num_nodes, num_edges;
			vector<int> row_ptr, col_ind;
			if (read_graphs(fname, num_nodes, num_edges, row_ptr, col_ind) == -1)
			{
				cerr << "error reading graph" << endl;
				return 0;
			}
			graphs.push_back({row_ptr, col_ind});
			// create output file
			ofstream file;
			file.open(g + ".txt");
			file << g << " " << num_nodes << " " << num_edges << endl;

			//orderings
			vector<vector<pair<int, float>>> orders(8, vector<pair<int, float>>(num_nodes));

			// orderings
			degree_order(num_nodes, row_ptr, col_ind, orders[0]);
			file << "degree" << endl;
			degree_2_order(num_nodes, row_ptr, col_ind, orders[1]);
			file << "degree2" << endl;
			degree_3_order(num_nodes, row_ptr, col_ind, orders[2]);
			file << "degree3" << endl;
			closeness_centrality_approx(num_nodes, row_ptr, col_ind, orders[3]);
			file << "closeness" << endl;
			clustering_coeff(num_nodes, row_ptr, col_ind, orders[4]);
			file << "cluster" << endl;
			page_rank(num_nodes, row_ptr, col_ind, orders[5]);
			file << "page" << endl;

			// normalization
			normalize(orders, 6); // normalize first 6

			// combination of orderings
			for (int i = 0; i < num_nodes; i++)
			{
				// weighted
				orders[6][i] = make_pair(i, 0.05 * orders[0][i].second + 0.05 * orders[2][i].second + 0.9 * orders[3][i].second);
				//uniform
				orders[7][i] = make_pair(i, 1 / 6 * orders[0][i].second + 1 / 6 * orders[1][i].second + 1 / 6 * orders[2][i].second + 1 / 6 * orders[3][i].second + 1 / 6 * orders[4][i].second + 1 / 6 * orders[5][i].second);
			}

			// sort values to create permutations
			for (int i = 0; i < orders.size(); i++)
			{
				sort(orders[i].begin(), orders[i].end(), descending);
			}

			// color the graph for each permutation
			int random_size = 5;
			float d, totalcolors, randomsum = 0;
			d = graph_2d_coloring(row_ptr, col_ind, orders[1]);
			for (int i = 0; i < results.size() - 1; i++)
			{
				if (i == 1)
				{
					file << gnames[k] << " order " << i << ": " << d << " degreeone: " << d << endl;
					results[i] += totalcolors / d;
					continue;
				}
				totalcolors = graph_2d_coloring(row_ptr, col_ind, orders[i]);
				results[i] += totalcolors / d;
				file << gnames[k] << " order " << i << ": " << totalcolors << " degreeone: " << d << endl;
			}
			// color the graph randomly
			for (int i = 0; i < random_size; i++)
			{
				random_shuffle(orders[0].begin(), orders[0].end());
				randomsum += graph_2d_coloring(row_ptr, col_ind, orders[0]) / d;
			}
			randomsum /= random_size;
			results[8] += randomsum;

			k++;
			file.close();
		}
		closedir(pDIR);
	}
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
