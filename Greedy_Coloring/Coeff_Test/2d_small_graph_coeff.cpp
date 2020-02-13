#include "./../API/graph.h"

using namespace std;

int main(int argc, char **argv)
{
	const char *path = argv[1];
	DIR *pDIR;
	struct dirent *entry;
	map<string, int> ht;

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
			cout << g << " " << num_nodes << " " << num_edges << endl;

			//baseline
			vector<pair<int, float>> orders(num_nodes);
			vector<int> color_arr;

			degree_2_order(num_nodes, row_ptr, col_ind, orders);
			sort(orders.begin(), orders.end(), descending);
			int k = graph_2d_coloring(row_ptr, col_ind, orders);
			ht.insert(pair<string, int>(g, k));
		}
		closedir(pDIR);
	}

	float sum;
	vector<vector<float>> coefficients(32, vector<float>(6)); // for each thread
	vector<float> min_colors(32, INT_MAX);					  // for each thread
	vector<float> coef_list(21);
	for (int i = 0; i <= 20; i++)
	{
		coef_list[i] = i * 0.05;
	}

#pragma omp parallel for collapse(6) schedule(dynamic) num_threads(32)
	for (int i1 = 0; i1 < 21; i1++)
	{
		for (int i2 = 0; i2 < 21; i2++)
		{
			for (int i3 = 0; i3 < 21; i3++)
			{
				for (int i4 = 0; i4 < 21; i4++)
				{
					for (int i5 = 0; i5 < 21; i5++)
					{
						for (int i6 = 0; i6 < 21; i6++)
						{
							float c1, c2, c3, c4, c5, c6; // thread private variables
							c1 = coef_list[i1];
							c2 = coef_list[i2];
							c3 = coef_list[i3];
							c4 = coef_list[i4];
							c5 = coef_list[i5];
							c6 = coef_list[i6];
							sum = c1 + c2 + c3 + c4 + c5 + c6;
							if (sum == 1)
							{ // sum of coeff must be one
								printf("%f %f %f %f %f %f\n", c1, c2, c3, c4, c5, c6);
								float total_colors = 0;
								int tid = omp_get_thread_num();
								int i = 0;
								for (auto &graph : graphs)
								{
									// initializations
									auto &row_ptr = graph.first;
									auto &col_ind = graph.second;
									int num_nodes = row_ptr.size() - 1;
									vector<vector<pair<int, float>>> orders(6, vector<pair<int, float>>(num_nodes));

									// orderings
									degree_order(num_nodes, row_ptr, col_ind, orders[0]);
									degree_2_order(num_nodes, row_ptr, col_ind, orders[1]);
									degree_3_order(num_nodes, row_ptr, col_ind, orders[2]);
									closeness_centrality(num_nodes, row_ptr, col_ind, orders[3]);
									clustering_coeff(num_nodes, row_ptr, col_ind, orders[4]);
									page_rank(num_nodes, row_ptr, col_ind, orders[5]);

									// normalization
									normalize(orders);

									vector<int> color_arr;
									vector<pair<int, float>> new_order(num_nodes);
									for (int i = 0; i < num_nodes; i++)
									{
										new_order[i] = make_pair(i, c1 * orders[0][i].second + c2 * orders[1][i].second + c3 * orders[2][i].second + c4 * orders[3][i].second + c5 * orders[4][i].second + c6 * orders[5][i].second);
										//new_order[i] = make_pair(i,0.05*orders[2][i].second + 0.1*orders[1][i].second + 0.15*orders[0][i].second + 0.7*orders[3][i].second);
										//new_order[i] = make_pair(i, orders[1][i].second);
									}

									// coloring in that permutation
									sort(new_order.begin(), new_order.end(), descending);
									float k = graph_2d_coloring(row_ptr, col_ind, new_order);
									float d = ht[gnames[i]]; // 2d coloring baseline
									if (d == 0)
									{
										continue;
									}
									total_colors += k / d; // ratio to baseline
									i++;
								}
								total_colors = total_colors / i; // average ratio to baseline
								if (total_colors < min_colors[tid])
								{ // if it is the best performance keep the best coefficients
									coefficients[tid][0] = c1;
									coefficients[tid][1] = c2;
									coefficients[tid][2] = c3;
									coefficients[tid][3] = c4;
									coefficients[tid][4] = c5;
									coefficients[tid][5] = c6;
									printf("Total colors : %f\n", total_colors);
									min_colors[tid] = total_colors;
								}
							}
						}
					}
				}
			}
		}
	}
	float mincolors = INT_MAX;
	int tid = -1;
	for (int i = 0; i < min_colors.size(); i++)
	{
		if (min_colors[i] < mincolors)
		{
			mincolors = min_colors[i];
			tid = i;
		}
	}
	cout << "best result is " << mincolors << endl;
	cout << "Coefficients are ";
	for (auto &x : coefficients[tid])
	{
		cout << x << " ";
	}
	cout << endl;
	return 0;
}
