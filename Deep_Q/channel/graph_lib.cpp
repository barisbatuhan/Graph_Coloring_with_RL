#include "graph.h"
#include <iostream>
using namespace std;

/* ---------------------------------------------
COMMUNICATING FUNCTIONS
--------------------------------------------- */

std::vector<Graph> graphs;
Graph curr_graph;
std::vector<std::vector<std::vector<float>>> node_embeds;
std::vector<std::vector<float>> graph_embeds;
std::vector<std::vector<int>> color_arrs;

int nfeatures_size = 9;
int gfeatures_size = 15;

extern "C" void insert_batch(int batch, int n, int e)
{
	for (int i = 0; i < batch; i++)
	{
		Graph g(n, e);
		graphs.push_back(g);
	}
	node_embeds = std::vector<std::vector<std::vector<float>>>(batch,
															   std::vector<std::vector<float>>(nfeatures_size, std::vector<float>(n, 0)));
	graph_embeds = std::vector<std::vector<float>>(batch);
	color_arrs = std::vector<std::vector<int>>(batch, std::vector<int>(n, -1));
}

extern "C" void reset_batch()
{
	graphs.clear();
	graphs.resize(0);
}

void normalize_batch()
{
	for (unsigned int cols = 0; cols < graph_embeds[0].size(); cols++)
	{
		float sq_sum = 0;
		float sum = 0;
		for (unsigned int e = 0; e < graph_embeds.size(); e++)
		{
			auto &val = graph_embeds[e][cols];
			sum += val;
			sq_sum += val * val;
		}
		sum /= graph_embeds.size();
		sq_sum /= graph_embeds.size();
		float std = sqrt(abs(sum * sum - sq_sum)) + 0.0001;
		for (unsigned int e = 0; e < graph_embeds.size(); e++)
		{
			float &val = graph_embeds[e][cols];
			val = (float)(val - sum) / std;
		}
	}
}

void init_graph_embed(int index)
{
	auto &curr_graph = graphs[index];
	auto &row_ptr = curr_graph.row_ptr;
	int size = curr_graph.num_nodes;
	vector<float> closeness;
	curr_graph.closeness_centrality_approx(closeness);

	graph_embeds[index] = std::vector<float>(gfeatures_size, 0);
	auto &graph_embed = graph_embeds[index];

	graph_embed[0] = (float)curr_graph.num_edges / size; // num edges / num nodes

	graph_embed[1] = 0; // nof adjacents of colored nodes
	graph_embed[2] = 0; // nof colored adjacents of colored nodes
	graph_embed[3] = 0; // closeness sum of colored nodes
	graph_embed[4] = 0; // nof colored nodes
	for (int i = 0; i < (int)closeness.size(); i++)
	{ // closeness sum of uncolored nodes
		graph_embed[5] += (float)closeness[i];
	}
	graph_embed[6] = size; // nof uncolored nodes
	graph_embed[7] = 0;	   // degree above mean (among colored nodes)
	graph_embed[8] = 0;	   // degree below mean (among colored nodes)
	graph_embed[9] = 0;	   // degree above mean (among uncolored nodes)
	graph_embed[10] = 0;   // degree below mean (among uncolored nodes)
	double degree_mean = (double)row_ptr.back() / size;
	double degree_sq_sum = 0;
	for (int v = 0; v < (int)row_ptr.size() - 1; v++)
	{
		int degree = row_ptr[v + 1] - row_ptr[v];
		if (degree > degree_mean)
		{
			graph_embed[9]++;
		}
		else
		{
			graph_embed[10]++;
		}
		degree_sq_sum += degree * degree;
	}
	double variance = degree_sq_sum / size - (degree_mean * degree_mean);
	graph_embed[11] = 0;		   // mean degree of colored nodes
	graph_embed[12] = degree_mean; // mean degree of uncolored nodes
	graph_embed[13] = 0;		   // variance of degrees (colored nodes)
	graph_embed[14] = variance;	   // variance of degrees (uncolored nodes)
}

extern "C" void init_graph_embeddings()
{
	for (unsigned int i = 0; i < graphs.size(); i++)
	{
		init_graph_embed(i);
	}
	normalize_batch();
}

// !!!!!!!! in ctypes, subset will be taken
extern "C" float *get_graph_embed(int index, int *size)
{
	*size = graph_embeds[index].size();
	return graph_embeds[index].data();
}

void transpose(vector<vector<float>> &node_embeddings)
{
	int rows = (int)node_embeddings.size();
	int cols = (int)node_embeddings[0].size();
	vector<vector<float>> res(cols, vector<float>(rows));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			res[j][i] = node_embeddings[i][j];
		}
	}
	node_embeddings = res;
}

// !!!!!! no transpose is taken
void init_node_embed(int index)
{
	auto &curr_graph = graphs[index];
	curr_graph.degree_order(node_embeds[index][0]);
	curr_graph.degree_2_order(node_embeds[index][1]);
	curr_graph.degree_3_order(node_embeds[index][2]);
	curr_graph.closeness_centrality_approx(node_embeds[index][3]);
	curr_graph.clustering_coeff(node_embeds[index][4]);
	curr_graph.page_rank(node_embeds[index][5]);
	// dynamic coeffs
	// 6th index -> nof colored neighbors
	// 7th index -> nof different colored neighbors
	// 8th index -> is node itself colored
	// dynamic coefficients are automatically 0 in the beginning
	transpose(node_embeds[index]);
}

extern "C" void init_node_embeddings()
{
	for (unsigned int i = 0; i < graphs.size(); i++)
	{
		init_node_embed(i);
	}
}

// !!!!!!!! in ctypes, subset will be taken
extern "C" float **get_node_embed(int index, int *row, int *col)
{
	*row = node_embeds[index].size();
	*col = node_embeds[index][0].size();
	float **res = new float *[*row];
	for (int i = 0; i < *row; i++)
	{
		res[i] = node_embeds[index][i].data();
	}
	return res;
}

void update_node_embed(int index, int node, int color)
{
	auto &curr_graph = graphs[index];
	auto &node_embed = node_embeds[index];
	for (int edge = curr_graph.row_ptr[node]; edge < curr_graph.row_ptr[node + 1]; edge++)
	{
		int adj = curr_graph.col_ind[edge];
		node_embed[adj][6] += 1;
		if (color > node_embed[adj][7])
		{
			node_embed[adj][7] = color;
		}
	}
	node_embed[node][8] = 1;
}

// extern "C" void update_node_embeddings(int *node, int *color) {
//   for(int i = 0; i < graphs.size(); i++) {
//     update_node_embed(i, node[i], color[i]);
//   }
// }

void update_adj_values(int index)
{
	std::unordered_set<int> adj_of_colored;
	std::unordered_set<int> colored_adj_of_colored;
	auto &curr_graph = graphs[index];
	auto &graph_embed = graph_embeds[index];
	auto &color_arr = color_arrs[index];

	for (int v = 0; v < (int)color_arr.size(); v++)
	{
		if (color_arr[v] == -1)
			continue;
		for (int u = curr_graph.row_ptr[v]; u < curr_graph.row_ptr[v + 1]; u++)
		{
			int &adj = curr_graph.col_ind[u];
			if (color_arr[adj] != -1)
			{
				colored_adj_of_colored.insert(adj);
			}
			adj_of_colored.insert(adj);
		}
	}
	graph_embed[1] = adj_of_colored.size();
	graph_embed[2] = colored_adj_of_colored.size();
}

void update_graph_embed(int index)
{
	auto &curr_graph = graphs[index];
	auto &graph_embed = graph_embeds[index];
	auto &color_arr = color_arrs[index];

	double degree_mean = (double)curr_graph.row_ptr.back() / curr_graph.num_nodes;
	double uncolored_degree_sq_sum = 0;
	double colored_degree_sq_sum = 0;
	int colored_degree_sum = 0;
	int colored_size = 0;
	int uncolored_degree_sum = 0;
	int uncolored_size = 0;
	graph_embed[7] = 0;
	graph_embed[8] = 0;
	graph_embed[9] = 0;
	graph_embed[10] = 0;
	graph_embed[11] = 0; // mean degree of colored nodes
	graph_embed[12] = 0;
	graph_embed[13] = 0;
	graph_embed[14] = 0;
	graph_embed[3] = 0;
	graph_embed[5] = 0;

	for (int v = 0; v < curr_graph.num_nodes; v++)
	{
		int degree = curr_graph.row_ptr[v + 1] - curr_graph.row_ptr[v];
		if (color_arr[v] != -1)
		{
			if (degree > degree_mean)
			{
				graph_embed[7]++;
			}
			else
			{
				graph_embed[8]++;
			}
			colored_degree_sq_sum += degree * degree;
			colored_degree_sum += degree;
			colored_size++;
			graph_embed[3] += node_embeds[index][v][3];
		}
		else if (color_arr[v] == -1)
		{
			if (degree > degree_mean)
			{
				graph_embed[9]++;
			}
			else
			{
				graph_embed[10]++;
			}
			uncolored_degree_sq_sum += degree * degree;
			uncolored_degree_sum += degree;
			uncolored_size++;
			graph_embed[5] += node_embeds[index][v][3];
		}
	}
	graph_embed[4] = colored_size;
	graph_embed[6] = uncolored_size;
	graph_embed[11] = (double)colored_degree_sq_sum / colored_size;
	graph_embed[12] = (double)uncolored_degree_sq_sum / uncolored_size;
	graph_embed[13] = (double)colored_degree_sq_sum / curr_graph.num_nodes - (degree_mean * degree_mean);
	graph_embed[14] = (double)uncolored_degree_sq_sum / curr_graph.num_nodes - (degree_mean * degree_mean);
	update_adj_values(index);
}

extern "C" void update_graph_embeddings()
{
	for (unsigned int i = 0; i < graphs.size(); i++)
	{
		update_graph_embed(i);
	}
}

void color_graph(int index, int node, int &color)
{
	auto &curr_graph = graphs[index];
	auto &color_arr = color_arrs[index];
	int n = node;

	// for(unsigned int i = 0; i < curr_graph.row_ptr.size() - 1; i++) {
	//   cout << "For node: " << i << " Adjs: ";
	//   for(int edge = curr_graph.row_ptr[i]; edge < curr_graph.row_ptr[i + 1]; edge++) {
	//     cout << curr_graph.col_ind[edge] << ", ";
	//   }
	//   cout << endl;
	// }

	std::vector<int> forbid_arr(curr_graph.num_nodes, -1);
	for (int edge = curr_graph.row_ptr[n]; edge < curr_graph.row_ptr[n + 1]; edge++)
	{
		int &adj = curr_graph.col_ind[edge];
		if (color_arr[adj] != -1)
			forbid_arr[color_arr[adj]] = n;
	}
	for (; color < curr_graph.num_nodes; color++)
	{
		if (forbid_arr[color] != n)
		{
			color_arr[n] = color;
			break;
		}
	}

	// cout << "Selected: " << color << endl;
}

extern "C" int *color_batch(int *nodes, int *size)
{
	vector<int> colors(graphs.size(), 0);
	*size = graphs.size();
	for (unsigned int i = 0; i < graphs.size(); i++)
	{
		color_graph(i, *(nodes + i), colors[i]);
		update_node_embed(i, *(nodes + i), colors[i]);
	}
	int* res = new int[*size];
	for(int a = 0; a < *size; a++) {
		res[a] = colors[a];
	}
	return res;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
// extern "C" float** initialize_graph_embeddings_for_batch(int * rows, int * cols){
//
//     *rows = graphs.size();
//
//     float** res = new float*[graphs.size()];
//     for(int i=0; i<(int)graphs.size(); i++){
//         vector<float> vec;
//         initialize_graph_state(graphs[i], vec);
//         res[i] = new float[vec.size()];
//         for(int j=0; j < (int)vec.size(); j++){
//             res[i][j] = vec[j];
//         }
//         *cols = vec.size();
//     }
//
//     /*
//     for(int i=0; i<*rows; i++){
//         for(int j=0; j < *cols; j++){
//             cout << res[i][j] << " ";
//         }
//         cout << endl;
//     }
//     */
//     return res;
// }
