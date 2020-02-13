#include "./API/graph.h"

int main()
{
    // sample main
    string path = "./../Matrices/small/494_bus.mtx";
    int num_nodes, num_edges;
    vector<int> row_ptr, col_ind;
    read_graphs(path, num_nodes, num_edges, row_ptr, col_ind);
    
    vector<vector<pair<int, float>>> order(2, vector<pair<int, float>>(num_nodes));
    degree_2_order(num_nodes, row_ptr, col_ind, order[0]);
    page_rank(num_nodes, row_ptr, col_ind, order[1]);
    
    vector<string> labels = {"DEG2", "PR"};
    write_to_csv(labels, order, "out");
    return 0;
}