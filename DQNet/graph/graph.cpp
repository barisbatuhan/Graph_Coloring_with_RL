#include "graph.h"
// CONSTRUCTORS
Graph::Graph()
{
}
/**
 * Graph constructor:
 * Gets a filename and reads the graph from the file
 * Each graph is considered as undirected and unweighted
 * Designed for Suite-Sparse Matrix Collection Matrix Market files
 * */
Graph::Graph(std::string fname)
{
    max_degree = 0;
    std::ifstream input(fname.c_str());
    if (input.fail())
    {
        throw "No file is found in the current path!";
    }
    else
    {
        relative_path = fname;
    }
    // read graph
    std::string line = "%";
    family = "%";
    while (line.find("%") != std::string::npos)
    {
        getline(input, line);
        if (family == "%" && line.find("kind:") != std::string::npos)
        {
            family = line.substr(8);
            family = family.substr(0, family.length() - 1);
        }
    }

    std::istringstream ss(line);
    ss >> num_nodes >> num_nodes >> num_edges;
    int v1, v2;

    std::vector<std::vector<int>> adj_list(num_nodes);
    for (int i = 0; i < num_edges; i++)
    {
        getline(input, line);
        std::istringstream inp(line);
        inp >> v1 >> v2;
        v1--; // make it 0 based
        v2--;

        if (v1 != v2)
        {
            adj_list[v1].push_back(v2); // add the edge v1->v2
            adj_list[v2].push_back(v1); // add the edge v2->v1
        }
    }

    row_ptr = std::vector<int>(num_nodes + 1);
    col_ind = std::vector<int>(2 * num_edges);
    row_ptr[0] = 0;
    int index = 0;
    for (int v = 0; v < num_nodes; v++)
    {
        row_ptr[v + 1] = adj_list[v].size(); // assign number of edges going from node v
        if(row_ptr[v + 1] > max_degree) {
            max_degree = row_ptr[v + 1];
        }
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
}

/**
 * Graph constructor:
 * Gets node count and edge count and generates a random undirected unweighted graph
 * with these node and edge numbers.
 * */
Graph::Graph(int node_cnt, int edge_cnt)
{
    max_degree = 0;
    std::vector<std::vector<bool>> adj_list(node_cnt, std::vector<bool>(node_cnt, false));
    num_nodes = node_cnt;
    num_edges = edge_cnt;
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
    row_ptr = std::vector<int>(node_cnt + 1);
    col_ind = std::vector<int>(2 * edge_cnt);
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
        if(adj_cnt > max_degree) {
            max_degree = adj_cnt;
        }
        row_ptr[v + 1] = row_ptr[v] + adj_cnt; // assign number of edges going from node v
    }
}

// COLORING METHODS

/**
 * Applies distance 1 coloring on the graph greedily the with first fitting color. 
 * An ordering is passed to determine which nodes to color first.
 * */
int Graph::color_1d(const std::vector<std::pair<int, float>> &ordering)
{
    std::vector<int> color_arr(ordering.size(), -1);
    int nofcolors = 0;
    std::vector<int> forbid_arr(row_ptr.size() - 1, -1);
    bool hasEdge = false;
    for (int i = 0; i < (int)ordering.size(); i++)
    {
        const int &node = ordering[i].first; // for each node in ordering
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++)
        {
            hasEdge = true;
            const int &adj = col_ind[edge]; // for each adjacent node
            if (color_arr[adj] != -1)
            {                                      // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
        }
        for (int color = 0; color < (int)ordering.size(); color++)
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
    // check is the coloring is valid
    // if (!is_valid_1d(color_arr) == true) cout << "ERROR" << endl;
    return nofcolors;
}


/**
 * Applies distance 2 coloring on the graph greedily the with first fitting color. 
 * An ordering is passed to determine which nodes to color first.
 * */
int Graph::color_2d(const std::vector<std::pair<int, float>> &ordering)
{
    std::vector<int> color_arr(ordering.size(), -1);
    int nofcolors = 0;
    std::vector<int> forbid_arr(row_ptr.size() - 1, -1);
    bool hasEdge = false;
    for (int i = 0; i < (int)ordering.size(); i++)
    {
        const int &node = ordering[i].first; //for each node in ordering
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++)
        {
            hasEdge = true;
            const int &adj = col_ind[edge]; //for each adjacent node
            if (color_arr[adj] != -1)
            {                                      // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
            for (int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj + 1]; edge_2++)
            {
                const int &adj_neigh = col_ind[edge_2];
                if (color_arr[adj_neigh] != -1)
                {                                            // if it is already colored
                    forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
                }
            }
        }
        for (int color = 0; color < (int)ordering.size(); color++)
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
    // checkis if the coloring is valid
    // if (!is_valid_2d(color_arr) == true) cout << "ERROR" << endl;
    return nofcolors;
}


/**
 * Applies distance 1 coloring on the graph greedily the with first fitting color. 
 * At each step, chooses the node for coloring, which has most number of colored neighbors. 
 * If start is not randomized, then starts with the node having maximum neighbors.
 * */
int Graph::color_dynamic_1d(bool random_start){
    int start=-1;
    if(random_start){
        start = rand()%num_nodes;
    }
    else{ // highest degree1
        int max_degree = -1;
        for(int v=0; v<num_nodes; v++){
            int degree = row_ptr[v+1]-row_ptr[v];
            if(max_degree < degree){
                start = v;
                max_degree = degree;
            }
        }
    }

    Heap orderHeap;
    vector<pair<int, float>> zero(num_nodes, pair<int,float>(0,0));
    for(int i=0; i<(int)zero.size(); i++){
        zero[i].first = i;
    }
    zero[start].second = 1;
    orderHeap.buildHeap(zero);
    int nofcolors = 0, colored = 0;
    vector<int> forbid_arr(num_nodes-1, -1);
    vector<int> color_arr(num_nodes, -1);

    while(colored < num_nodes){
        auto deleted = orderHeap.deleteMax(); // structured binding
        auto node = deleted.first;
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++) {
            const int & adj = col_ind[edge]; //for each adjacent node
            int index = orderHeap.locArr[adj];
            if(index != -1){ // if it is not colored
                orderHeap.arr[index].second++;
                orderHeap.percolateUp(index);
            }
            if (color_arr[adj] != -1) { // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
        }
        for (int color = 0; color < num_nodes; color++) { // greedily choose the smallest possible color
            if (forbid_arr[color] != node) {
                color_arr[node] = color;
                if (nofcolors < color) {
                    nofcolors = color;
                }
                break;
            }
        }
        colored++;
    }
    nofcolors++; // coloring was zero based so increment
    
    // if (!is_valid_1d(color_arr)) cout << "ERROR" << endl;
    return nofcolors;
}

/**
 * Applies distance 2 coloring on the graph greedily the with first fitting color. 
 * At each step, chooses the node for coloring, which has most number of colored neighbors. 
 * If start is not randomized, then starts with the node having maximum neighbors.
 * */
int Graph::color_dynamic_2d(bool random_start){
    int start=-1;
    if(random_start){
        start = rand()%num_nodes;
    }
    else{ // highest degree1
        int max_degree = -1;
        for(int v=0; v<num_nodes; v++){
            int degree = row_ptr[v+1]-row_ptr[v];
            if(max_degree < degree){
                start = v;
                max_degree = degree;
            }
        }
    }

    Heap orderHeap;
    vector<pair<int, float>> zero(num_nodes, pair<int,float>(0,0));
    for(int i=0; i<(int)zero.size(); i++){
        zero[i].first = i;
    }
    zero[start].second = 1;
    orderHeap.buildHeap(zero);
    int nofcolors = 0, colored = 0;
    vector<int> forbid_arr(num_nodes-1, -1);
    vector<int> color_arr(num_nodes, -1);

    while(colored < num_nodes){
        auto deleted = orderHeap.deleteMax(); // structured binding
        auto node = deleted.first;
        for (int edge = row_ptr[node]; edge < row_ptr[node + 1]; edge++) {
            const int & adj = col_ind[edge]; //for each adjacent node
            int index = orderHeap.locArr[adj];
            if(index != -1){ // if it is not colored
                orderHeap.arr[index].second++;
                orderHeap.percolateUp(index);
            }
            if (color_arr[adj] != -1) { // if it is already colored
                forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
            }
            for(int edge_2 = row_ptr[adj]; edge_2 < row_ptr[adj+1]; edge_2++){
                const int & adj_neigh = col_ind[edge_2];
                index = orderHeap.locArr[adj_neigh];
                if(index != -1){ // if it is not colored
                    orderHeap.arr[index].second ++;
                    orderHeap.percolateUp(index);
                }
                if (color_arr[adj_neigh] != -1) { // if it is already colored
                    forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
                }
            }
        }
        for (int color = 0; color < num_nodes; color++) { // greedily choose the smallest possible color
            if (forbid_arr[color] != node) {
                color_arr[node] = color;
                if (nofcolors < color) {
                    nofcolors = color;
                }
                break;
            }
        }
        colored++;
    }

    nofcolors++; // coloring was zero based so increment
    if (!is_valid_2d(color_arr)){
        cout << "ERROR" << endl;
    }
    return nofcolors;
}


/**
 * Applies distance 1 coloring on the graph greedily the with first fitting color. 
 * At each step, chooses the node for coloring, which has most number of different colored neighbors (saturation). 
 * Takes a spare order to break the tie, where nodes to select have the same amount of different 
 * colored neighbors.
 * */
int Graph::color_saturation_1d(std::vector<std::pair<int, float>> &spare_order)
{
    std::vector<std::unordered_set<int>> color_infos(num_nodes); // for each node, set of colors of node's neighbors hold
    std::vector<std::pair<int, int>> node_values(2, {1, 0}); // holds 2 nodes with max different colored neighbors
    std::vector<bool> node_colored(num_nodes, false);

    std::vector<int> color_arr(num_nodes, -1);
    int nofcolors = 0;
    bool hasEdge = false;
    int colored_num = 0;

    do
    {
        int i = 0;
        std::pair<int, int> &node = node_values[i];
        std::pair<int, int> &next_node = node_values[i + 1];
        if (next_node.first != -1 && next_node.second == node.second) // checks if the second node has the same value with first
        {
            if (node_colored[next_node.first] != true)
            {
                if (spare_order[next_node.first] > spare_order[node.first]) // if so, takes the one with higher spare order value
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

        // new color is added to the set of adjacent neighbors
        for (int edge = row_ptr[node.first]; edge < row_ptr[node.first + 1]; edge++)
        {
            hasEdge = true;
            const int &adj = col_ind[edge];
            color_infos[adj].insert(color);
        }

        node_values[0].first = -1;
        node_values[0].second = -9999;
        node_values[1].first = -1;
        node_values[1].second = -9999;

        for (int i = 0; i < num_nodes; i++) // nodes with the highest saturation are selected
        {
            if (node_colored[i])
            {
                continue;
            }
            int curr_colors = color_infos[i].size();
            if (curr_colors > node_values[0].second)
            {
                node_values[1].second = node_values[0].second;
                node_values[1].first = node_values[0].first;
                node_values[0].second = curr_colors;
                node_values[0].first = i;
            }
            else if (curr_colors > node_values[1].second)
            {
                node_values[1].second = curr_colors;
                node_values[1].first = i;
            }
        }
    } while (colored_num < num_nodes);

    if (hasEdge)
    {
        nofcolors++;
    }
    // validity of coloring is checked
    // if (!is_valid_1d(color_arr) == true) cout << "ERROR" << endl;
    return nofcolors;
}


/**
 * Applies distance 2 coloring on the graph greedily the with first fitting color. 
 * At each step, chooses the node for coloring, which has most number of different colored neighbors (saturation). 
 * Takes a spare order to break the tie, where nodes to select have the same amount of different 
 * colored neighbors.
 * */
int Graph::color_saturation_2d(std::vector<std::pair<int, float>> &spare_order)
{
    std::vector<std::unordered_set<int>> color_infos(num_nodes); // for each node, set of colors of node's neighbors hold
    std::vector<std::pair<int, int>> node_values(2, {1, 0}); // holds 2 nodes with max different colored neighbors
    std::vector<bool> node_colored(num_nodes, false);

    std::vector<int> color_arr(num_nodes, -1);
    int nofcolors = 0;
    bool hasEdge = false;
    int colored_num = 0;

    do
    {
        int i = 0;
        std::pair<int, int> &node = node_values[i];
        while (node_colored[node.first] == true)
        {
            node = node_values[++i];
        }
        std::pair<int, int> &next_node = node_values[i + 1];
        if (next_node.first != -1 && next_node.second == node.second) // checks if the second node has the same value with first
        {
            if (node_colored[next_node.first] != true)
            {
                if (spare_order[next_node.first] > spare_order[node.first]) // if so, takes the one with higher spare order value
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

        for (int i = 0; i < num_nodes; i++) // nodes with the highest saturation are selected
        {
            if (node_colored[i])
            {
                continue;
            }
            int curr_colors = color_infos[i].size();
            if (curr_colors > node_values[0].second)
            {
                node_values[1].second = node_values[0].second;
                node_values[1].first = node_values[0].first;
                node_values[0].second = curr_colors;
                node_values[0].first = i;
            }
            else if (curr_colors > node_values[1].second)
            {
                node_values[1].second = curr_colors;
                node_values[1].first = i;
            }
        }
    } while (colored_num < num_nodes);

    if (hasEdge)
    {
        nofcolors++;
    }
    // validity of coloring is checked
    // if (!is_valid_2d(color_arr) == true) cout << "ERROR" << endl;
    return nofcolors;
}

// COLORING VALIDITY CHECKERS
bool Graph::is_valid_1d(const std::vector<int> &color_arr)
{
    for (int v = 0; v < num_nodes; v++)
    { // for each node v
        if (color_arr[v] == -1)
        {
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

bool Graph::is_valid_2d(const std::vector<int> &color_arr)
{
    for (int v = 0; v < num_nodes; v++)
    { // for each node v
        if (color_arr[v] == -1)
        {
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
                if (adj_neigh == v)
                    continue;
                if (color_arr[adj_neigh] == color_arr[v])
                { // if color of v equals its adjacent's neighbor's color
                    return false;
                }
            }
        }
    }
    return true;
}

// ORDERING METHODS
/**
 * Writes the number of neighbors for each node to ordering vector
 * First element of the "ordering" pair is the node number, second is the degree value
 * */
void Graph::degree_order(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        ordering[v] = std::make_pair(v, row_ptr[v + 1] - row_ptr[v]);
    }
}

/**
 * Writes the number of neighbors for each node to ordering vector
 * */
void Graph::degree_order(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        ordering[v] = row_ptr[v + 1] - row_ptr[v];
    }
}

/**
 * Writes the number of nodes with distance <= 2 for each node to ordering vector
 * First element of the "ordering" pair is the node number, second is the degree-2 value
 * */
void Graph::degree_2_order(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 2); // take distance array for node v until distance 2
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2)
            {
                count++;
            }
        }
        ordering[v] = std::make_pair(v, count);
    }
}


/**
 * Writes the number of nodes with distance <= 2 for each node to ordering vector
 * */
void Graph::degree_2_order(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 2); // take distance array for node v until distance 2
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2)
            {
                count++;
            }
        }
        ordering[v] = count;
    }
}


/**
 * Writes the number of nodes with distance <= 3 for each node to ordering vector
 * First element of the "ordering" pair is the node number, second is the degree-3 value
 * */
void Graph::degree_3_order(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 3); // take distance array for node v until distance 3
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2 || val == 3)
            {
                count++;
            }
        }
        ordering[v] = std::make_pair(v, count);
    }
}


/**
 * Writes the number of nodes with distance <= 3 for each node to ordering vector
 * */
void Graph::degree_3_order(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    {
        std::vector<int> dist_arr(num_nodes);
        bfs(v, dist_arr, 3); // take distance array for node v until distance 3
        int count = 0, val;
        for (int i = 0; i < (int)dist_arr.size(); i++)
        {
            val = dist_arr[i];
            if (val == 1 || val == 2 || val == 3)
            {
                count++;
            }
        }
        ordering[v] = count;
    }
}


/**
 * Calculates closeness centrality for the graph
 * Closeness formula is: number of nodes / sum of all distances from selected node
 * First element of the "ordering" pair is the node number, second is the closeness value
 * */
void Graph::closeness_centrality(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    std::vector<int> dist_arr;
    for (int v = 0; v < num_nodes; v++)
    {
        bfs(v, dist_arr);                                                       // take distance array for node v
        int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
        float coeff = sum_of_dist > 0 ? (float)num_nodes / sum_of_dist : 0;     // if coefficient is negative(meaning that graph is not connected) assign to 0
        ordering[v] = std::make_pair(v, coeff);
    }
}


/**
 * Calculates closeness centrality for the graph
 * Closeness formula is: number of nodes / sum of all distances from selected node
 * */
void Graph::closeness_centrality(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    std::vector<int> dist_arr;
    for (int v = 0; v < num_nodes; v++)
    {
        bfs(v, dist_arr);                                                       // take distance array for node v
        int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
        float coeff = sum_of_dist > 0 ? (float)num_nodes / sum_of_dist : 0;     // if coefficient is negative(meaning that graph is not connected) assign to 0
        ordering[v] = coeff;
    }
}


/**
 * Calculates an approximation for closeness centrality especially for big sized graphs
 * A size is taken as an input and size many BFS algorithms are run. Values are approximated
 * using these BFS results.
 * Closeness formula is: number of nodes / sum of all distances from selected node
 * First element of the "ordering" pair is the node number, second is the closeness value
 * */
void Graph::closeness_centrality_approx(std::vector<std::pair<int, float>> &ordering, int size)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    if (size > num_nodes)
    {
        size = num_nodes;
    }
    std::vector<std::vector<int>> dist_arr(size, std::vector<int>(num_nodes)); // holds distance array

    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < size; v++)
    {
        //srand(1);
        int start_node = rand() % num_nodes;
        bfs(start_node, dist_arr[v]); // take distance array for node v
    }
    #pragma omp barrier

    #pragma omp parallel for num_threads(32)
    for (int v = 0; v < num_nodes; v++)
    {
        int sum_of_dist = 0;
        for (int i = 0; i < size; i++)
        {
            sum_of_dist += dist_arr[i][v]; // calculates sum of distances for each node
        }
        // if coefficient is negative(meaning that graph is not connected) assign to 0
        float coeff = sum_of_dist > 0 ? (float)size / sum_of_dist : 0;
        ordering[v] = std::make_pair(v, coeff);
    }
}

/**
 * Calculates an approximation for closeness centrality especially for big sized graphs
 * A size is taken as an input and size many BFS algorithms are run. Values are approximated
 * using these BFS results.
 * Closeness formula is: number of nodes / sum of all distances from selected node
 * */
void Graph::closeness_centrality_approx(std::vector<float> &ordering, int size)
{
    ordering = std::vector<float>(num_nodes);
    if (size > num_nodes)
    {
        size = num_nodes;
    }
    std::vector<std::vector<int>> dist_arr(size, std::vector<int>(num_nodes));

    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < size; v++)
    {
        //srand(1);
        int start_node = rand() % num_nodes;
        bfs(start_node, dist_arr[v]); // take distance array for node v
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
        // if coefficient is negative(meaning that graph is not connected) assign to 0
        float coeff = sum_of_dist > 0 ? (float)size / sum_of_dist : 0; 
        ordering[v] = coeff;
    }
}


/**
 * Checks how many triangular node bingings each node has
 * First element of the "ordering" pair is the node number, second is the clustering value
 * */
void Graph::clustering_coeff(std::vector<std::pair<int, float>> &ordering)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    { // for each node
        int degree = row_ptr[v + 1] - row_ptr[v];
        int possiblelinks = degree * (degree - 1) / 2;
        int noflinks = 0;
        std::unordered_set<int> set;
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
        ordering[v] = std::make_pair(v, coeff);
    }
}

/**
 * Checks how many triangular node bingings each node has
 * */
void Graph::clustering_coeff(std::vector<float> &ordering)
{
    ordering = std::vector<float>(num_nodes);
    #pragma omp parallel for num_threads(32) schedule(dynamic)
    for (int v = 0; v < num_nodes; v++)
    { // for each node
        int degree = row_ptr[v + 1] - row_ptr[v];
        int possiblelinks = degree * (degree - 1) / 2;
        int noflinks = 0;
        std::unordered_set<int> set;
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
        ordering[v] = coeff;
    }
}


/**
 * Runs Google's PageRank algorithm implementation. Iteration count and alpha (update rate)
 * are taken as inputs
 * First element of the "ordering" pair is the node number, second is the PageRank value
 * */
void Graph::page_rank(std::vector<std::pair<int, float>> &ordering, int iter, float alpha)
{
    ordering = std::vector<std::pair<int, float>>(num_nodes);
    // distribute %15 evenly
    float dist_value = (1 - alpha) / num_nodes;
    for (int i = 0; i < num_nodes; i++)
    {
        ordering[i] = std::make_pair(i, (float)1 / num_nodes);
    }
    // initially likelyhoods are uniformly distributed
    for (int i = 0; i < iter; i++)
    {
        // on each iteration (required for convergence)
        std::vector<std::pair<int, float>> copy_ordering(num_nodes, std::pair<int, float>(0, dist_value));
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


/**
 * Runs Google's PageRank algorithm implementation. Iteration count and alpha (update rate)
 * are taken as inputs
 * */
void Graph::page_rank(std::vector<float> &ordering, int iter, float alpha)
{
    ordering = std::vector<float>(num_nodes);
    // distribute %15 evenly
    float dist_value = (1 - alpha) / num_nodes;
    for (int i = 0; i < num_nodes; i++)
    {
        ordering[i] = 1 / num_nodes;
    }
    // initially likelyhoods are uniformly distributed
    for (int i = 0; i < iter; i++)
    {
        // on each iteration (required for convergence)
        std::vector<float> copy_ordering(num_nodes, dist_value);
        for (int v = 0; v < num_nodes; v++)
        { // for each node v
            // assign total page ranks %85
            float &pr_v = copy_ordering[v]; // update page rank of v by looking its in-degree nodes
            for (int edge = row_ptr[v]; edge < row_ptr[v + 1]; edge++)
            {
                // for each in degree neighbor of v (since graph is symmetric in or out degree does not matter)
                const int &adj = col_ind[edge];
                float &pr_adj = ordering[adj];
                int degree_adj = row_ptr[adj + 1] - row_ptr[adj];
                pr_v += alpha * (pr_adj / degree_adj); // page_rank_of_v <- (page_rank_of_v + page_rank_of_neighbor/out_degree_of_neighbor)
            }
        }
        ordering = copy_ordering;
    }
}

// HELPER METHODS
/**
 * Implementatipon of BFS algorithm with queue logic.
 * */
void Graph::bfs(int start_node, std::vector<int> &distance_arr, int step_size)
{
    std::vector<int> frontier(num_nodes, -1);
    frontier[0] = start_node;
    int queuestart = 0, queueend = 1, frontsize = 0;
    distance_arr.assign(num_nodes, -1); // every node is unvisited
    distance_arr[start_node] = 0;       // distance from a node to itself is 0
    int dist = 1;                       // initial distance
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
                    distance_arr[adj] = dist;               // assign corresponding distance
                }
            }
        } while (queuestart < queueend);
        queueend += frontsize; // add the offset
        frontsize = 0;         // reset the offset
        dist++;                // next frontier will be further
    }
}

// PRINTER METHODS
void Graph::print_graph()
{
    std::cout << "Path: " << relative_path << std::endl
              << "Number of Nodes: " << num_nodes << std::endl
              << "Number of Edges: " << num_edges << std::endl
              << "Graph Family: " << family << std::endl;
}

// STATIC METHODS
/**
 * Standard normalization function for single ordering vector
 * */
void Graph::normal_params(std::vector<std::pair<int, float>> &order, float &mean, float &stdev)
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
    if (std::isnan(stdev) || stdev == 0)
        stdev = 0.001;
}


/**
 * Standard normalization function for collection of ordering vectors.
 * */
void Graph::normalize(std::vector<std::vector<std::pair<int, float>>> &orders, int num)
{
    if (num == -1)
    {
        num = orders.size();
    }
    float mean, stdev;
    for (int i = 0; i < num; i++)
    {
        normal_params(orders[i], mean, stdev);
        for_each(orders[i].begin(), orders[i].end(), [mean, stdev](std::pair<int, float> &x) { x.second = (x.second - mean) / stdev; });
    }
}

/**
 * Helper functions for sort and vector related operation
 * */

bool Graph::descending(const std::pair<int, float> &left, const std::pair<int, float> &right)
{
	if ((left.second > right.second) || (left.second == right.second && left.first < right.first))
	{
		return true;
	}
	return false;
}

bool Graph::ascending(const std::pair<int, float> &left, const std::pair<int, float> &right)
{
	if ((left.second < right.second) || (left.second == right.second && left.first < right.first))
	{
		return true;
	}
	return false;
}

std::pair<int, float> Graph::add(const float &left, const std::pair<int, float> &right)
{
	return std::pair<int, float>(right.first, left + right.second);
}
