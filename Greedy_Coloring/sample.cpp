#include "./API/Graph.hpp"
using namespace std;

int main()
{
    string path = "../Matrices/small/494_bus.mtx";
    Graph graph = Graph(path);
    vector<vector<pair<int, float>>> orderings(6);
    graph.print_graph();
    graph.degree_order(orderings[0]);
    graph.degree_2_order(orderings[1]);
    graph.degree_3_order(orderings[2]);
    graph.closeness_centrality(orderings[3]);
    graph.clustering_coeff(orderings[4]);
    graph.page_rank(orderings[5]);
    
    Graph::normalize(orderings);
    
    int index = 0;
    for(auto &order: orderings) 
    {
        int color_sat_1d = graph.color_saturation_1d(order);
        int color_sat_2d = graph.color_saturation_2d(order);
        
        sort(order.begin(), order.end(), Graph::descending);
        int color_1d = graph.color_1d(order);
        int color_2d = graph.color_2d(order);
        
        cout << "Index: " << index << endl
             << "--> 1d: " << color_1d << endl
             << "--> 2d: " << color_2d << endl
             << "--> sat 1d: " << color_sat_1d << endl
             << "--> sat 2d: " << color_sat_2d << endl << endl;
        index++;
    }
  
    return 0;
}

