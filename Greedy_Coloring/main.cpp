#include "./API/graph.h"

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

void print_similarities(vector<vector<int>> &curr_similarities) {                                      
    cout << "----------------------------------------------------------" << endl;
    cout << "    \t | deg1\t | deg2\t | deg3\t | clos\t | coef\t | pran\t |" << endl;
    cout << "----------------------------------------------------------" << endl;
    cout << " deg1\t | " << curr_similarities[0][0] << "\t | " << curr_similarities[0][1] << "\t | " << curr_similarities[0][2]<< "\t | " << curr_similarities[0][3]
         << "\t | " << curr_similarities[0][4] << "\t | " << curr_similarities[0][5] << "\t | "<< endl;
    cout << " deg2\t | " << curr_similarities[1][0] << "\t | " << curr_similarities[1][1] << "\t | " << curr_similarities[1][2]<< "\t | " << curr_similarities[1][3]
         << "\t | " << curr_similarities[1][4] << "\t | " << curr_similarities[1][5] << "\t | "<< endl;
    cout << " deg3\t | " << curr_similarities[2][0] << "\t | " << curr_similarities[2][1] << "\t | " << curr_similarities[2][2]<< "\t | " << curr_similarities[2][3]
         << "\t | " << curr_similarities[2][4] << "\t | " << curr_similarities[2][5] << "\t | "<< endl;
    cout << " clos\t | " << curr_similarities[3][0] << "\t | " << curr_similarities[3][1] << "\t | " << curr_similarities[3][2]<< "\t | " << curr_similarities[3][3]
         << "\t | " << curr_similarities[3][4] << "\t | " << curr_similarities[3][5] << "\t | "<< endl;
    cout << " coef\t | " << curr_similarities[4][0] << "\t | " << curr_similarities[4][1] << "\t | " << curr_similarities[4][2]<< "\t | " << curr_similarities[4][3]
         << "\t | " << curr_similarities[4][4] << "\t | " << curr_similarities[4][5] << "\t | "<< endl;
    cout << " pran\t | " << curr_similarities[5][0] << "\t | " << curr_similarities[5][1] << "\t | " << curr_similarities[5][2]<< "\t | " << curr_similarities[5][3]
         << "\t | " << curr_similarities[5][4] << "\t | " << curr_similarities[5][5] << "\t | "<< endl;
    cout << "----------------------------------------------------------" << endl;
}

void get_similarities(vector<vector<pair<int, float>>> &orders, vector<vector<int>> &similarities) {
    vector<vector<int>> curr_similarities(6, vector<int>(6, 0));
    for(int i = 0; i < orders.size(); i++) {
        for(int row = 0; row < orders[i].size(); row++) {
            for(int j = i; j < orders.size(); j++) {
                if(orders[i][row].first == orders[j][row].first) {
                    curr_similarities[i][j]++;
                     similarities[i][j]++;
                    if(i != j) {
                        curr_similarities[j][i]++;
                        similarities[j][i]++;
                    } 
                }
            }    
        }    
    }
    print_similarities(curr_similarities);
}

int main() {
    vector<string> locations = {"./../Matrices/large/"};
    vector<string> files;
    get_filenames(files, locations);

    /**
     * For holding similarities, the order will be always as follows:
     * 1) degree 1
     * 2) degree 2
     * 3) degree 3
     * 4) closeness centrality
     * 5) clustering coefficient
     * 6) page rank
     */
    vector<vector<int>> similarities(6, vector<int>(6, 0));

    cout << "files,deg1,sentiment" << endl;
    #pragma omp parallel for num_threads(8) schedule(guided)
    for (int i = 10; i < files.size(); i++)
    {   
        vector<int> row_ptr, col_ind;
        int num_nodes, num_edges;
        read_graphs(files[i], num_nodes, num_edges, row_ptr, col_ind);
        
        vector<vector<pair<int, float>>> orders(6, vector<pair<int, float>>(num_nodes));
		degree_order(num_nodes, row_ptr, col_ind, orders[0]);
		// degree_2_order(num_nodes, row_ptr, col_ind, orders[1]);
		// degree_3_order(num_nodes, row_ptr, col_ind, orders[2]);
		// closeness_centrality(num_nodes, row_ptr, col_ind, orders[3]);
		// clustering_coeff(num_nodes, row_ptr, col_ind, orders[4]);
		// page_rank(num_nodes, row_ptr, col_ind, orders[5]);

        // normalize(orders);

        // // sort values to create permutations
		// for (int i = 0; i < orders.size(); i++)
		// {
		// 	sort(orders[i].begin(), orders[i].end(), descending);
		// }

        // get_similarities(orders, similarities);

        int colors = saturation_1d_coloring(num_nodes, row_ptr, col_ind, orders[0]);
        sort(orders[0].begin(), orders[0].end(), descending);
        int deg_colors = graph_1d_coloring(row_ptr, col_ind, orders[0]);

        cout << files[i] << "," << deg_colors << "," << colors  << endl;

        // cout << "For the graph " << files[i] << endl
        //      << " ----- Sentiment colors: " << colors << endl
        //      << " ----- Closeness colors: " << clos_colors << endl;
    }

    // print_similarities(similarities);

    return 0;
}