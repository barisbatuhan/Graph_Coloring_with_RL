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

int main()
{
    // sample main
    vector<string> locations = {"./../Matrices/small/"};
    vector<string> files;
    get_filenames(files, locations);
    int total1 = 0, total2 = 0;
    for (int i = 0; i < files.size(); i++)
    {
        int num_nodes, num_edges;
        vector<int> row_ptr, col_ind;
        read_graphs(files[i], num_nodes, num_edges, row_ptr, col_ind);

        vector<pair<int, float>> order(num_nodes);
        closeness_centrality(num_nodes, row_ptr, col_ind, order);
        int color1 = graph_1d_coloring(row_ptr, col_ind, order);
        int color2 = dynamic_1d_coloring(num_nodes, row_ptr, col_ind, order);
        total1 += color1;
        total2 += color2;
        if(color1 > color2)
            cout << files[i] << " - " << color1  << " - " << color2 << endl;
        else if(color1 < color2)
            cout << "Reverse !!! - " << files[i] << " - " << color1  << " - " << color2 << endl;
    }
    cout << endl << total1 << " - " << total2 << endl;
    return 0;
}