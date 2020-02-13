#include "./../API/graph.h"

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

bool ascending_nodes(const pair<string, pair<int, float>> &left, const pair<string, pair<int, float>> &right)
{
    if (left.second.first < right.second.first)
        return true;
    else if (left.second.first == right.second.first)
        return left.second.second < right.second.second;
    else
        return false;
}

bool check_if_same_graphs(vector<int> &first, vector<int> &second)
{
    if (first.size() != second.size())
        return false;

    for (int i = 0; i < first.size(); i++)
    {
        if (first[i] != second[i])
            return false;
    }

    return true;
}

int read_vertex_edge_count(string &fname, int &num_nodes, int &num_edges)
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
    input.close();
}

int main()
{
    vector<string> locations = {"./../../Matrices/large/"};
    vector<string> files;
    get_filenames(files, locations);

    vector<pair<string, pair<int, int>>> graph_info;

    for (int i = 0; i < files.size(); i++)
    {
        int num_nodes, num_edges;
        read_vertex_edge_count(files[i], num_nodes, num_edges);
        graph_info.push_back({files[i], {num_nodes, num_edges}});
    }
    sort(graph_info.begin(), graph_info.end(), ascending_nodes);
    cout << "Sorted!!!" << endl;
    vector<string> same_graphs;

    for (int i = 0; i < graph_info.size() - 1; i++)
    {
        pair<string, pair<int, int>> &prev = graph_info[i];
        vector<int> prev_row_ptr, prev_col_ind;
        for (int j = i + 1; j < graph_info.size(); j++)
        {
            pair<string, pair<int, int>> &next = graph_info[j];
            if (prev.second.first == next.second.first && prev.second.second == next.second.second)
            {
                int num_nodes, num_edges;
                if(prev_row_ptr.size() == 0)
                    read_graphs(prev.first, num_nodes, num_edges, prev_row_ptr, prev_col_ind);
                vector<int> next_row_ptr, next_col_ind;
                read_graphs(next.first, num_nodes, num_edges, next_row_ptr, next_col_ind);

                if (check_if_same_graphs(prev_col_ind, next_col_ind))
                {
                    same_graphs.push_back(next.first);
                }
            }
            else
                break;
        }
    }

    unordered_set<string> same_files(same_graphs.begin(), same_graphs.end());
    for(const string &str: same_files) {
        cout << str << endl;
    } 

    cout << "\nProcess finished!!!" << endl;

    return 0;
}