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

int read_family(string &fname, string &family)
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
        if(line.find("kind:") != string::npos) {
            family = line.substr(8);
            cout << fname << " - " << family << endl;
            break;
        }
	}
    input.close();
}

int main() {
    vector<string> locations = {"./../../Matrices/large/"};
    vector<string> files;
    get_filenames(files, locations);

    vector<pair<string, string>> graph_info;

    for (int i = 0; i < files.size(); i++)
    {
        string family;
        read_family(files[i], family);
        graph_info.push_back({files[i], family});
    }

    ofstream output("families_large.csv");
    output << "Name,Family" << endl;
    for(int i = 0; i < graph_info.size(); i++) {
        output << graph_info[i].first << "," << graph_info[i].second << endl;
    }
    return 0;
}