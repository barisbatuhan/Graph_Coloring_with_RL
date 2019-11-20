#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <numeric> // for accumulate
#include <string>
#include<dirent.h>

using namespace std;

bool descending(const pair<int,float> & left, const pair<int,float> & right){
  return left.second>right.second;
}
bool ascending(const pair<int,float> & left,const pair<int,float> & right){
  return left.second<right.second;
}

pair<int,float> add(const float & left,const pair<int,float> & right){
  return pair<int,float>(right.first, left + right.second);
}


bool isValid(const vector<int> & color_arr,const vector<int>& row_ptr,const vector<int> & col_ind, int num_nodes){
  for(int v=0; v < num_nodes; v++){ // for each node v
				for(int e = row_ptr[v]; e < row_ptr[v+1]; e++){
						const int & adj = col_ind[e]; // for each adjacent of v
						if(color_arr[adj] == color_arr[v]){ // if color of v equals its adjacent's color
								return false;
						}
				}
		}
		return true;
}


int graph_coloring(const vector<int> & row_ptr,const vector<int> & col_ind, const vector<pair<int, float> > & ordering, vector<int> & color_arr, int maxdegree, string type){
  color_arr.resize(ordering.size(), -1);
  int nofcolors=0;
  vector<int> forbid_arr(maxdegree+1, -1);
  for(int i=0; i<ordering.size(); i++){
      const int & node = ordering[i].first; //for each node in ordering
      for(int edge = row_ptr[node]; edge < row_ptr[node+1]; edge++){
          const int & adj = col_ind[edge];//for each adjacent node
          if(color_arr[adj] != -1){ // if it is already colored
            forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
          }
      }
      for(int color = 0; color < maxdegree; color++){ // greedily choose the smallest possible color
        if(forbid_arr[color] != node){
          color_arr[node] = color;
          if(nofcolors < color){
            nofcolors = color;
          }
          break;
        }
      }
  }
  nofcolors++;
  cout << "with " << type << " nof colors: " << nofcolors << endl;
  if(!isValid(color_arr, row_ptr, col_ind, ordering.size()) == true){
			cout << "ERROR" <<endl;
	}
  return nofcolors;
}


void clustering_coeff(int num_nodes,const vector<int> & row_ptr,const vector<int> & col_ind, vector<pair<int, float> > & ordering){
  for(int v = 0; v<num_nodes; v++){ // for each node
    int degree = row_ptr[v+1]-row_ptr[v];
    int possiblelinks = degree*(degree-1)/2;
    int noflinks = 0;
    unordered_set<int> set;
    for(int edge = row_ptr[v]; edge < row_ptr[v+1]; edge++){ // insert all of its neighbors
        int adj = col_ind[edge];
        set.insert(adj);
    }

    for(int edge = row_ptr[v]; edge < row_ptr[v+1]; edge++){
        const int & adj = col_ind[edge]; // for each neighbor of v
        for(int link = row_ptr[adj]; link < row_ptr[adj+1]; link++){
          int adj_neigh = col_ind[link];// for each neighbor of neighbor
          if(set.find(adj_neigh) != set.end()){ // count number of links
            noflinks++;
          }
        }
    }
    float coeff = possiblelinks != 0 ? (float)noflinks/possiblelinks:0;
    ordering[v] = make_pair(v, coeff);
  }
}


void bfs(int start_node, int num_nodes, const vector<int> & row_ptr,const vector<int> & col_ind, vector<int> & distance_arr){
  distance_arr.assign(num_nodes, -1); // every node is unvisited
  distance_arr[start_node] = 0; // distance from a node to itself is 0
  queue<int> frontier; // FIFO queue
  frontier.push(start_node);
  int dist = 1; // initial distance
  do{
    int & front = frontier.front();
    for(int edge = row_ptr[front]; edge < row_ptr[front+1]; edge++){ // for each adjacent of front
      const int & adj = col_ind[edge];
      if(distance_arr[adj] == -1){ // if it is not visited
        distance_arr[adj] = dist; // assign corresponding distance
        frontier.push(adj); // add it to the frontier
      }
    }
    dist++; // next frontier will be further
    frontier.pop();
  }while(!frontier.empty());
  /*
  for(auto & x: distance_arr){
    cout << x << endl;
  }
  */
}


void closeness_centrality(int num_nodes,const vector<int> & row_ptr,const vector<int> & col_ind, vector<pair<int, float> > & ordering){
  vector<int> dist_arr;
  for(int v=0; v<num_nodes; v++){
    bfs(v, num_nodes, row_ptr, col_ind, dist_arr); // take distance array for node v
    int sum_of_dist = std::accumulate(dist_arr.begin(), dist_arr.end(), 0); // sum of d(v,x) for all x in the graph
    float coeff = sum_of_dist > 0 ? (float)num_nodes/sum_of_dist:0; // if coefficient is negative(meaning that graph is not connected) assign to 0
    ordering[v] = make_pair(v, coeff);
  }
}

void degree_order(int num_nodes,const vector<int> & row_ptr,const vector<int> & col_ind, vector<pair<int, float> > & ordering){
  for(int v=0; v<num_nodes; v++){
    ordering[v] = make_pair(v,row_ptr[v+1]-row_ptr[v]);
  }
}

void degree_2_order(int num_nodes,const vector<int> & row_ptr,const vector<int> & col_ind, vector<pair<int, float> > & ordering){
  int degree_2;
  for(int v=0; v<num_nodes; v++){
    degree_2 = 0;
    for(int edge = row_ptr[v]; edge<row_ptr[v+1]; edge++){
      int adj = col_ind[edge];
      degree_2 += row_ptr[adj+1]-row_ptr[adj];
    }
    ordering[v] = make_pair(v,degree_2);
  }
}

void degree_3_order(int num_nodes,const vector<int> & row_ptr,const vector<int> & col_ind, vector<pair<int, float> > & ordering){
  int degree_3;
  for(int v=0; v<num_nodes; v++){
    degree_3 = 0;
    for(int edge = row_ptr[v]; edge<row_ptr[v+1]; edge++){
      int adj = col_ind[edge];
      for(int e_adj=row_ptr[adj]; e_adj < row_ptr[adj+1]; e_adj++){
        int adj_2 = col_ind[e_adj];
        degree_3 += row_ptr[adj_2+1]-row_ptr[adj_2];
      }
    }
    ordering[v] = make_pair(v,degree_3);
  }
}

void page_rank(int num_nodes,const vector<int> & row_ptr,const vector<int> & col_ind, vector<pair<int, float> > & ordering, int iter=100, float alpha = 0.85){
  for(int i=0; i<num_nodes; i++){ordering[i] = make_pair(i, (float)1/num_nodes);} // initially likelyhoods are uniformly distributed
  vector<float> offset(num_nodes, 0.0);
  for(int i=0; i<iter; i++){ // on each iteration (required for convergence)
    vector<pair<int, float> > copy_ordering(num_nodes, pair<int,float>(0, 0.0));
    for(int j = 0; j < num_nodes; j++){
      copy_ordering[j].first = j;
    }
    for(int v=0; v<num_nodes; v++){ // for each node v
      // assign total page ranks %85
      float & pr_v = copy_ordering[v].second; // update page rank of v by looking its in-degree nodes
      for(int edge = row_ptr[v]; edge<row_ptr[v+1]; edge++){ // for each in degree neighbor of v (since graph is symmetric in or out degree does not matter)
        const int & adj = col_ind[edge];
        float & pr_adj = ordering[adj].second;
        int degree_adj = row_ptr[adj+1]-row_ptr[adj];
        pr_v += pr_adj/degree_adj; // page_rank_of_v <- (page_rank_of_v + page_rank_of_neighbor/out_degree_of_neighbor)
      }
      // distribute %15 evenly
      float dist_value = pr_v * (1-alpha) / (num_nodes-1);
      pr_v *= alpha;
      for_each(offset.begin(), offset.end(), [dist_value](float& d) { d+=dist_value;});
      pr_v -= dist_value;
    }
    transform(offset.begin(), offset.end(), copy_ordering.begin(), ordering.begin(), add);
    offset.assign(num_nodes, 0);
  }
}


int read_graph(string & fname, int & num_nodes, int &num_edges, vector<int> & row_ptr, vector<int> & col_ind){
  ifstream input(fname.c_str());
  if(input.fail()){
      return -1;
  }
  // read graph
  string line = "%";
  while(line.find("%") != string::npos){
    getline(input, line);
  }
  istringstream ss(line);
  ss >> num_nodes >> num_nodes >> num_edges;
  int v1, v2;
  double weight;
  vector<vector<int>> adj_list(num_nodes);
  for(int i=0; i<num_edges; i++){
      getline(input, line);
      istringstream inp(line);
      inp>>v1>>v2;
      v1--; // make it 0 based
      v2--;
      if(v1!=v2){
        adj_list[v1].push_back(v2); // add the edge v1->v2
    		adj_list[v2].push_back(v1); // add the edge v2->v1
      }
  }
	row_ptr=vector<int>(num_nodes+1);
	col_ind=vector<int>(2*num_edges);
	row_ptr[0] = 0;
	int index = 0;
	for (int v = 0; v<num_nodes; v++) {
		row_ptr[v+1] = adj_list[v].size(); // assign number of edges going from node v
		for (int i = 0; i<(int)adj_list[v].size(); i++) {
			col_ind[index] = adj_list[v][i]; // put all edges in order wrt row_ptr
			index++;
		}
	}
	for (int v = 1; v<num_nodes + 1; v++) {  // cumulative sum
		row_ptr[v] += row_ptr[v - 1];
	}
  cout << "nof nodes " << num_nodes << endl;
  cout << "nof edges " << num_edges << endl;

  return 0;
}


int main(int argc, char** argv ){

  const char* path = argv[1];
  ofstream out;
  out.open("metric_evaluation1.csv");
  out << "Graph Name, Num_nodes, Num_edges, DegreeOrderDesc, DegreeOrderAsc, Random, ClusteringCoeffDesc,ClusteringCoeffAsc, ClosenessCentralityDesc, ClosenessCentralityAsc, PageRankDesc, PageRankDesc "<<endl;

  DIR *pDIR;
  struct dirent *entry;
  if( pDIR=opendir(path) ){
      while(entry = readdir(pDIR)){
        cout << entry->d_name << endl;
        string fname = path+ ((string)entry->d_name);
        if(fname.at(fname.length()-1) == '.'){
          continue;
        }
        cout << fname << endl;
        int num_nodes, num_edges;
        vector<int> row_ptr, col_ind;
        if(read_graph(fname, num_nodes, num_edges, row_ptr, col_ind)==-1){
          cerr << "error reading graph" << endl;
          return 0;
        }
        // initializations
        vector<pair<int,float> > order(num_nodes);
        vector<int> color_arr;
        out << entry->d_name << "," << num_nodes << "," << num_edges << ",";
        // test metrics
        degree_order(num_nodes, row_ptr, col_ind, order);
        sort(order.begin(), order.end(), descending);
        int maxdegree = order[0].second;

        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "descending degree ordering") << ",";

        sort(order.begin(), order.end(), ascending);
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "ascending degree ordering") << ",";

        random_shuffle(order.begin(), order.end());
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "random ordering") << ",";

        clustering_coeff(num_nodes, row_ptr, col_ind, order);
        sort(order.begin(), order.end(), descending);
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "descending clustering coeff ordering") << ",";

        sort(order.begin(), order.end(), ascending);
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "ascending clustering coeff ordering") << ",";

        closeness_centrality(num_nodes, row_ptr, col_ind, order);
        sort(order.begin(), order.end(), descending);
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "descending closeness centrality ordering") << ",";

        sort(order.begin(), order.end(), ascending);
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "ascending closeness centrality ordering") << ",";

        page_rank(num_nodes, row_ptr, col_ind, order);
        sort(order.begin(), order.end(), descending);
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "descending page rank ordering") << ",";

        sort(order.begin(), order.end(), ascending);
        out << graph_coloring(row_ptr, col_ind, order, color_arr,  maxdegree, "ascending page rank ordering");
        out << endl;

      }
      closedir(pDIR);
  }
  out.close();
  /*
  for(auto & x: order){
    cout << x.first << " " << x.second <<  endl;
  }
  */

  /*
  for(auto & x: color_arr){
    cout << x << endl;
  }
  */

  return 0;
}
