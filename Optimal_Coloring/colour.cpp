#include <ilcplex/ilocplex.h>
ILOSTLBEGIN
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include<dirent.h>

using namespace std;

void usage()
{
  cerr << "USAGE: ./exec <filename>" << endl;
  exit(0);
}

int main(int argc, char**argv){
    if(argc != 2){
        usage();
    }
    const char* path = argv[1];
    ofstream out;
    out.open("chromatic_numbers.csv");
    out << "Graph Name, Num_nodes, Num_edges"<<endl;
    //string path = argv[1];
    DIR *pDIR;
    struct dirent *entry;
    if( pDIR=opendir(path) ){
            while(entry = readdir(pDIR)){
              cout << entry->d_name << endl;
              string entityname = path+ ((string)entry->d_name);
              ifstream input(entityname.c_str());

              if(input.fail()){
                  cerr << "no such input file" << endl;
                  return 0;
              }
              // read graph
              string line = "%";
              while(line.find("%") != string::npos){
                getline(input, line);
              }
              istringstream ss(line);
              int num_nodes, num_edges;
              ss >> num_nodes >> num_nodes >> num_edges;
              vector<pair<int,int>> edge_list;
              int v1, v2;
              double weight;
              for(int i=0; i<num_edges; i++){
                  getline(input, line);
                  istringstream inp(line);
                  inp>>v1>>v2;
                  v1--;
                  v2--;
                  if(v1!=v2){
                    edge_list.push_back(pair<int,int>(v1, v2));
                  }
              }
              num_edges = edge_list.size();
              try{
                  // create environment
                  IloEnv env;
                  IloModel model(env);

                  // create variables
                  vector<IloIntVar> y(num_nodes);
                  vector<vector<IloIntVar>> x(num_nodes);

                  for(int v=0; v<num_nodes; v++){
                      y[v] = IloIntVar(env, 0,1);
                      for(int c=0; c<num_nodes; c++){
                          x[v].push_back(IloIntVar(env, 0, 1));
                      }
                  }

                  // constraints

                  // every node should be colored only once
                  for(int i=0; i<num_nodes; i++){
                      IloExpr obj(env);
                      for(int color=0; color<num_nodes; color++){
                          obj += x[i][color];
                      }
                      model.add(obj == 1);
                  }

                  // there is no such neighbors such that their color is same
                  for(int color =0; color < num_nodes; color++){
                      for(auto & p: edge_list){
                          IloExpr obj(env);
                          obj += x[p.first][color] + x[p.second][color];
                          model.add(obj <= 1);
                      }
                  }

                  // x_color <= num_nodes * isUsed(color)
                  for(int color =0; color < num_nodes; color++){
                      IloExpr obj(env);
                      for(int i=0; i<num_nodes; i++){
                          obj += x[i][color];
                      }
                      model.add(obj <= num_nodes*y[color]);
                  }

                  // small |color|s are preferred
                  for(int i=0; i<num_nodes-1; i++){
                      model.add(y[i+1] <= y[i]);
                  }

                  // objective that is to be minimized
                  IloExpr obj(env);
                  for(int color=0; color<num_nodes; color++){
                      obj += y[color];
                  }
                  model.add(IloMinimize(env, obj));

                  // solve and print results
                  IloCplex cplex(model);
                  cplex.solve();

                  int totalcolors = 0;
                  for( auto & a: y){
                      totalcolors+= cplex.getValue(a);
                  }
                  out << entry->d_name << "," << num_nodes << ", " << num_edges<< endl;
                  env.end();
              }
              catch (IloException& ex) {
                 cerr << "Error: " << ex << endl;
              }
              catch (...) {
                 cerr << "Error" << endl;
              }

            }
            closedir(pDIR);
    }


    return 0;
}
