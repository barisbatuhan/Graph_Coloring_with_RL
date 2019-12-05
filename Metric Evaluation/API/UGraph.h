#ifndef UGRAPH_H
#define UGRAPH_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include<dirent.h>
using namespace std;

class UGraph {
    
    public:
        // string argument is the name of the graph
        // constructs an undirected graph from the edge lists
        // first pair is: number of nodes, number of edges
        // 1st bool parameter indicates if node numbers start from 1 (false) or 0 (true)
        // 2nd bool parameter stands for if vertices with no edges are eliminated or not
        UGraph(string, vector<pair<int, int>> &, bool, bool);
		UGraph();
		UGraph(const UGraph&);
        
        // getters
        int getNodeSize() const;
        int getEdgeSize() const;
        string getGraphName() const;
		vector<int> getRowPtr() const;
		vector<int> getColInd() const;
		void printRowPtr();

    private:
        string graphName;
        int numOfNodes;
        int numOfEdges;
        vector<int> rowPtr;
        vector<int> colInd;
};

// constructor
UGraph::UGraph(string graphName, vector<pair<int, int>> & edges, bool ifZeroBased = false, bool eliminateUnused = true) {
    
    // initializations
    numOfEdges = edges[0].second;
    graphName = graphName;
    
    // parameters needed
    int initNodeCnt = edges[0].first;
    vector<vector<int> > adjList(initNodeCnt);
    vector<int> renameArr(initNodeCnt, -1);
	int counter = 0;

    // pushing edges to adjList
    for(int i = 1;  i < edges.size(); i++) {      
        int v1 = edges[i].first;
		int v2 = edges[i].second;
        if(!ifZeroBased) {
            v1--;
            v2--;
        }
		//for detecting vetices that are unused
		if (renameArr[v1] == -1 && eliminateUnused) {
			renameArr[v1] = counter;
			v1 = counter;
			counter++;
		}
		else if (eliminateUnused) {
			v1 = renameArr[v1];
		}
		if (renameArr[v2] == -1 && eliminateUnused) {
			renameArr[v2] = counter;
			v2 = counter;
			counter++;
		}
		else if (eliminateUnused) {
			v2 = renameArr[v2];
		}
		if (v1 != v2) {
			adjList[v1].push_back(v2); // add the edge v1->v2
			adjList[v2].push_back(v1); // add the edge v2->v1
		}
	}
    if (eliminateUnused) numOfNodes = counter;
    else numOfNodes = edges[0].first;
    
    rowPtr = vector<int>(numOfNodes + 1);
	colInd = vector<int>(2 * numOfEdges);
	rowPtr[0] = 0;
	int index = 0;
	for (int v = 0; v < numOfNodes; v++) {
		rowPtr[v + 1] = adjList[v].size(); // assign number of edges going from node v
		for (int i = 0; i < (int) adjList[v].size(); i++) {
			colInd[index] = adjList[v][i]; // put all edges in order wrt rowPtr
			index++;
		}
	}
	for (int v = 1; v<numOfNodes + 1; v++) {  // cumulative sum
		rowPtr[v] += rowPtr[v - 1];
	}
}

UGraph::UGraph() {
	numOfEdges = 0;
	numOfNodes = 0;
	graphName = "UNDEFINED";
}

UGraph::UGraph(const UGraph & graph) {
	graphName = graph.getGraphName();
	numOfNodes = graph.getNodeSize();
	numOfEdges = graph.getEdgeSize();
	rowPtr = graph.getRowPtr();
	colInd = graph.getColInd();
}

// getters
int UGraph::getNodeSize() const { return numOfNodes; }
int UGraph::getEdgeSize() const { return numOfEdges; }
string UGraph::getGraphName() const { return graphName; }
vector<int> UGraph::getRowPtr() const { return rowPtr; }
vector<int> UGraph::getColInd() const { return colInd; }

// debugging functions
void UGraph::printRowPtr() {
	for(int i = 0; i < rowPtr.size(); i++) {
		cout << rowPtr[i] << " - ";
	}
}

#endif