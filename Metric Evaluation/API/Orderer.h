#ifndef ORDERER_H
#define ORDERER_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <queue>
#include <climits>
#include "UGraph.h"
using namespace std;

class Orderer {

    public:
        Orderer(const UGraph&);
        void bfs(int, vector<int> &, int);

    private:
        vector<int> rowPtr;
        vector<int> colInd;
        int numOfNodes;

};

// constructor
Orderer::Orderer(const UGraph & graph) {
    rowPtr = graph.getRowPtr();
    colInd = graph.getColInd();
    numOfNodes = graph.getNodeSize();
}

// helper functions
void Orderer::bfs(int startNode, vector<int> & distanceArr, int stepSize = INT_MAX) {
    distanceArr.assign(numOfNodes, -1); // every node is unvisited
	distanceArr[startNode] = 0; // distance from a node to itself is 0
	queue<int> frontier; // FIFO queue
	frontier.push(startNode);
	int dist = 1; // initial distance
	bool improvement = true;
	while (improvement && dist <= stepSize) {
		improvement = false;
		queue<int> new_frontier; // FIFO queue
		do {
			int & front = frontier.front();
			frontier.pop();
			for (int edge = rowPtr[front]; edge < rowPtr[front + 1]; edge++) { // for each adjacent of front
				const int & adj = colInd[edge];
				if (distanceArr[adj] == -1) { // if it is not visited
					improvement = true;
					distanceArr[adj] = dist; // assign corresponding distance
					new_frontier.push(adj); // add it to the frontier
				}
			}
		} while (!frontier.empty());
		frontier = new_frontier;
		dist++; // next frontier will be further
	}
}


// ordering algorithms


#endif