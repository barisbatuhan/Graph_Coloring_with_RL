#ifndef COLORER_H
#define COLORER_H

#include <iostream>
#include <vector>
#include "UGraph.h"
#include "Orderer.h"
using namespace std;

class Colorer {
    public:
        Colorer(UGraph &);
        // paints the graph greedily according to given order
        int colorGreedily(vector<pair<int, float>> &, int);
		vector<int> colorGreedilyNodes(vector<pair<int, float>> &, int);
		int recolorGreedily(vector<int> &, int);
		int dist2ColorGreedily(vector<pair<int, float>> &, int);

    private:
        vector<int> rowPtr;
        vector<int> colInd;
        int nodeSize;

        // checks if the coloring made is valid or not
        bool isValid(const vector<int> & );
};

Colorer::Colorer(UGraph & graph) {
    rowPtr = graph.getRowPtr();
    colInd = graph.getColInd();
    nodeSize = graph.getNodeSize();
}

int Colorer::colorGreedily(vector<pair<int, float>> & ordering, int maxDegree) {
    vector<int> color_arr;
	color_arr.resize(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(maxDegree + 1, -1);
	bool hasEdge = false;
	for (int i = 0; i<ordering.size(); i++) {
		const int & node = ordering[i].first; //for each node in ordering
		for (int edge = rowPtr[node]; edge < rowPtr[node + 1]; edge++) {
			hasEdge = true;
			const int & adj = colInd[edge];//for each adjacent node
			if (color_arr[adj] != -1) { // if it is already colored
				forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
			}
		}
		for (int color = 0; color < maxDegree; color++) { // greedily choose the smallest possible color
			if (forbid_arr[color] != node) {
				color_arr[node] = color;
				if (nofcolors < color) {
					nofcolors = color;
				}
				break;
			}
		}
	}
	if(hasEdge){
		nofcolors++;
	}
	if (!isValid(color_arr) == true) {
		cout << "The Coloring made by colorGreedily function is not valid!" << endl;
	}

	return nofcolors;
}

vector<int> Colorer::colorGreedilyNodes(vector<pair<int, float>> & ordering, int maxDegree) {
    vector<int> color_arr;
	color_arr.resize(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(maxDegree + 1, -1);
	bool hasEdge = false;
	for (int i = 0; i<ordering.size(); i++) {
		const int & node = ordering[i].first; //for each node in ordering
		for (int edge = rowPtr[node]; edge < rowPtr[node + 1]; edge++) {
			hasEdge = true;
			const int & adj = colInd[edge];//for each adjacent node
			if (color_arr[adj] != -1) { // if it is already colored
				forbid_arr[color_arr[adj]] = node; // that color is forbidden to node
			}
		}
		for (int color = 0; color < maxDegree; color++) { // greedily choose the smallest possible color
			if (forbid_arr[color] != node) {
				color_arr[node] = color;
				if (nofcolors < color) {
					nofcolors = color;
				}
				break;
			}
		}
	}
	if(hasEdge){
		nofcolors++;
	}
	if (!isValid(color_arr) == true) {
		cout << "The Coloring made by colorGreedily function is not valid!" << endl;
	}

	// number of colors that is used is appended to the end of the result
	color_arr.push_back(nofcolors);
	return color_arr;
}

int Colorer::dist2ColorGreedily(vector<pair<int, float>> & ordering, int maxDegree) {
	vector<int> color_arr;
	color_arr.resize(ordering.size(), -1);
	int nofcolors = 0;
	vector<int> forbid_arr(maxDegree + 1, -1);
	bool hasEdge = false;
	for (int i = 0; i<ordering.size(); i++) {
		const int & node = ordering[i].first; //for each node in ordering
		for (int edge = rowPtr[node]; edge < rowPtr[node + 1]; edge++) {
			hasEdge = true;
			const int & adj = colInd[edge];//for each adjacent node
			for (int edge_2 = rowPtr[adj]; edge_2 < rowPtr[adj + 1]; edge_2++) {
				const int & adj_neigh = colInd[edge_2];
				if (color_arr[adj_neigh] != -1) { // if it is already colored
					forbid_arr[color_arr[adj_neigh]] = node; // that color is forbidden to node
				}
			}
		}
		for (int color = 0; color < maxDegree; color++) { // greedily choose the smallest possible color
			if (forbid_arr[color] != node) {
				color_arr[node] = color;
				if (nofcolors < color) {
					nofcolors = color;
				}
				break;
			}
		}
	}
	if (hasEdge) {
		nofcolors++;
	}
	if (!isValid(color_arr) == true) {
		cout << "The Coloring made by colorGreedily function is not valid!" << endl;
	}
	return nofcolors;
}

int Colorer::recolorGreedily(vector<int> & color_arr, int prevResult) {
	vector<pair<int, float>> ordering(color_arr.size());
	vector<int> cntOfColors(prevResult, 0);
	for(int i = 0; i < color_arr.size(); i++) {
		cntOfColors[color_arr[i]] += 1;
		pair<int, float> nodeInfo = {i, (float) color_arr[i]};
		ordering[i] = nodeInfo;
	}
	// perform a different ordering
	sort(ordering.begin(), ordering.end(), Orderer::ascendingSecond);
	return colorGreedily(ordering, prevResult);
}

bool Colorer::isValid(const vector<int> & colorArr) {
	for (int v = 0; v < nodeSize; v++) { // for each node v
		for (int e = rowPtr[v]; e < rowPtr[v + 1]; e++) {
			const int & adj = colInd[e]; // for each adjacent of v
			if (colorArr[adj] == colorArr[v]) { // if color of v equals its adjacent's color
				return false;
			}
		}
	}
	return true;
}

#endif