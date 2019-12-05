#ifndef ORDERER_H
#define ORDERER_H

#include <iostream>
#include <vector>
#include <unordered_set>
#include <queue>
#include <numeric> // for accumulate
#include <climits>
#include <cmath>
#include <algorithm>
#include "UGraph.h"
using namespace std;

class Orderer {

    public:
        Orderer(const UGraph&);
        
		// ordering functions
		void clusteringCoefficient(vector<pair<int, float>>&); 
		void closenessCentrality(vector<pair<int, float>>&);
		void degreeOrder(vector<pair<int, float>>&);
		void degree2Order(vector<pair<int, float>>&);
		void degree3Order(vector<pair<int, float>>&);
		void pageRank(vector<pair<int, float>>&, int = 100, float = 0.85);
		void weightedAnalysis(vector<pair<int, float>>&, vector<float> = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5});

		// helper functions for ordering algorithms
		void bfs(int, vector<int>&, int = INT_MAX);

		// static helpers
		static pair<int, float> add(const float & left, const pair<int, float> & right) {
			return pair<int, float>(right.first, left + right.second);
		}
		static bool descendingFirst(const pair<int, float> & left, const pair<int, float> & right) {
			return left.first>right.first;
		}

		static bool ascendingFirst(const pair<int, float> & left, const pair<int, float> & right) {
			return left.first<right.first;
		}

		static bool descendingSecond(const pair<int, float> & left, const pair<int, float> & right) {
			return left.second>right.second;
		}

		static bool ascendingSecond(const pair<int, float> & left, const pair<int, float> & right) {
			return left.second<right.second;
		}

    private:
        vector<int> rowPtr;
        vector<int> colInd;
        int numOfNodes;
		int maxDegree;
		
		// helper functions for ordering algorithms
		void normalizeOrder(vector <pair<int, float>>&);
};

// constructor
Orderer::Orderer(const UGraph & graph) {
    rowPtr = graph.getRowPtr();
    colInd = graph.getColInd();
    numOfNodes = graph.getNodeSize();
	// add maxDegree initialization to here, too
	maxDegree = 0;
}

// ordering algorithms
void Orderer::clusteringCoefficient(vector<pair<int, float>> & ordering) {
	for (int v = 0; v<numOfNodes; v++) { // for each node
		int degree = rowPtr[v + 1] - rowPtr[v];
		int possiblelinks = degree*(degree - 1) / 2;
		int noflinks = 0;
		unordered_set<int> set;
		for (int edge = rowPtr[v]; edge < rowPtr[v + 1]; edge++) { // insert all of its neighbors
			int adj = colInd[edge];
			set.insert(adj);
		}

		for (int edge = rowPtr[v]; edge < rowPtr[v + 1]; edge++) {
			const int & adj = colInd[edge]; // for each neighbor of v
			for (int link = rowPtr[adj]; link < rowPtr[adj + 1]; link++) {
				int adj_neigh = colInd[link];// for each neighbor of neighbor
				if (set.find(adj_neigh) != set.end()) { // count number of links
					noflinks++;
				}
			}
		}
		float coeff = noflinks != 0 ? (float)possiblelinks / noflinks : possiblelinks+1;
		ordering[v] = make_pair(v, coeff);
	}
	sort(ordering.begin(), ordering.end(), descendingSecond);
}

void Orderer::closenessCentrality(vector<pair<int, float>> & ordering) {
	vector<int> distArr;
	for (int v = 0; v<numOfNodes; v++) {
		bfs(v, distArr); // take distance array for node v
		int sum_of_dist = accumulate(distArr.begin(), distArr.end(), 0); // sum of d(v,x) for all x in the graph
		float coeff = sum_of_dist > 0 ? (float)numOfNodes / sum_of_dist : 0; // if coefficient is negative(meaning that graph is not connected) assign to 0
		ordering[v] = make_pair(v, coeff);
	}
	sort(ordering.begin(), ordering.end(), descendingSecond);
}

void Orderer::degreeOrder(vector<pair<int, float>> & ordering) {
	for (int v = 0; v<numOfNodes; v++) {
		ordering[v] = make_pair(v, rowPtr[v + 1] - rowPtr[v]);
	}
	sort(ordering.begin(), ordering.end(), descendingSecond);
}

void Orderer::degree2Order(vector<pair<int, float>> & ordering) {
	vector<int> distArr;
	for (int v = 0; v<numOfNodes; v++) {
		bfs(v, distArr, 2); // take distance array for node v
		ordering[v] = make_pair(v, count(distArr.begin(), distArr.end(), 2));
	}
	sort(ordering.begin(), ordering.end(), descendingSecond);
}

void Orderer::degree3Order(vector<pair<int, float> > & ordering) {
	vector<int> distArr;
	for (int v = 0; v<numOfNodes; v++) {
		bfs(v, distArr, 3); // take distance array for node v
		ordering[v] = make_pair(v, count(distArr.begin(), distArr.end(), 3));
	}
	sort(ordering.begin(), ordering.end(), descendingSecond);
}

void Orderer::pageRank(vector<pair<int, float>> & ordering, int iter, float alpha) {
	for (int i = 0; i<numOfNodes; i++) { ordering[i] = make_pair(i, (float)1 / numOfNodes); } // initially likelyhoods are uniformly distributed
	vector<float> offset(numOfNodes, 0.0);
	for (int i = 0; i<iter; i++) { // on each iteration (required for convergence)
		vector<pair<int, float> > copy_ordering(numOfNodes, pair<int, float>(0, 0.0));
		for (int j = 0; j < numOfNodes; j++) {
			copy_ordering[j].first = j;
		}
		for (int v = 0; v<numOfNodes; v++) { // for each node v
											// assign total page ranks %85
			float & pr_v = copy_ordering[v].second; // update page rank of v by looking its in-degree nodes
			for (int edge = rowPtr[v]; edge<rowPtr[v + 1]; edge++) { // for each in degree neighbor of v (since graph is symmetric in or out degree does not matter)
				const int & adj = colInd[edge];
				float & pr_adj = ordering[adj].second;
				int degree_adj = rowPtr[adj + 1] - rowPtr[adj];
				pr_v += pr_adj / degree_adj; // page_rank_of_v <- (page_rank_of_v + page_rank_of_neighbor/out_degree_of_neighbor)
			}
			// distribute %15 evenly
			float dist_value = pr_v * (1 - alpha) / (numOfNodes - 1);
			pr_v *= alpha;
			for_each(offset.begin(), offset.end(), [dist_value](float& d) { d += dist_value; });
			pr_v -= dist_value;
		}
		transform(offset.begin(), offset.end(), copy_ordering.begin(), ordering.begin(), add);
		offset.assign(numOfNodes, 0);
	}
	sort(ordering.begin(), ordering.end(), descendingSecond);
}

void Orderer::weightedAnalysis(vector<pair<int, float>>& ordering, vector<float> weights) {
	vector<vector<pair<int, float>>> orders(6, vector<pair<int,float>> (numOfNodes));
	degreeOrder(orders[0]);
	degree2Order(orders[1]);
	degree3Order(orders[2]);
	clusteringCoefficient(orders[3]);
	closenessCentrality(orders[4]);
	pageRank(orders[5]);

	for(auto & order: orders){
		normalizeOrder(order);
		sort(order.begin(), order.end(), ascendingFirst);
	} 

	// IMPLEMENT FURTHER
	for(int i = 0; i < numOfNodes; i++) {
		ordering[i] = make_pair(i, orders[0][i].second * weights[0] + orders[1][i].second * weights[1] 
					+ orders[2][i].second * weights[2] + orders[3][i].second * weights[3] 
					+ orders[4][i].second * weights[4] + orders[5][i].second * weights[5]);
	}
	sort(ordering.begin(), ordering.end(), descendingSecond);
}

// helper functions
void Orderer::bfs(int startNode, vector<int> & distanceArr, int stepSize) {
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

void Orderer::normalizeOrder(vector <pair<int, float>> & order) {
	float sum = 0;
	float sq_sum = 0;
	for(auto & x: order) {
    	sum += x.second;
    	sq_sum += x.second * x.second;
  	}
  	float mean = sum / order.size();
  	float stdev = sqrt(sq_sum / order.size() - mean * mean);
	if(stdev==0){
		stdev = 1;
	}
	for_each(order.begin(), order.end(), [mean, stdev](pair<int, float> & x) { 
		x.second = (x.second-mean)/stdev;
	});
}


#endif