#include <iostream>
#include <string.h>
#include "IOHandler.h"
#include "UGraph.h"
#include "Orderer.h"
#include "Colorer.h"
using namespace std;

/**
 * In order to run the code:
 * g++ api.cpp UGraph.h Orderer.h IOHandler.h Colorer.h
 * In linux     : ./a.out ../../matrices/
 * In Windows   : .\a.exe ..\..\matrices\
 */

int main(int argc, char** argv) {

    string path = string(argv[1]);
    // reading the directory matrix files
    vector<string> filenames = IOHandler::readAllMtxInDir(path);
    vector<int> totalColors(7, 0);
    
    cout << "Graph,Degree1,Degree2,Degree3,ClusteringCoeff,ClosenessCentrality,PageRank,WeightedAnalysis" << endl;

    for(int i = 0; i < filenames.size(); i++) {
        
        // reading the graph
        string absolutePath = path + filenames[i];
        vector<pair<int, int>> lines;
        bool res = IOHandler::readSuiteSparseMtx(absolutePath, lines);
        UGraph graph(path, lines);

        // analysis of orderings
        vector<vector<pair<int, float>>> orders(7, vector<pair<int, float>>(graph.getNodeSize()));
        Orderer orderer(graph);
        orderer.degreeOrder(orders[0]);
        orderer.degree2Order(orders[1]);
        orderer.degree3Order(orders[2]);
        orderer.clusteringCoefficient(orders[3]);
        orderer.closenessCentrality(orders[4]);
        orderer.pageRank(orders[5]);
        orderer.weightedAnalysis(orders[6], {0.15, 0.0, 0.1, 0.7, 0.05, 0});

        // gets the max degree for coloring and colors
        int maxDegree = orders[0][0].second;
        Colorer colorer(graph);
        int deg1Colors = colorer.colorGreedily(orders[0], maxDegree);
        totalColors[0] += deg1Colors;
        int deg2Colors = colorer.colorGreedily(orders[1], maxDegree);
        totalColors[1] += deg2Colors;
        int deg3Colors = colorer.colorGreedily(orders[2], maxDegree);
        totalColors[2] += deg3Colors;
        int clusteringCoefficientColors = colorer.colorGreedily(orders[3], maxDegree);
        totalColors[3] += clusteringCoefficientColors;
        int closenessCentralityColors = colorer.colorGreedily(orders[4], maxDegree);
        totalColors[4] += closenessCentralityColors;
        int pageRankColors = colorer.colorGreedily(orders[5], maxDegree);
        totalColors[5] += pageRankColors;
        int weightedAnalysisColors = colorer.colorGreedily(orders[6], maxDegree);
        totalColors[6] += weightedAnalysisColors;

        cout << filenames[i] << "," << deg1Colors << "," << deg2Colors << "," << deg3Colors << "," 
             << clusteringCoefficientColors << "," << closenessCentralityColors << "," << pageRankColors 
             << "," << weightedAnalysisColors << endl;
    }

    cout << endl
        << "Degree 1        : " << totalColors[0] << endl
        << "Degree 2        : " << totalColors[1] << endl
        << "Degree 3        : " << totalColors[0] << endl
        << "ClusCoeff       : " << totalColors[1] << endl
        << "ClosenessCent   : " << totalColors[0] << endl
        << "PageRank        : " << totalColors[1] << endl
        << "Weighted        : " << totalColors[0] << endl
        << "Optimal         : " << 1347 << endl;
    
    return 0;
}