#include <iostream>
#include <string.h>
#include "IOHandler.h"
#include "UGraph.h"
#include "Orderer.h"
#include "Colorer.h"
using namespace std;

/**
 * In order to run the code:
 * g++ api.cpp UGraph.h Orderer.h IOHandler.h Colorer.h -O3
 * In linux     : ./a.out ../../matrices/
 * In Windows   : .\a.exe ..\..\matrices\
 */

int main(int argc, char** argv) {

    string path = string(argv[1]);
    // reading the directory matrix files
    vector<string> filenames = IOHandler::readAllMtxInDir(path);
    vector<int> totalColors(8, 0);
    
    cout << "Graph,Degree1,Degree2,Degree3,ClusteringCoeff,ClosenessCentrality,PageRank,WeightedAnalysis,RecoloringOfWeighted" << endl;

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
        orderer.weightedAnalysis(orders[6], {0.15, 0.0, 0.1, 0.05, 0.7, 0.0});

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
        // int weightedAnalysisColors = colorer.colorGreedily(orders[6], maxDegree);
        // totalColors[6] += weightedAnalysisColors;
        vector<int> weightedColorArr = colorer.colorGreedilyNodes(orders[6], maxDegree);
        int weightedAnalysisColors = weightedColorArr[weightedColorArr.size()-1];
        totalColors[6] += weightedAnalysisColors;

        // recoloring of weighted 
        weightedColorArr.pop_back();
        int recoloringColors = colorer.recolorGreedily(weightedColorArr, weightedAnalysisColors);
        totalColors[7] += recoloringColors;

        cout << filenames[i] << "," << deg1Colors << "," << deg2Colors << "," << deg3Colors << "," 
             << clusteringCoefficientColors << "," << closenessCentralityColors << "," << pageRankColors 
             << "," << weightedAnalysisColors  << "," << recoloringColors << endl;
    }

    cout << endl
        << "Algorithm       | Total # of Colors | Ratio" << endl

        << "Degree 1        : " << totalColors[0] << " \t\t" << (float) totalColors[0] / 1347 << endl
        << "Degree 2        : " << totalColors[1] << " \t\t" << (float) totalColors[1] / 1347 << endl
        << "Degree 3        : " << totalColors[2] << " \t\t" << (float) totalColors[2] / 1347 << endl
        << "ClusCoeff       : " << totalColors[3] << " \t\t" << (float) totalColors[3] / 1347 << endl
        << "ClosenessCent   : " << totalColors[4] << " \t\t" << (float) totalColors[4] / 1347 << endl
        << "PageRank        : " << totalColors[5] << " \t\t" << (float) totalColors[5] / 1347 << endl
        << "Weighted        : " << totalColors[6] << " \t\t" << (float) totalColors[6] / 1347 << endl
        << "Recolored Wei.  : " << totalColors[7] << " \t\t" << (float) totalColors[7] / 1347 << endl
        << "Optimal         : " << 1347 << endl;
    
    return 0;
}