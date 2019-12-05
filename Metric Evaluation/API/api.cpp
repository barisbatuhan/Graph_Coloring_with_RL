#include <iostream>
#include <string>
#include "IOHandler.h"
#include "UGraph.h"
#include "Orderer.h"
using namespace std;

int main(int argc, char** argv) {

    string path = string(argv[1]);

    vector<pair<int, int>> lines;
    bool res = IOHandler::readSuiteSparseMtx(path, lines);
    UGraph graph(path, lines);
    // graph.printRowPtr();
    Orderer orderer(graph);
    vector<pair<int, float>> ordering(graph.getNodeSize());
    orderer.weightedAnalysis(ordering);
    for(int i = 0; i < ordering.size(); i++) {
        cout << ordering[i].first << " - " << ordering[i].second << endl;
    }
    return 0;
}