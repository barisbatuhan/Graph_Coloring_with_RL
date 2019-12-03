#include <iostream>
#include <string>
#include "IOHandler.h"
#include "UGraph.h"
using namespace std;

int main(int argc, char** argv) {

    string path = string(argv[1]);

    vector<pair<int, int>> lines;
    bool res = IOHandler::readSuiteSparseMtx(path, lines);
    UGraph graph(path, lines);
    graph.printRowPtr();

    return 0;
}