#ifndef IOHANDLER_H
#define IOHANDLER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
using namespace std;

class IOHandler {
    public:
      
        static bool readSuiteSparseMtx(string path, vector<pair<int, int>> & lines) {
            cout << path << endl;
            ifstream input(path.c_str());
            if(input.fail()) return false;
            string line;
            bool firstLine = true;

            while(getline(input, line)){
                if(line.find("%") != string::npos) continue;
                istringstream ss(line);
                int num1, num2;
                if(firstLine) {
                    ss >> num1 >> num1 >> num2;
                    firstLine = false;
                } else {
                    ss >> num1 >> num2;
                }
                pair<int, int> readPair(num1, num2);
                lines.push_back(readPair);
            }
            return true;
        }
        
        // pass an empty lines vector, the function fills it in 1D
        static bool readCsvByRow(string path, vector<string> & lines) {
            ifstream input(path.c_str());
            if(input.fail()) return false;
            string line;
            while(getline(input, line)) {
                lines.push_back(line);
            }
            return true;
        }

        // pass an empty lines vector, the function fills it in 2D
        static bool readCsvByCell(string path, vector<vector<string>> & lines) {
            ifstream input(path.c_str());
            if(input.fail()) return false;
            string line;
            while(getline(input, line)) {
                istringstream subInput(line);
			    string element;
                vector<string> elements;
			    while(getline(subInput, element, ',')){
                    elements.push_back(element);
			    }
                lines.push_back(elements);
            }
            return true;
        }

        // pass a 2D lines vector with each cell you want to write, the function writes
        static bool writeToCsvByCell(string path, vector<vector<string>> & lines) {
            ofstream output;
            output.open(path);
            for(int row = 0; row < lines.size(); row++) {
                int colLength = lines[row].size();
                for(int col = 0; col < colLength; col++) {
                    if(col == colLength -1) {
                        output << lines[row][col] << endl;
                    } else {
                        output << lines[row][col] << ",";
                    }
                    
                }
            }
            output.close();
            return true;
        }

        // pass a 1D lines vector with row you want to write, the function writes
        static bool writeToCsvByRow(string path, vector<string> & lines) {
            ofstream output;
            output.open(path);
            for(int row = 0; row < lines.size(); row++) {
                output << lines[row] << endl;
            }
            output.close();
            return true;
        }
};

#endif