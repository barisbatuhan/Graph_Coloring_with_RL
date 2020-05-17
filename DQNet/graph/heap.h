/**
 * Implementation of heap data structure.
 * */

#ifndef HEAP_H
#define HEAP_H

#include <vector>
#include <iostream>
using namespace std;

class Heap{
public:
    Heap();
    Heap(int cap);

    bool isEmpty(){return currentSize==0;}
    bool isFull(){return currentSize == (int)arr.size()-1;}
    pair<int, float> findMax(){return arr[1];}

    void insert(const pair<int,float> & item);
    pair<int, float> deleteMax();
    void makeEmpty();

    void percolateDown(int hole);
    void percolateUp(int hole);
    void buildHeap(vector<pair<int,float> > & vec);


    int currentSize;
    vector<pair<int, float> > arr;
    vector<int> locArr;
};

#endif
