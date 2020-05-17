#include "heap.h"

Heap::Heap(){
    currentSize = 0;
}

Heap::Heap(int cap){
    currentSize = 0;
    arr.resize(cap+1);
    for(int i=0; i<cap; i++){
        arr[i].first = i;
        arr[i].second = 0;
    }
    locArr.resize(cap+1,-1);
}

void Heap::insert(const pair<int,float> & item){

    if(isFull()){
        cout << "Heap is full" << endl;
    }
    int hole = ++currentSize;
    arr[hole] = item;
    locArr[item.first] = hole;
    percolateUp(hole);
}

pair<int, float> Heap::deleteMax(){

    if(isEmpty()){
        cout << "Heap is empty" << endl;
    }
    auto res = arr[1];
    arr[1] = arr[currentSize--];
    locArr[res.first] = -1;
    locArr[arr[1].first] = 1;

    percolateDown(1);

    return res;
}

void Heap::makeEmpty(){
    locArr.clear();
    locArr.resize(0);
    arr.clear();
    arr.resize(0);
    currentSize = 0;
}

void Heap::percolateDown(int hole){
    int child;
    auto temp = arr[hole];
    for(; hole*2 <= currentSize; hole = child){
        child = hole*2;
        if(child != currentSize && (arr[child+1].second > arr[child].second || (arr[child+1].second == arr[child].second && arr[child+1].first < arr[child].first))){
            child++;
        }
        if(arr[child].second > temp.second || (arr[child].second == temp.second && arr[child].first < temp.first)){
            arr[hole] = arr[child];
            locArr[arr[child].first] = hole;
        }
        else{
            break;
        }
    }
    arr[hole] = temp;
    locArr[temp.first] = hole;
}

void Heap::percolateUp(int hole){
    auto temp = arr[hole];
    for(; hole>1 && (temp.second > arr[hole/2].second || (temp.second == arr[hole/2].second && temp.first<arr[hole/2].first)); hole/=2){

        arr[hole] = arr[hole/2];
        locArr[arr[hole/2].first] = hole;
    }

    arr[hole] = temp;
    locArr[temp.first] = hole;
}

void Heap::buildHeap(vector<pair<int,float> > & vec){
    makeEmpty();
    arr.resize(vec.size()+1);
    locArr = vector<int>(vec.size()+1, -1);
    for(int i=1; i<(int)arr.size(); i++){
        arr[i] = vec[i-1];
        locArr[vec[i-1].first] = i;
    }
    currentSize = vec.size();


    for(int i=currentSize/2; i>0; i--){
        percolateDown(i);
    }

}
