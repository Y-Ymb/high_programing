//! g++ -std=c++11 dataset.cpp -fsyntax-only
#include "dataset.h"
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

using namespace std;

const string Dataset::dataDir      = "./data/";
const string Dataset::fileListPath = "./data/list.txt";
typedef map< Dataset::DataType, vector< string > > fileListMap_t;
fileListMap_t Dataset::fileListMap;
typedef map< string, Dataset::DataType > typeMap_t;
typeMap_t Dataset::typeMap;

Dataset::Dataset()
{
    if (typeMap.empty()) createTypeMap();
    if (fileListMap.empty()) createFileListMap();

    // // print all
    // fileListMap_t::iterator it = fileListMap.begin();
    // for (; it != fileListMap.end(); it++ ) {
    //     cout << "==== " << it->first << " ====" << endl;
    //     vector<string> fnames = it->second;
    //     vector<string>::iterator fit = fnames.begin();
    //     cout << "Count: " << fnames.size() << endl;
    //     for (; fit != fnames.end(); fit++ ) {
    //         cout << (*fit) << endl;
    //     }
    // }
}

Dataset::~Dataset()
{
}

void Dataset::createFileListMap()
{
    ifstream listfs(fileListPath.c_str());
    if (listfs.fail()) {
        throw string("\"") + fileListPath + "\" is not found!";
    }
    
    string type_str;
    // Decode first line
    listfs >> type_str;
    if (typeMap.count(type_str) == 0) {
        throw string("Unknown DataType ") + type_str;
    }
    DataType type = typeMap[type_str];
    while (!listfs.eof()) {
        string line;
        listfs >> line;
        if ( line == "" ) break;
        // Check whether the line matches the header lines
        if (typeMap.count(line) != 0) {
            // Set new type
            type_str = line;
            type = typeMap[type_str];
        } else {
            fileListMap[type].push_back(line);
        }
    }
    
}

void Dataset::createTypeMap()
{
    typeMap["<free>"]   = free;
    typeMap["<square>"] = square;
    typeMap["<mv>"]     = mv;
    typeMap["<symm>"]   = symm;
    typeMap["<trmm>"]   = trmm;
    typeMap["<hemm>"]   = hemm;
}

void Dataset::prepare(DataType type,
                      uint32_t &n, uint32_t &m, uint32_t &k)
{
    // prepare
    // random_device seed_gen;
    // mt19937 random(seed_gen());
    vector<string> files = fileListMap[type];
    if (files.size() == 0) {
        throw "There is no file in this type";
    }
    string file = files[rand()%files.size()];
    ifs.open((dataDir + file).c_str());
    cout << "# " << file << " is selected!" << endl;
    if (ifs.fail()) {
        throw "FAILED TO OPEN!";
    }
    
    // read n, m, k
    ifs.read((char*)(&n), sizeof(uint32_t));
    ifs.read((char*)(&m), sizeof(uint32_t));
    ifs.read((char*)(&k), sizeof(uint32_t));
    cout << "# n=" << n << " m=" << m << " k=" << k << endl;
    this->n = n; this->m = m; this->k = k;
}

void Dataset::set(int la, int lb, int lc,
                  melem_t *A, melem_t *B, melem_t *C )
{
    // Fill in A, B, C
    for ( uint32_t i = 0; i < n; i++ ) {
        ifs.read((char*)(A+i*la), sizeof(melem_t)*k);
    }
    for ( uint32_t i = 0; i < k; i++ ) {
        ifs.read((char*)(B+i*lb), sizeof(melem_t)*m);
    }
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < m; j++) {
            C[i*m+j] = 0;
        }
    }
    this->la = la; this->lb = lb; this->lc = lc;
}

int Dataset::check(melem_t *C)
{
    // Check Answer
    melem_t *ans = new melem_t[n*m]();
    ifs.read((char*)ans, sizeof(melem_t)*n*m);
    int wcount = 0;             // # of wrong answer
    float delta = pow(2,(log2f(k)+7*2)-23); // MAX*2^(-2) (float pricision is 2^23, max of each value is +/-2^7)
    cout << "# (Delta = " << delta << ")" << endl;
    for ( uint32_t i = 0; i < n; i++ ) {
        for ( uint32_t j = 0; j < m; j++ ) {
            if ( fabs(C[i*lc+j] - ans[i*m+j]) > delta ) {
                cerr << "### WRONG: "
                     << "C(" << i << ", " << j << ") != "
                     << "Ans(" << i << ", " << j << ") :: "
                     << "C = " << C[i*lc+j] << ", "
                     << "Ans = " << ans[i*m+j] << endl;
                wcount++;
            }
        }
    }
    return wcount;
}
