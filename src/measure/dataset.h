#pragma once

#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <stdint.h>

typedef float melem_t;

class Dataset
{
public:
    enum DataType {
        free,
        square,
        mv,
        symm,
        trmm,
        hemm
    };

    Dataset();
    ~Dataset();
    void prepare(DataType type,
             uint32_t &n, uint32_t &m, uint32_t &k);
    void set(int la, int lb, int lc,
             melem_t *A, melem_t *B, melem_t *C);
    int  check(melem_t *C);

private:
    std::ifstream ifs;
    uint32_t n, m, k;
    int la, lb, lc;

    static void createTypeMap();
    static void createFileListMap();
    static std::map< DataType, std::vector< std::string > > fileListMap;
    const static std::string fileListPath;
    const static std::string dataDir;
    static std::map< std::string, DataType > typeMap;
};
