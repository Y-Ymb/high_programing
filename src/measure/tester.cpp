#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <sys/time.h>
#include <math.h>
#include "tester.h"
#include "dataset.h"
#include "imulmat.h"
#include "cmdline.h"

#if USEMPI
#  include "mpi.h"
#endif

using namespace std;

#ifndef MMCLASS
#  define MMCLASS DummyMM
#  include "dummymm.h"
#endif

#ifndef VERSION
#  define VERSION "undefined"
#endif

#define RANK0 if (myrank==0)

Tester::Tester() : myrank(0)
{
    // cout << "Tester constructed" << endl;
#if USEMPI
    myrank = MPI::COMM_WORLD.Get_rank();
#endif
}

Tester::~Tester()
{
    // cout << "Tester destructed" << endl;
}

void Tester::print(string str)
{
    if (myrank == 0) {
        cout << str << endl;
    }
}

void Tester::_run(Dataset::DataType type)
{
    print("# Run");
    MMCLASS mm;
    
    uint32_t n, m, k;
    int la, lb, lc;
    melem_t *A, *B, *C;
    print("# Prepare ");
    
    // Prepare dataset
    Dataset dataset;
    RANK0 {
        dataset.prepare(type, n, m, k);
    }
#if USEMPI    
    MPI::COMM_WORLD.Bcast(&n, 1, MPI::UNSIGNED, 0);
    MPI::COMM_WORLD.Bcast(&m, 1, MPI::UNSIGNED, 0);
    MPI::COMM_WORLD.Bcast(&k, 1, MPI::UNSIGNED, 0);
#endif
    // Allocate the spaces of matrix
    mm.init(n, m, k, &la, &lb, &lc, &A, &B, &C);
    // Set A, B, C
    RANK0 {
        dataset.set(la, lb, lc, A, B, C);
    }
    print("# Run ");
    // Measure
    uint64_t before, after;
    before = getus();
    mm.multiply();
    after = getus();

    print("# Check ");
    // Check the answer
    int wcount = 0;
    if (myrank == 0) {
        wcount = dataset.check(C);
    
        // Print result
        uint64_t flop = 2ULL * n * m * k;
        double gflops = (double)flop/(after-before)*1E-3;
        ostringstream os;
        os << "# Elapsed: " << (double)(after-before)/1E3 << " [ms]" << endl;
        os << "# Flops:   " << gflops << " [GFLOPS]" << endl;
        os << "# Wrong:   " << wcount << " / " << n*m << endl;
        print(os.str());
    }
}

void Tester::run()
{
    _run(Dataset::square);
    
}

void Tester::run(int argc, char *argv[])
{
    Dataset::DataType type = Dataset::free;
    cmdline::parser p;
    p.add<string>("type", 't', "type of input matrix (free, square, mv, symm, trmm, hemm)", false, "free");
    p.add("help", 'h', "print help");
    p.add("version", 'v', "print version");

    // help
    if (!p.parse(argc, argv)||p.exist("help")) {
        print(p.error_full());
        print(p.usage());
        return;
    }

    if (p.exist("version")) {
        ostringstream os;
        os << "MulMat Tester -- version: " << VERSION << endl;
        os << "by Makoto Shimazu" << endl;
        print(os.str());
        return;
    }

    if (p.exist("type")) {
        string typeName = p.get<string>("type");
        if (typeName == "free")        type = Dataset::free;
        else if (typeName == "square") type = Dataset::square;
        else if (typeName == "mv")     type = Dataset::mv;
        else if (typeName == "symm")   type = Dataset::symm;
        else if (typeName == "trmm")   type = Dataset::trmm;
        else if (typeName == "hemm")   type = Dataset::hemm;
        else {
            ostringstream os;
            os << "Unknown type!" << endl;
            os << p.usage();
            print(os.str());
            return;
        }
    }
    
    _run(type);
}

uint64_t Tester::getus()
{
    struct ::timeval tv;
    ::gettimeofday(&tv, NULL);
    return tv.tv_sec*1E6 + tv.tv_usec;
}
