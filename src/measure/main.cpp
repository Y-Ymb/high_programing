#include "tester.h" 
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#if USEMPI
#  include "mpi.h"
#endif

int main( int argc, char *argv[] )
{
    try {
#if USEMPI
        MPI::Init();
#endif
        srand(getpid()*time(NULL));
        Tester test;
        test.run(argc, argv);
    } catch (const char *ex) {
        std::cerr << "# ERROR! :: " << ex << std::endl;
#if USEMPI
        MPI::COMM_WORLD.Abort(1);
#endif
    } catch (std::string ex) {
        std::cerr << "# ERROR! :: " << ex << std::endl;
#if USEMPI
        MPI::COMM_WORLD.Abort(1);
#endif
    }
#if USEMPI
    catch ( MPI::Exception e ) {
        std::cout << "MPI::Exception" << std::endl;
        std::cout << "# ERROR! :: " << e.Get_error_string() << e.Get_error_code();
        MPI::COMM_WORLD.Abort(1);
    }
    catch (...) {
        std::cout << "# ERROR! :: some error has been occured..." << std::endl;
        MPI::COMM_WORLD.Abort(1);
    }
#endif
#if USEMPI
    if (!MPI::Is_finalized()) {
        MPI::Finalize();
    }
#endif
    return 0;
}
