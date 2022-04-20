%module gfg
%{
  #include "gfg.h"
%}

%include "std_vector.i"
// Instantiate templates used by example
namespace std {
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

// Include the header file with above prototypes
%include "gfg.h"
std::vector<int> in_or_out(std::vector<double> refx,std::vector<double> refy, std::vector<double> newshx , std::vector<double> newshy) ;