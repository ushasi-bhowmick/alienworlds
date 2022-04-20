%module gfg
%{
  #include "gfg.h"
%}

%include "std_vector.i"
// Instantiate templates used by example
namespace std {
   %template(IntVector) vector<int>;
   %template(FloatVector) vector<float>;
}

// Include the header file with above prototypes
%include "gfg.h"
std::vector<int> in_or_out(std::vector<float> refx,std::vector<float> refy, std::vector<float> newshx , std::vector<float> newshy) ;