//  MSTM fast data loader
#include "pybind11/pybind11.h"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include <iostream>
#include <fstream>
#include <string>

using namespace std;

std::tuple<xt::pyarray<double>, xt::pyarray<double>, xt::pyarray<double>> load_mstm_fast(const std::string &filename,
const std::size_t &Ns) {
    
  string line;
//  string filename;
  string sp, hh, ka;

  int ii,jj;
  int nheader;
  int smline, smlineend;
  int sphline, sphlineend;
  int aggoptline; 
  int i;
  double value;
  
  
//  double spherematrix[Ns][9];
  auto mstmout = xt::pyarray<double>::from_shape({4});
  auto scatmatrix = xt::pyarray<double>::from_shape({181*11});
  auto spherematrix = xt::pyarray<double>::from_shape({Ns*10});
  
  
  nheader = 4;
  smline= nheader+36+Ns+2;
  smlineend = smline+182;
  sphline = nheader+35;
  sphlineend = sphline+Ns+1;
  aggoptline = nheader+36+Ns;
  
  
  i = 0;
  
  ifstream myfile (filename);
  if (myfile.is_open())
  { 
    while ( getline (myfile,line) )
    {
       if((i>sphline)&&(i<sphlineend))
       {
          jj = i-(sphline+1);
          istringstream iss(line);
            iss >> sp >> hh ;
          for (int q=0; q<10; q++){
            iss >> value;
            spherematrix[jj*10+q]=value;
          }
       }
       if((i>aggoptline-0.5)&&(i<aggoptline+0.5))
       {
          for (int k=0; k<4; k++){
            myfile >> mstmout[k];
         }
       }
       if((i>smline)&&(i<smlineend))
       {
          ii = i-(smline+1);
          for (int j=0; j<11; j++){
            myfile >> scatmatrix[ii*11+j];
          }
       }
      i++;
    } 
    myfile.close();
  } 
  else runtime_error("File not opened.");

  return {mstmout,scatmatrix,spherematrix};
}

// Python Module and Docstrings

PYBIND11_MODULE(mstm_output_reader, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        An efficient reader of MSTM output files
        .. currentmodule:: mstm_output_reader
        .. autosummary::
           :toctree: _generate
           load_mstm_fast
    )pbdoc";

    m.def("load_mstm_fast", load_mstm_fast, "Load a MSTM output file. Returns three NumPy arrays corresponding to MSMT output, scattering matrix, and sphere matrix.");
}
