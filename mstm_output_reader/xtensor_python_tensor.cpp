#include <numeric>
#include "pybind11/pybind11.h"
#include "xtensor/xmath.hpp"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

double sum_of_sines(xt::pyarray<double>& m)
{
   auto sines = xt::sin(m);
   return std::accumulate(sines.begin(),sines.end(),0.0);
}

PYBIND11_MODULE(xtensor_python_test, m)
{
   xt::import_numpy();
   m.doc() = "Test module for xtensor python bindings";
   m.def("sum_of_sines", sum_of_sines, "Sum the sines of the input values");
}
