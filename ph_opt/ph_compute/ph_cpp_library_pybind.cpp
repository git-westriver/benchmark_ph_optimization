#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ph_cpp_library.hpp"
namespace py = pybind11;

PYBIND11_MODULE(example1, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("mns", &mns, "A function that minuses two numbers");
}

PYBIND11_MODULE(example2, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("add", py::overload_cast<ll, ll>(&add)).def("add", py::overload_cast<ll>(&add));
}

PYBIND11_MODULE(example3, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("inner_add", &inner_add, "A function that adds 1 to the input and return None");
}

PYBIND11_MODULE(dsu_cpp, m) {
    py::class_<DSU>(m, "DSU")
        .def(py::init<int>())
        .def("merge", &DSU::merge)
        .def("same", &DSU::same)
        .def("leader", &DSU::leader)
        .def("size", &DSU::size)
        .def("get_ph_root", &DSU::get_ph_root)
        .def("groups", &DSU::groups);
}

PYBIND11_MODULE(rips_cpp, m) {
    py::class_<RipsPersistentHomology>(m, "RipsPersistentHomology")
        .def(py::init<vector<vector<diameter_t>>, dim_t, unsigned long long>(), 
            py::arg("dist"), py::arg("maxdim"), py::arg("num_threads")=1ULL) 
        .def("binomial", &RipsPersistentHomology::binomial) 
        .def("get_simplex_index", &RipsPersistentHomology::get_simplex_index)
        .def("get_max_vertex", &RipsPersistentHomology::get_max_vertex)
        .def("get_simplex_vertices", &RipsPersistentHomology::get_simplex_vertices)
        .def("get_diameter", &RipsPersistentHomology::get_diameter)
        .def("get_max_edge", &RipsPersistentHomology::get_max_edge)
        .def("compute_ph", &RipsPersistentHomology::compute_ph, 
            py::arg("enclosing_opt")=true, py::arg("emgergent_opt")=true, py::arg("clearing_opt")=true, py::arg("get_inv")=false)
        .def("compute_ph_right", &RipsPersistentHomology::compute_ph_right, 
            py::arg("enclosing_opt")=true, py::arg("emgergent_opt")=true, py::arg("get_inv")=false)
        .def_readwrite("death_to_birth", &RipsPersistentHomology::death_to_birth)
        .def_readwrite("birth_to_death", &RipsPersistentHomology::birth_to_death)
        .def_readwrite("W", &RipsPersistentHomology::W)
        .def_readwrite("invW", &RipsPersistentHomology::invW)
        .def_readwrite("V", &RipsPersistentHomology::V)
        .def_readwrite("invV", &RipsPersistentHomology::invV);
}
