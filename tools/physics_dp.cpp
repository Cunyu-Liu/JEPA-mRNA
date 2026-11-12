#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "param/turner.h"
#include "physics_zuker.cpp"

namespace py = pybind11;

class PhysicsNearestNeighbor : public TurnerNearestNeighbor
{
public:
    PhysicsNearestNeighbor(const std::string& seq, py::object obj, py::array_t<float> scores)
        : TurnerNearestNeighbor(seq, obj), parameters_(obj), scores_(scores.unchecked<2>()) {}

    ScoreType score_hairpin(size_t i, size_t j) const
    {
        return TurnerNearestNeighbor::score_hairpin(i, j) + pair_score(i, j);
    }

    ScoreType score_single_loop(size_t i, size_t j, size_t k, size_t l) const
    {
        return TurnerNearestNeighbor::score_single_loop(i, j, k, l) + pair_score(i, j);
    }

    ScoreType score_multi_loop(size_t i, size_t j) const
    {
        return TurnerNearestNeighbor::score_multi_loop(i, j) + pair_score(i, j);
    }

    ScoreType score_helix(size_t i, size_t j, size_t m) const
    {
        return TurnerNearestNeighbor::score_helix(i, j, m) + pair_score(i, j);
    }

private:
    py::object parameters_;
    pybind11::detail::unchecked_reference<float, 2> scores_;
    ScoreType pair_score(size_t i, size_t j) const { return scores_(i-1, j-1); }
};

py::array_t<float> zeros_like(const py::array_t<float>& values)
{
    py::array_t<float> result(values.request().shape);
    std::fill(result.mutable_data(), result.mutable_data() + result.size(), 0.0f);
    return result;
}

py::object parameters()
{
    py::dict obj;
    py::module_ tables = py::module_::import("mxfold2.param_turner2004");
    for (const auto& name : {
        "score_stack", "score_hairpin", "score_bulge", "score_internal",
        "score_mismatch_external", "score_mismatch_hairpin", "score_mismatch_internal",
        "score_mismatch_internal_1n", "score_mismatch_internal_23", "score_mismatch_multi",
        "score_int11", "score_int21", "score_int22", "score_dangle5", "score_dangle3",
        "score_ml_base", "score_ml_closing", "score_ml_intern", "score_ninio",
        "score_max_ninio", "score_duplex_init", "score_terminalAU", "score_lxc"})
    {
        py::array_t<float> values = tables.attr(name).cast<py::array_t<float>>();
        if (std::string(name) == "score_mismatch_external" || std::string(name) == "score_dangle5" || std::string(name) == "score_dangle3")
            values = zeros_like(values);
        obj[py::str(name)] = values;
        obj[py::str(std::string("count_") + (std::string(name).substr(6)))] = zeros_like(values);
    }
    return obj;
}

py::tuple decode(const std::string& sequence, py::array_t<float> scores)
{
    if (scores.ndim() != 2 || scores.shape(0) != scores.shape(1) || scores.shape(0) != sequence.size())
        throw std::invalid_argument("scores must be an L by L matrix");
    Zuker<PhysicsNearestNeighbor>::Options options;
    options.min_hairpin_loop_length(3).max_helix_length(0);
    auto param = std::make_unique<PhysicsNearestNeighbor>(sequence, parameters(), scores);
    Zuker<PhysicsNearestNeighbor> model(std::move(param));
    model.compute_viterbi(sequence, options);
    auto [energy, pairs] = model.traceback_viterbi(sequence, options);
    return py::make_tuple(energy, Zuker<PhysicsNearestNeighbor>::make_paren(pairs), pairs);
}

PYBIND11_MODULE(physics_dp, m)
{
    m.def("decode", &decode, "Turner loop-state DP with one injection of frozen candidate scores");
    m.def("parameters", &parameters);
}
