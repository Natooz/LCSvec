#include <nanobind/stl/vector.h>
#include "cpu/lcs_cpu.cpp"
#include "cpu/lccs_cpu_dyn.cpp"


NB_MODULE(lcsvec_ext, m) {
    m.doc() = "A python extension for fast Longest Common Subsequence (LCS) calculation on scalar vectors;";

    m.def("lcs", &lcs, "seq1"_a, "seq2"_a,
          "Returns the longest common subsequence (lcs) from `seq1` and `seq2`.");
    m.def("lcs_length", &lcsLength, "seq1"_a, "seq2"_a,
          "Returns the length of the longest common subsequence (lcs) from `seq1` and `seq2`. If you only need to get the length of the lcs, calling this method will be more efficient than calling `lcs()`.");
    m.def("lcs_table", &createLCSTable, "seq1"_a, "seq2"_a,
          "Returns the longest common subsequence (lcs) table from `seq1` and `seq2`.");

    m.def("lccs", &lccs, "seq1"_a, "seq2"_a,
          "Returns the longest common contiguous subsequence (lccs) from `seq1` and `seq2`.");
    m.def("lccs_length", &lccsLength, "seq1"_a, "seq2"_a,
          "Returns the length of the longest common contiguous subsequence (lccs) from `seq1` and `seq2`.");
}
