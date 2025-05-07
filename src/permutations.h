#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace bcf {

template <typename T,  typename U>
inline std::vector<T> getIdentityPermutation(U n) {
    std::vector<T> p(n);
    std::iota(p.begin(), p.end(), 0);
    return p;
}

// returns permutation p st v[p[0]] <= v[p[1]] <= .... <= v[p[n-1]]
template <class T>
inline std::vector<T> getSortPermutation(const std::vector<T>& v) {
    const auto n = v.size();
    auto p = getIdentityPermutation<T>(n);
    std::ranges::sort(p, [&v](T a, T b) {
        return v[a] <= v[b];
    });
    return p;
}

// returns the inverse of the permutation p
template <typename T>
inline std::vector<T> getInversePermutation(const std::vector<T>& p) {
    const auto n = p.size();
    std::vector<T> inv_p(n);
    for (typename std::vector<T>::size_type i = 0; i < n; i++) {
        inv_p[p[i]] = i;
    }
    return inv_p;
}

}  // namespace bcf
