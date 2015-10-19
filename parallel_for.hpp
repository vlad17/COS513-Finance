#pragma once
#include <future>
#include <thread>
#include <type_traits>
#include <utility>

template<typename F, typename C, typename... Args>
using parallel_for_t = typename std::result_of<
  F(decltype(std::declval<C>().begin()), decltype(std::declval<C>().end()),
    Args&&...)
  >::type;

template<typename F, typename Container, typename... Args>
std::vector<parallel_for_t<F, Container, Args&&...>>
parallel_for(int concurrency, typename std::enable_if<std::is_void<
             parallel_for_t<F, Container, Args&&...>>::value, F>::type f,
             Container& v, Args&&... args) {
  typedef parallel_for_t<F, Container, Args&&...> R;
  std::vector<std::future<R>> futs;
  int jump = v.size() / concurrency;
  if (jump == 0) jump = 1;
  for (int i = 0; i < concurrency; ++i) {
    auto start = v.begin() + i * jump;
    auto end = i + 1 == concurrency ? v.end() : start + jump;
    futs.push_back(std::async(std::launch::async, f, start, end,
                              std::forward<Args>(args)...));
  }

  std::vector<R> results;
  for (auto& fut : futs) results.push_back(fut.get());
  return results;
}

template<typename F, typename Container, typename... Args>
void parallel_for(int concurrency, F f, Container& v, Args&&... args) {
  std::vector<std::future<void>> futs;
  int jump = v.size() / concurrency;
  if (jump == 0) jump = 1;
  for (int i = 0; i < concurrency; ++i) {
    auto start = v.begin() + i * jump;
    auto end = i + 1 == concurrency ? v.end() : start + jump;
    futs.push_back(std::async(std::launch::async, f, start, end,
                              std::forward<Args>(args)...));
  }

  for (auto& fut : futs) fut.get();
}

template<typename F, typename... Args>
void parallel_ifor(int concurrency, F f, int lo, int hi, Args&&... args) {
  std::vector<std::future<void>> futs;
  int jump = (hi - lo) / concurrency;
  if (jump == 0) jump = 1;
  for (int i = 0; i < concurrency; ++i) {
    auto start = lo + i * jump;
    auto end = i + 1 == concurrency ? hi : start + jump;
    futs.push_back(std::async(std::launch::async, f, start, end,
                              std::forward<Args>(args)...));
  }

  for (auto& fut : futs) fut.get();
}
