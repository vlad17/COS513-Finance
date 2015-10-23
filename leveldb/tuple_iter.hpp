// Tuple iterator, taken from SO:
// http://stackoverflow.com/questions/1198260/iterate-over-tuple
#include <tuple>
#include <utility>

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
  for_each(std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
  { }

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
  for_each(std::tuple<Tp...>& t, FuncT f)
  {
    f(std::get<I>(t));
    for_each<I + 1, FuncT, Tp...>(t, f);
  }


  template<template<std::size_t> class FuncT, std::size_t I = 0, typename Tuple, typename... Args>
  inline typename std::enable_if<I == std::tuple_size<Tuple>::value, void>::type
  for_eachi(Tuple&, Args&&...) // Unused arguments are given no names.
  { }

template<template<std::size_t> class FuncT, std::size_t I = 0, typename Tuple, typename... Args>
inline typename std::enable_if<I < std::tuple_size<Tuple>::value, void>::type
for_eachi(Tuple& t, Args&&... args)
  {
    FuncT<I>()(t, args...);
    for_eachi<FuncT, I + 1>(t, std::forward<Args>(args)...);

}

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
  for_each(const std::tuple<Tp...> &, FuncT) // Unused arguments are given no names.
  { }

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
  for_each(const std::tuple<Tp...>& t, FuncT f)
  {
    f(std::get<I>(t));
    for_each<I + 1, FuncT, Tp...>(t, f);
  }

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each2(std::tuple<Tp...> &, std::tuple<Tp...>&, FuncT) // Unused arguments are given no names.
  { }

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
for_each2(std::tuple<Tp...>& t1, std::tuple<Tp...>& t2, FuncT f)
  {
    f(std::get<I>(t1), std::get<I>(t2));
    for_each2<I + 1, FuncT, Tp...>(t1, t2, f);
  }


template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
for_each2(const std::tuple<Tp...> &, const std::tuple<Tp...>&, FuncT) // Unused arguments are given no names.
  { }

template<std::size_t I = 0, typename FuncT, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type
for_each2(const std::tuple<Tp...>& t1, const std::tuple<Tp...>& t2, FuncT f)
  {
    f(std::get<I>(t1), std::get<I>(t2));
    for_each2<I + 1, FuncT, Tp...>(t1, t2, f);
  }
