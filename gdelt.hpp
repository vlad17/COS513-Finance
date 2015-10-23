#include <string>
#include "tuple_iter.hpp"

template<typename T>
struct DiffClass {
  T value_;
  double distance(const DiffClass& other) {
    return fabs(value_ - other.value_);
  }
};

struct CategoryClass {
  int64_t value_; // hash of a string or other category
  double distance(const CategoryClass& other) {
    if (value_ == 0 || other.value_ == 0) return 1;
    return value_ != other.value_;
  }
};

template<typename T>
std::istream& operator>>(std::istream& in, DiffClass<T>& v) {
  in >> v.value_;
  int last = in.get();
  if (last != EOF && last != '\t' && last != '\r')
    throw std::invalid_argument("expected tab or EOF");
  return in;
}

std::istream& operator>>(std::istream& in, CategoryClass& v) {
  in >> v.value_;
  int last = in.get();
  if (last != EOF && last != '\t' && last != '\r')
    throw std::invalid_argument("expected tab or EOF");
  return in;
}

typedef DiffClass<int> DaysSinceEpoch;
typedef CategoryClass EventCode; // TODO needs custom distance
typedef CategoryClass QuadClass;
typedef DiffClass<double> GoldsteinScale;
typedef CategoryClass Actor1Name;
typedef CategoryClass Actor2Name;
typedef DiffClass<bool> IsRootEvent;
typedef DiffClass<int> NumMentions;
typedef DiffClass<int> NumSources;
typedef DiffClass<int> NumArticles;
typedef DiffClass<double> AvgTone;
typedef CategoryClass Actor1Geo;
typedef CategoryClass Actor2Geo;
typedef CategoryClass BaseURL;

typedef std::tuple<DaysSinceEpoch, EventCode, QuadClass,
                   GoldsteinScale, Actor1Name, Actor2Name,
                   IsRootEvent, NumMentions, NumSources, NumArticles,
                   AvgTone, Actor1Geo, Actor2Geo, BaseURL> GDELTMini;

template<std::size_t S>
struct read_cell { void operator()(GDELTMini& m, std::istream& in) {
  in >> std::get<S>(m);
}};

std::istream& operator>>(std::istream& in, GDELTMini& gdm) {
  for_eachi<read_cell>(gdm, in);
  return in;
}

template<std::size_t I>
struct get_dist { void operator() (GDELTMini& m1, GDELTMini& m2, double& sum) {
  sum += std::get<I>(m1).distance(std::get<I>(m2));
}
};

double Rho(const GDELTMini& m1, const GDELTMini& m2) {
  double dist = 0;
  for_eachi<get_dist>(const_cast<GDELTMini&>(m1),
                      const_cast<GDELTMini&>(m2), dist);
  return dist;
}
