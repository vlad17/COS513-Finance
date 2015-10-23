// Uses leveldb to run the kmedoids algorithm.
//
// Compile with
// cd .. && g++ -pthread -Wl,-soname -Ileveldb -ICOS513-Finance -Ileveldb/include -std=c++0x -fno-builtin-memcmp -pthread -DOS_LINUX -DLEVELDB_PLATFORM_POSIX -DLEVELDB_ATOMIC_PRESENT -DSNAPPY -O2 -DNDEBUG -fPIC -lsnappy leveldb/db/builder.cc leveldb/db/c.cc leveldb/db/dbformat.cc leveldb/db/db_impl.cc leveldb/db/db_iter.cc leveldb/db/dumpfile.cc leveldb/db/filename.cc leveldb/db/log_reader.cc leveldb/db/log_writer.cc leveldb/db/memtable.cc leveldb/db/repair.cc leveldb/db/table_cache.cc leveldb/db/version_edit.cc leveldb/db/version_set.cc leveldb/db/write_batch.cc leveldb/table/block_builder.cc leveldb/table/block.cc leveldb/table/filter_block.cc leveldb/table/format.cc leveldb/table/iterator.cc leveldb/table/merger.cc leveldb/table/table_builder.cc leveldb/table/table.cc leveldb/table/two_level_iterator.cc leveldb/util/arena.cc leveldb/util/bloom.cc leveldb/util/cache.cc leveldb/util/coding.cc leveldb/util/comparator.cc leveldb/util/crc32c.cc leveldb/util/env.cc leveldb/util/env_posix.cc leveldb/util/filter_policy.cc leveldb/util/hash.cc leveldb/util/histogram.cc leveldb/util/logging.cc leveldb/util/options.cc leveldb/util/status.cc leveldb/port/port_posix.cc COS513-Finance/leveldb_kmedoids.cpp -o leveldb_kmedoids leveldb/helpers/memenv/memenv.cc -lsnappy && mv leveldb_kmedoids COS513-Finance
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <future>
#include <thread>
#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include "leveldb/db.h"
#include "parallel_for.hpp"
#include "bhtsne/gdelt.hpp"

void add_atomic_double(std::atomic<double>* d, double x) {
  double prev = d->load();
  while (!std::atomic_compare_exchange_weak(d, &prev, prev + x));
}

typedef std::vector<double> vd;
typedef std::vector<std::unique_ptr<std::atomic<double>>> vuad;
typedef std::vector<std::unique_ptr<std::atomic<int>>> vuai;

#define CHECK_OK(status) do {                   \
    if (!(status).ok()) { \
      std::cerr << __FILE__ << ": " << __func__ << ": " << __LINE__ << ": " \
                << (status).ToString() << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

#define CHECK(boolean) do { \
    if (!(boolean)) { \
      std::cerr << __FILE__ << ": " << __func__ << ": " << __LINE__ << ": " \
                << #boolean << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

std::unique_ptr<leveldb::Iterator> iter(leveldb::DB* db) {
  leveldb::ReadOptions ro;
  ro.fill_cache = false;
  return std::unique_ptr<leveldb::Iterator>(db->NewIterator(ro));
}

template<typename Time>
int secs(const Time& t1, const Time& t2) {
  return std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
}

static void RunKMedoids(const leveldb::Slice& begin,
                        const leveldb::Slice& end,
                        int K, leveldb::DB* db,
                        leveldb::DB* work_db, int concurrency,
                        std::ostream& ivar_out, std::ostream& cent_out);

int main(int argc, char* argv[]) {
  if (argc != 5 && argc != 7 && argc != 8) {
    std::cout
      << "Usage: leveldb_kmedoids dbpath K intravariance-out-file "
      << "kmedoids-out-file [start end] [concurrency]\n"
      << "Runs K-Medoid on the key range [start, end].\n"
      << "Prints intermediate centroids per iteration to the out files, "
      << "as well as within-cluster total distances (\"intravariances\")."
      << std::endl;
    return 1;
  }

  int concurrency = std::thread::hardware_concurrency();
  if (argc == 8) concurrency = std::atoi(argv[7]);
  std::cout << "Using concurrency " << concurrency << std::endl;

  int K = std::atoi(argv[2]);

  leveldb::DB* db;
  CHECK_OK(leveldb::DB::Open({}, argv[1], &db));
  std::cout << "Database open" << std::endl;

  leveldb::DB* work_db;
  leveldb::Options options;
  options.create_if_missing = true;
  CHECK_OK(leveldb::DB::Open(options, "/tmp/kmedoids-work", &work_db));
  std::cout << "Work database open" << std::endl;

  std::string start_base, end_base;
  leveldb::Slice start, end;
  if (argc > 5) {
    start = argv[5];
    end = argv[6];
  } else {
    auto it = iter(db);
    it->SeekToFirst();
    CHECK(it->Valid());
    start_base = it->key().ToString();

    it->SeekToLast();
    CHECK(it->Valid());
    end_base = it->key().ToString();

    start = start_base;
    end = end_base;
  }
  std::cout << "Running K-medoids on range [" << start.ToString()
            << ", " << end.ToString()
            << "]" << std::endl;

  std::ofstream ivar_out(argv[3]);
  std::ofstream cent_out(argv[4]);

  RunKMedoids(start, end, K, db, work_db,
              concurrency, ivar_out, cent_out);
  delete work_db;
  CHECK_OK(leveldb::DestroyDB("/tmp/kmedoids-work", {}));

  std::cout << "Closing databases" << std::endl;
  delete db;
}

static std::vector<std::string> uniform_init(const leveldb::Slice& begin,
                                             const leveldb::Slice& end,
                                             int K, int prefix_len = 0) {
  auto ibegin = std::stoull(begin.ToString().substr(prefix_len));
  auto iend = std::stoull(end.ToString().substr(prefix_len));
  std::vector<std::string> ret(K);
  for (auto i = ibegin; i < iend; i += (iend - ibegin) / K) {
    ret.push_back(begin.ToString().substr(0, prefix_len) + std::to_string(i));
  }
  return ret;
}

struct IterRange {
  std::unique_ptr<leveldb::Iterator> it;
  std::function<void(void)> reset;
  std::function<bool()> should_continue;
};

std::vector<IterRange> get_par_ranges(
    const std::vector<std::string>& splits, leveldb::DB* db,
    const leveldb::Slice& end) {
  std::vector<IterRange> ret(splits.size());
  for (int i = 0; i < splits.size(); ++i) {
    auto it = iter(db);
    auto start = splits[i];
    auto ptr = it.get();
    auto reset = [=]() {
      ptr->Seek(start);
      CHECK(ptr->Valid());
    };
    if (i == splits.size() - 1) {
      ret[i] = {std::move(it), std::move(reset), [=]() {
            return ptr->Valid() && ptr->key().compare(end) <= 0; }};
    } else {
      ret[i] = {std::move(it), std::move(reset), [=]() {
            return ptr->Valid() && ptr->key().compare(splits[i + 1]) < 0; }};
    }
  }
  return ret;
}

static std::vector<IterRange> range_init(const leveldb::Slice& begin,
                                         const leveldb::Slice& end,
                                         int K, leveldb::DB* db) {
  std::vector<IterRange> ret(K);
  for (int i = 0; i < K; ++i) {
    ret[i].it = iter(db);
    auto ptr = ret[i].it.get();
    ret[i].reset = [=]() {
      ptr->Seek(begin);
      CHECK(ptr->Valid());
    };
    ret[i].should_continue = [=]() {
      return ptr->Valid() && ptr->key().compare(end) <= 0;
    };
  }
  return ret;
}

static std::vector<IterRange> range_uniform(const leveldb::Slice& begin,
                                            const leveldb::Slice& end,
                                            int K, leveldb::DB* db,
                                            int size) {
  int jump = size / K;
  std::vector<IterRange> ret(K);
  if (jump == 0) {
    jump = 1;
    for (int i = size; i < K; ++i) {
      ret[i].reset = []() {};
      ret[i].should_continue = []() { return false; };
    }
    K = size;
  }
  auto it = iter(db);
  it->Seek(begin);
  CHECK(it->Valid());
  int i = 0;
  for (; i < K; ++i) {
    CHECK(it->Valid());
    CHECK(it->key().compare(end) < 0);
    ret[i].it = iter(db);
    std::string start = it->key().ToString();
    auto ptr = ret[i].it.get();
    ret[i].reset = [=]() {
      ptr->Seek(start);
      CHECK(ptr->Valid());
    };
    for (int j = 0; j < jump; ++j) {
      it->Next();
      CHECK(it->Valid());
    }
    std::string lend = it->key().ToString();
    if (i == K - 1) lend = end.ToString();
    ret[i].should_continue = [=]() {
      return ptr->Valid() && ptr->key().compare(lend) < 0;
    };
  }
  return ret;
}

static void read(const leveldb::Slice& strval, GDELTMini& val) {
  std::stringstream sstr;
  sstr.str(std::move(strval.ToString()));
  sstr >> val;
}

static void lookup(leveldb::DB* db, const leveldb::Slice& key, GDELTMini& val) {
  std::string strval;
  leveldb::Status status = db->Get({}, key, &strval);
  if (status.IsNotFound()) {
    std::cerr << "Key " << key.ToString() << " not found." << std::endl;
  }
  CHECK_OK(status);
  std::stringstream sstr;
  sstr.str(std::move(strval));
  sstr >> val;
}

template<typename T>
static std::ostream& operator<<(std::ostream& out,
                                const std::vector<T>& v) {
  for (const auto& s : v) {
    out << s << " ";
  }
  return out;
}

std::string get_centroid_prefix(int centroid, int K) {
  int n = ceil(log10(K + 1));
  auto str = std::to_string(centroid);
  if (centroid < 0) str.clear();
  if (n > str.length()) {
    str.insert(0, n - str.length(), '-');
  }
  return str;
}

int get_centroid(leveldb::DB* work_db, const leveldb::Slice& key, int K) {
  std::string str;
  auto keystr = get_centroid_prefix(-1, K) + key.ToString();
  auto status = work_db->Get({}, keystr, &str);
  if (status.IsNotFound()) return -1;
  CHECK_OK(status);
  return std::stoi(str);
}

void set_centroid(leveldb::DB* work_db, const leveldb::Slice& key,
                  int oldc, int newc, int K) {
  if (oldc == newc) return;
  auto rawkey = key.ToString();
  auto keystr = get_centroid_prefix(-1, K) + rawkey;
  CHECK_OK(work_db->Put({}, key, std::to_string(newc)));
  CHECK_OK(work_db->Delete({}, get_centroid_prefix(oldc, K) + rawkey));
  CHECK_OK(work_db->Put({}, get_centroid_prefix(newc, K) + rawkey, key));
}

typedef std::vector<IterRange>::iterator It;
void assign_closest_serial(It begin, It end, int K, leveldb::DB* work_db,
                           const std::vector<GDELTMini>& centroids,
                           vuad& totals, vuai& cluster_sizes) {
  GDELTMini current_row;
  for (auto itit = begin; itit != end; ++itit) {
    auto& range = *itit;
    for (range.reset(); range.should_continue(); range.it->Next()) {
      read(range.it->value(), current_row);
      int min = -1;
      double min_dist = std::numeric_limits<double>::infinity();
      for (int i = 0; i < K; ++i) {
        double d = Rho(current_row, centroids[i]);
        if (d < min_dist) {
          min = i;
          min_dist = d;
        }
      }
      CHECK(min >= 0);
      add_atomic_double(totals[min].get(), min_dist);
      auto i = cluster_sizes[min]->fetch_add(1) + 1;
      int prev = get_centroid(work_db, range.it->key(), K);
      set_centroid(work_db, range.it->key(), prev, min, K);
    }
  }
}

void assign_closest(std::vector<IterRange>& parvec,
                    vuad& totals,
                    int K,
                    leveldb::DB* work_db,
                    const std::vector<GDELTMini>& centroids,
                    int concurrency,
                    vuai& cluster_sizes) {
  for (auto& i : cluster_sizes) i->store(0);
  for (auto& d : totals) d->store(0);
  parallel_for(concurrency, assign_closest_serial,
               parvec, K, work_db, std::cref(centroids),
               std::ref(totals), std::ref(cluster_sizes));
}

typedef std::pair<std::string, GDELTMini> KV;
void update_medoids_serial(int idx, std::vector<IterRange>& its,
                           std::vector<IterRange>& jts,
                           std::vector<KV>& thread_best,
                           std::vector<double> thread_best_dist,
                           leveldb::DB* db) {
  auto& i = its[idx];
  auto& j = jts[idx];
  GDELTMini current, other;
  for (i.reset(); i.should_continue(); i.it->Next()) {
    lookup(db, i.it->value(), current);
    double tot = 0;
    for (j.reset(); j.should_continue(); j.it->Next()) {
      lookup(db, j.it->value(), other);
        tot += Rho(current, other);
    }
    if (tot < thread_best_dist[idx]) {
      thread_best_dist[idx] = tot;
      thread_best[idx] = std::make_pair(i.it->value().ToString(), current);
    }
  }
}

bool update_medoids(int concurrency, int K, leveldb::DB* db,
                    leveldb::DB* work_db,
                    std::vector<GDELTMini>& val_centroids,
                    std::vector<std::string>& key_centroids,
                    const vuad& distances,
                    const vuai& cluster_sizes) {
  std::vector<KV> thread_best(concurrency);
  std::vector<double> thread_best_dist(concurrency);

  bool changed = false;

  for (int cluster = 0; cluster < K; ++cluster) {
    double min_dist = distances[cluster]->load();
    auto current_key = get_centroid_prefix(cluster, K);
    auto next_key = get_centroid_prefix(cluster + 1, K);

    for (auto& d : thread_best_dist) d = min_dist;

    auto parvec_i = range_uniform(current_key, next_key, concurrency,
                                  work_db, cluster_sizes[cluster]->load());
    auto parvec_j = range_init(current_key, next_key, concurrency, work_db);

    parallel_ifor(concurrency, update_medoids_serial, std::ref(parvec_i),
                  std::ref(parvec_j), std::ref(thread_best),
                  std::ref(thread_best_dist), db);

    auto min = std::distance(thread_best_dist.begin(),
                            std::min_element(thread_best_dist.begin(),
                                             thread_best_dist.end()));
    if (thread_best_dist[min] < min_dist) {
      key_centroids[cluster] = thread_best[min].first;
      val_centroids[cluster] = thread_best[min].second;
    }

    if (true || (cluster + 1) % (K / 100) == 0) {
      std::cout << "      Cluster update " << (cluster + 1.0) * 100.0 / K
                << "% done" << std::endl;
    }
  }
  return changed;
}

void RunKMedoids(const leveldb::Slice& begin,
                 const leveldb::Slice& end,
                 int K, leveldb::DB* db,
                 leveldb::DB* work_db, int concurrency,
                 std::ostream& ivar_out, std::ostream& cent_out) {
  auto very_start = std::chrono::system_clock::now();
  auto key_centroids = uniform_init(begin, end, K);
  std::vector<GDELTMini> val_centroids(K);
  {
    auto it = iter(db);
    for (int i = 0; i < K; ++i) {
      it->Seek(key_centroids[i]);
      CHECK(it->Valid());
      read(it->value(), val_centroids[i]);
    }
  }

  std::cout << "Divying up range among threads... " << std::flush;
  auto parvec = get_par_ranges(uniform_init(begin, end, concurrency), db, end);
  std::cout << "DONE" << std::endl;

  int i = 0;
  bool centers_changed = true;
  vuad totals(K);
  vuai cluster_sizes(K);
  for (int i = 0; i < K; ++i) {
    cluster_sizes[i].reset(new std::atomic<int>);
    totals[i].reset(new std::atomic<double>);
  }

  while (centers_changed) {
    auto start = std::chrono::system_clock::now();
    assign_closest(parvec, totals, K, work_db, val_centroids, concurrency,
                   cluster_sizes);
    auto tot = std::accumulate(totals.begin(), totals.end(), 0.0,
                               [](double sum, typename vuad::value_type& d) {
                                 return sum + d->load();
                               });
    auto end = std::chrono::system_clock::now();
    std::cout << "Iteration " << ++i << " total intravariance " << tot;
    std::cout << "\n    Assigning medoids took " << secs(start, end)
              << "s" << std::endl;

    start = std::chrono::system_clock::now();
    for (auto& d : totals) {
      ivar_out << d->load() << " ";
    }
    ivar_out << std::endl;
    cent_out << key_centroids << std::endl;
    end = std::chrono::system_clock::now();
    std::cout << "    Saving medoids took " << secs(start, end)
              << " s" << std::endl;

    start = std::chrono::system_clock::now();
    centers_changed = update_medoids(concurrency, K, db, work_db,
                                     val_centroids, key_centroids,
                                     totals, cluster_sizes);
    end = std::chrono::system_clock::now();
    std::cout << "    Medoid update took " << secs(start, end)
              << " s" << std::endl;
    start = end;
  }
  auto very_end = std::chrono::system_clock::now();
  std::cout << "K-medoid clustering COMPLETE in "
            << secs(very_start, very_end) << " s" << std::endl;
}
