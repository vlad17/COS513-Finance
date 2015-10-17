// Feeds data into a leveldb database from the tsv format, using the first
// column, when encoded as an unsigned long long, as the key.
//
// Compile with
// g++ -o leveldb_feeder leveldb_feeder.cpp -O3 -L/n/fs/gcf/leveldb -I/n/fs/gcf/leveldb/include -Wall -lleveldb -pthread -std=c++11
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <future>
#include <thread>
#include <vector>
#include "leveldb/db.h"

#define QCHECK_OK(status, tid) do {                 \
    if (!(status).ok()) { \
      std::cerr << __FILE__ << ": " << __func__ << ": " << __LINE__ << ": " \
                << "tid " << tid << ": " \
                << (status).ToString() << std::endl;            \
    } \
  } while(0)

#define CHECK_OK(status) do {                   \
    if (!(status).ok()) { \
      std::cerr << __FILE__ << ": " << __func__ << ": " << __LINE__ << ": " \
                << (status).ToString() << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

template<typename F>
void iterate_lines(F f, std::istream& in) {
  std::string next;
  while (std::getline(in, next)) {
    f(std::move(next));
    next.clear();
  }
}

std::vector<std::string> read_all() {
  std::vector<std::string> ret;
  iterate_lines([&](std::string str) { ret.push_back(std::move(str)); },
                std::cin);
  return ret;
}

bool read_rec(int tid, leveldb::DB* db, std::string line, int& ctr) {
  auto num_end = line.find('\t');
  if (num_end == std::string::npos) {
    std::cerr
      << "tid " << tid << ": "
      << "line " << ctr << ": doesn't have first column" << std::endl;
    return true;
  }
  auto status = db->Put({}, line.substr(0, num_end), line.substr(num_end + 1));
  QCHECK_OK(status, tid);
  if (++ctr % 100000 == 0) {
    std::cout << "\ttid " << tid << ": read " << ctr / 1000
              << "Krecords" << std::endl;
  }
  return !status.ok();
}

typedef std::vector<std::string>::const_iterator It;
bool serial_read_files(int tid, It start, It end, leveldb::DB* db) {
  bool had_errors = false;
  int ctr = 0;
  for (auto it = start; it != end; ++it) {


    std::ifstream in(*it);
    std::cout << "tid " << tid << ": started " << *it << std::endl;
    iterate_lines([&had_errors, tid, db, &ctr](std::string str) {
        had_errors |= read_rec(tid, db, std::move(str), ctr);
      }, in);
  }
  return had_errors;
}

void put_all_files(leveldb::DB* db, const std::vector<std::string>& files,
                   int concurrency) {
  std::vector<std::future<bool>> futs;
  int jump = files.size() / concurrency;
  for (int i = 0; i < concurrency; ++i) {
    auto start = files.begin() + i * jump;
    auto end = i + 1 == concurrency ? files.end() : start + jump;
    futs.push_back(std::async(std::launch::async, serial_read_files, i,
                              start, end, db));
  }

  for (auto& fut : futs) {
    if (fut.get()) {
      std::cerr << "Errors occured!" << std::endl;
      std::exit(1);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2 && argc != 3) {
    std::cout
      << "Usage: leveldb_feeder dbpath [concurrency]\n"
      << "Reads file paths from stdin, one per line, and writes rows from\n"
      << "the files into the db in parallel." << std::endl;
    return 1;
  }

  int concurrency = std::thread::hardware_concurrency();
  if (argc == 3) concurrency = std::atoi(argv[2]);
  std::cout << "Using concurrency " << concurrency << std::endl;

  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  CHECK_OK(leveldb::DB::Open(options, argv[1], &db));
  std::cout << "Database open" << std::endl;

  auto filenames = read_all();
  put_all_files(db, filenames, concurrency);

  delete db;
  std::cout << "Database closed" << std::endl;
  return 0;
}
