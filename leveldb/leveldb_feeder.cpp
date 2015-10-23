// Feeds data into a leveldb database from the tsv format, using the first
// column, when encoded as an unsigned long long, as the key.
//
// Compile with
// g++ -o leveldb_feeder leveldb_feeder.cpp -O3 -L/n/fs/gcf/leveldb -I/n/fs/gcf/leveldb/include -Wall -lleveldb -pthread -std=c++11
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>
#include "leveldb/db.h"
#include "parallel_for.hpp"

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

bool read_rec(leveldb::DB* db, std::string line, int& ctr) {
  auto tid = std::this_thread::get_id();
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
              << " Krecords" << std::endl;
  }
  return !status.ok();
}

typedef std::vector<std::string>::const_iterator It;
bool serial_read_files(It start, It end, leveldb::DB* db) {
  auto tid = std::this_thread::get_id();
  bool had_errors = false;
  int ctr = 0;
  for (auto it = start; it != end; ++it) {
    std::string filename;
    {
      std::stringstream cmd;
      cmd << "/tmp/tmp-cleaned-" << tid << "-"
          << std::distance(start, it) << ".tsv";
      filename = cmd.str();
      cmd.str("");
      cmd << "python /n/fs/gcf/COS513-Finance/clean_single_csv.py ";
      cmd << *it << " " << filename;
      auto cmdstr = cmd.str();
      std::cout << "tid " << tid << ": running " << cmdstr << std::endl;
      system(cmdstr.c_str());
    }

    std::ifstream in(filename);
    std::cout << "tid " << tid << ": started " << filename << std::endl;
    iterate_lines([&had_errors, db, &ctr](std::string str) {
        had_errors |= read_rec(db, std::move(str), ctr);
      }, in);
  }
  return had_errors;
}

void put_all_files(leveldb::DB* db, const std::vector<std::string>& files,
                   int concurrency) {
  auto errors = parallel_for(concurrency, serial_read_files, files, db);

  for (bool err : errors)
    if (err) {
      std::cerr << "Errors occured!" << std::endl;
      std::exit(1);
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
