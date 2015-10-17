// Uses leveldb to run the kmedoids algorithm.
//
// Compile with
// g++ -o leveldb_kmedoids leveldb_kmedoids.cpp -O3 -L/n/fs/gcf/leveldb -I/n/fs/gcf/leveldb/include -Wall -lleveldb -pthread -std=c++11
#include <cstdlib>
#include <iostream>
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

int main(int argc, char* argv[]) {
  if (argc != 4 && argc != 5) {
    std::cout
      << "Usage: leveldb_kmedoids dbpath start end [concurrency]\n"
      << "Runs K-Medoid on the key range [start, end]." << std::endl;
    return 1;
  }
  int concurrency = std::thread::hardware_concurrency();
  if (argc == 3) concurrency = std::atoi(argv[2]);
  std::cout << "Using concurrency " << concurrency << std::endl;

  leveldb::DB* db;
  CHECK_OK(leveldb::DB::Open({}, argv[1], &db));
  std::cout << "Database open" << std::endl;

  auto start = argv[2];
  auto end = argv[3];
  std::cout << "Running K-medoids on range [" << start << ", " << end << "]"
            << std::endl;
