// Compacts an entire leveldb database.
//
// Compile with
// g++ -o leveldb_compacter leveldb_compacter.cpp -O3 -L/n/fs/gcf/leveldb -I/n/fs/gcf/leveldb/include -Wall -lleveldb -pthread -std=c++11
#include <cstdlib>
#include <iostream>
#include <future>
#include <thread>
#include <vector>
#include "leveldb/db.h"

#define CHECK_OK(status) do {                   \
    if (!(status).ok()) { \
      std::cerr << __FILE__ << ": " << __func__ << ": " << __LINE__ << ": " \
                << (status).ToString() << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } while(0)

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: leveldb_compacter dbpath" << std::endl;
    return 1;
  }
  int concurrency = std::thread::hardware_concurrency();
  if (argc == 3) concurrency = std::atoi(argv[2]);
  std::cout << "Using concurrency " << concurrency << std::endl;

  leveldb::DB* db;
  CHECK_OK(leveldb::DB::Open({}, argv[1], &db));
  std::cout << "Database open" << std::endl;

  std::cout << "Compacting..." << std::flush;

  db->CompactRange(nullptr, nullptr);
  std::cout << "   DONE" << std::endl;

  delete db;
  std::cout << "Database closed" << std::endl;
  return 0;
}
