// Compacts an entire leveldb database.
//
// Compile with
// cd .. && g++ -pthread -Wl,-soname -Ileveldb -ICOS513-Finance -Ileveldb/include -std=c++0x -fno-builtin-memcmp -pthread -DOS_LINUX -DLEVELDB_PLATFORM_POSIX -DLEVELDB_ATOMIC_PRESENT -DSNAPPY -O2 -DNDEBUG -fPIC -lsnappy leveldb/db/builder.cc leveldb/db/c.cc leveldb/db/dbformat.cc leveldb/db/db_impl.cc leveldb/db/db_iter.cc leveldb/db/dumpfile.cc leveldb/db/filename.cc leveldb/db/log_reader.cc leveldb/db/log_writer.cc leveldb/db/memtable.cc leveldb/db/repair.cc leveldb/db/table_cache.cc leveldb/db/version_edit.cc leveldb/db/version_set.cc leveldb/db/write_batch.cc leveldb/table/block_builder.cc leveldb/table/block.cc leveldb/table/filter_block.cc leveldb/table/format.cc leveldb/table/iterator.cc leveldb/table/merger.cc leveldb/table/table_builder.cc leveldb/table/table.cc leveldb/table/two_level_iterator.cc leveldb/util/arena.cc leveldb/util/bloom.cc leveldb/util/cache.cc leveldb/util/coding.cc leveldb/util/comparator.cc leveldb/util/crc32c.cc leveldb/util/env.cc leveldb/util/env_posix.cc leveldb/util/filter_policy.cc leveldb/util/hash.cc leveldb/util/histogram.cc leveldb/util/logging.cc leveldb/util/options.cc leveldb/util/status.cc leveldb/port/port_posix.cc COS513-Finance/leveldb_compacter.cpp -o leveldb_compacter leveldb/helpers/memenv/memenv.cc -lsnappy && mv leveldb_compacter COS513-Finance
#include <cstdlib>
#include <iostream>
#include <future>
#include <thread>
#include <vector>
#include "leveldb/db.h"

#define QCHECK_OK(status, vexit) do {                  \
    if (!(status).ok()) { \
      std::cerr << __FILE__ << ": " << __func__ << ": " << __LINE__ << ": " \
                << (status).ToString() << std::endl; \
      if (vexit) std::exit(EXIT_FAILURE);              \
    } \
  } while(0)

#define CHECK_OK(status) QCHECK_OK(status, true)

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "Usage: leveldb_compacter dbpath" << std::endl;
    return 1;
  }
  leveldb::DB* db;
  leveldb::Status open = leveldb::DB::Open({}, argv[1], &db);
  QCHECK_OK(open, false);
  if (!open.ok()) {
    std::cout << "Attempting to repair db" << std::endl;
    CHECK_OK(leveldb::RepairDB(argv[1], {}));
    return 0;
  }

  std::cout << "Database open" << std::endl;
  leveldb::Range all("", "9999999999999999999999999999999999999999");
  uint64_t size;

  db->GetApproximateSizes(&all, 1, &size);
  std::cout << "Before compaction size: " << size / 1000 << " KB" << std::endl;

  std::cout << "Compacting..." << std::flush;

  db->CompactRange(NULL, NULL);
  std::cout << "   DONE" << std::endl;

  db->GetApproximateSizes(&all, 1, &size);
  std::cout << "After compaction size : " << size / 1000 << " KB" << std::endl;

  delete db;
  std::cout << "Database closed" << std::endl;
  return 0;
}
