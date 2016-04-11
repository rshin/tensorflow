#ifndef TENSORFLOW_LIB_IO_SNAPPY_FILE_H_
#define TENSORFLOW_LIB_IO_SNAPPY_FILE_H_

#include <map>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace io {

class SnappyWritableFile : public WritableFile {
 public:
  explicit SnappyWritableFile(WritableFile *dest);
  ~SnappyWritableFile() override;

  Status Append(const StringPiece &data) override;
  Status Close() override;
  Status Flush() override;
  Status Sync() override;

 private:
  Status WriteChunk(uint8 id, const StringPiece &data);
  Status WriteCompressedChunk(const StringPiece &data);
  Status WriteRemainder();

  WritableFile *dest_;
  bool header_written_;

  string unwritten_remainder_;

  TF_DISALLOW_COPY_AND_ASSIGN(SnappyWritableFile);
};

class SnappyRandomAccessFile : public RandomAccessFile {
 public:
  explicit SnappyRandomAccessFile(RandomAccessFile *src);

  Status Read(uint64 offset, size_t n, StringPiece *result,
              char *scratch) const override;

 private:
  Status BuildOffsetMap(uint64 offset, string *storage) const;
  Status ReadSnappyChunkHeader(uint64 offset, uint8 *chunk_type,
                               uint32 *chunk_size) const;
  const RandomAccessFile *src_;
  mutable mutex global_mutex_;

  // If map contains:
  //   a -> x
  //   b -> y
  // To read starting from [0, a), consult block at x.
  // To read starting from [a, b), consult block at y.
  mutable std::map<size_t, size_t> raw_to_snappy_pos_;
  mutable size_t snappy_pos_, raw_pos_, cached_pos_;

  mutable StringPiece cached_data_;
  mutable string storage_, uncompressed_storage_;
};

}  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_SNAPPY_FILE_H_
