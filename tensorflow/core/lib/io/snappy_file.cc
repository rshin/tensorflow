#include "tensorflow/core/lib/io/snappy_file.h"

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/crc32c.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/snappy.h"

namespace tensorflow {
namespace io {

static const size_t kChunkMax = 16777215;
static const size_t kUncompressedChunkMax = 65536;
static const char kSnappyIdentifier[] = "sNaPpY";
static const uint8 kCompressedChunk = 0x00;
static const uint8 kUncompressedChunk = 0x01;
static const uint8 kStreamIdentifierChunk = 0xff;

SnappyWritableFile::SnappyWritableFile(WritableFile* dest)
    : dest_(dest), header_written_(false) {}

SnappyWritableFile::~SnappyWritableFile() {
  if (dest_) {
    Close();
  }
}

Status SnappyWritableFile::Append(const StringPiece& data) {
  if (!header_written_) {
    Status s = WriteChunk(kStreamIdentifierChunk, kSnappyIdentifier);
    if (!s.ok()) {
      return s;
    }
    header_written_ = true;
  }

  StringPiece uncompressed_chunk;
  Status s = Status::OK();
  for (size_t i = 0; i < data.size();) {
    if (data.size() - i + unwritten_remainder_.size() < kUncompressedChunkMax) {
      unwritten_remainder_.append(data.data() + i, data.size() - i);
      break;
    }

    if (!unwritten_remainder_.empty()) {
      CHECK_LT(unwritten_remainder_.size(), kUncompressedChunkMax);

      StringPiece additional_data =
          data.substr(i, kUncompressedChunkMax - unwritten_remainder_.size());
      unwritten_remainder_.append(additional_data.data(),
                                  additional_data.size());
      uncompressed_chunk.set(unwritten_remainder_.data(),
                             unwritten_remainder_.size());
      i += additional_data.size();
    } else {
      uncompressed_chunk = data.substr(i, kUncompressedChunkMax);
      i += uncompressed_chunk.size();
    }

    s = WriteCompressedChunk(uncompressed_chunk);
    unwritten_remainder_.clear();
    if (!s.ok()) {
      return s;
    }
  }
  return s;
}

Status SnappyWritableFile::Close() {
  Status s = WriteRemainder();
  if (!s.ok()) {
    return s;
  }
  s = dest_->Close();
  dest_ = NULL;
  return s;
}

Status SnappyWritableFile::Flush() {
  Status s = WriteRemainder();
  if (!s.ok()) {
    return s;
  }
  return dest_->Flush();
}

Status SnappyWritableFile::Sync() {
  Status s = WriteRemainder();
  if (!s.ok()) {
    return s;
  }
  return dest_->Sync();
}

Status SnappyWritableFile::WriteChunk(uint8 id, const StringPiece& data) {
  char header[4];
  header[0] = id;

  size_t data_size = data.size();
  CHECK_LE(data_size, kChunkMax) << "data is too big";

  if (port::kLittleEndian) {
    uint32 data_size_uint32 = static_cast<uint32>(data_size);
    memcpy(header + 1, &data_size_uint32, 3);
  } else {
    header[1] = data_size & 0xff;
    header[2] = (data_size >> 8) & 0xff;
    header[3] = (data_size >> 16) & 0xff;
  }

  Status s = dest_->Append(StringPiece(header, sizeof(header)));
  if (!s.ok()) {
    return s;
  }
  return dest_->Append(data);
}

Status SnappyWritableFile::WriteCompressedChunk(const StringPiece& data) {
  // Place to store output of Snappy_Compress that is run on |data|.
  static string compressed_data;
  // crc32(data) + compressed_data
  static string chunk;

  chunk.clear();
  core::PutFixed32(&chunk,
                   crc32c::Mask(crc32c::Value(data.data(), data.size())));
  if (port::Snappy_Compress(data.data(), data.size(), &compressed_data) &&
      compressed_data.size() < data.size() - (data.size() / 8u)) {
    chunk.append(compressed_data);
    return WriteChunk(kCompressedChunk, chunk);
  } else {
    // Snappy not supported, or data compressed by less than 12.5%
    chunk.append(data.data(), data.size());
    return WriteChunk(kUncompressedChunk, chunk);
  }
}

Status SnappyWritableFile::WriteRemainder() {
  Status s = Status::OK();
  if (!unwritten_remainder_.empty()) {
    s = WriteCompressedChunk(unwritten_remainder_);
    unwritten_remainder_.clear();
  }
  return s;
}

SnappyRandomAccessFile::SnappyRandomAccessFile(RandomAccessFile* src)
    : src_(src),
      raw_to_snappy_pos_({{0, 0}}),
      snappy_pos_(0),
      raw_pos_(0),
      cached_pos_(kuint64max) {}

Status SnappyRandomAccessFile::Read(uint64 offset, size_t n,
                                    StringPiece* result, char* scratch) const {
  mutex_lock lock(global_mutex_);

  // Read forward to build map
  string local_storage;
  Status s = BuildOffsetMap(offset, &local_storage);
  if (!s.ok()) {
    return s;
  }

  // Find block where raw offset starts
  auto offset_it = raw_to_snappy_pos_.upper_bound(offset);
  if (raw_to_snappy_pos_.end() == offset_it) {
    return errors::OutOfRange("offset ", offset,
                              " beyond end of uncompressed input");
  }

  // Read block and decompress it
  char* dst = scratch;
  while (n > 0 && offset_it != raw_to_snappy_pos_.end()) {
    // chunk_raw_offset = upper bound on raw offset found inside this block
    uint32 chunk_raw_offset = (--offset_it)->first;
    uint32 chunk_offset = (++offset_it)->second;

    if (chunk_offset != cached_pos_) {
      uint8 chunk_type;
      uint32 chunk_size;
      s = ReadSnappyChunkHeader(chunk_offset, &chunk_type, &chunk_size);
      if (!s.ok()) {
        return s;
      }

      if (chunk_type != kCompressedChunk && chunk_type != kUncompressedChunk) {
        continue;
      }

      // Read the chunk
      storage_.resize(chunk_size);
      s = src_->Read(chunk_offset + 4, chunk_size, result, &storage_[0]);
      if (!s.ok()) {
        return s;
      }
      if (result->size() != chunk_size) {
        return errors::DataLoss("truncated chunk at ", chunk_offset, ", read ",
                                result->size(), " bytes instead of ",
                                chunk_size, " bytes");
      }

      StringPiece result_without_crc = result->substr(4, result->size() - 4);

      // Decompress chunk if necessary
      StringPiece uncompressed_data;
      if (chunk_type == kCompressedChunk) {
        size_t uncompressed_size;
        if (!port::Snappy_GetUncompressedLength(result_without_crc.data(),
                                                result_without_crc.size(),
                                                &uncompressed_size)) {
          return errors::DataLoss(
              "Failed to read uncompressed length, or Snappy "
              "unavailable");
        }
        uncompressed_storage_.resize(uncompressed_size);
        if (!port::Snappy_Uncompress(result_without_crc.data(),
                                     result_without_crc.size(),
                                     &uncompressed_storage_[0])) {
          return errors::DataLoss(
              "Failed to uncompress Snappy data, or Snappy unavailable");
        }
        cached_data_ = uncompressed_storage_;
      } else {
        cached_data_ = result_without_crc;
      }

      // Verify chunk CRC
      uint32 masked_crc = core::DecodeFixed32(result->data());
      if (crc32c::Unmask(masked_crc) !=
          crc32c::Value(cached_data_.data(), cached_data_.size())) {
        return errors::DataLoss("corrupted chunk at ", chunk_offset);
      }
      cached_pos_ = chunk_offset;
    }

    // Skip to relevant part of uncompressed data
    CHECK_GE(offset, chunk_raw_offset);
    StringPiece uncompressed_data =
        cached_data_.substr(offset - chunk_raw_offset, n);

    // Assemble result
    memcpy(dst, uncompressed_data.data(), uncompressed_data.size());

    // Prepare to move onto next chunk
    dst += uncompressed_data.size();
    n -= uncompressed_data.size();
    offset += uncompressed_data.size();

    if (++offset_it == raw_to_snappy_pos_.end()) {
      s = BuildOffsetMap(offset, &local_storage);
      if (!s.ok()) {
        return s;
      }
      offset_it = raw_to_snappy_pos_.upper_bound(offset);
    }
  }

  *result = StringPiece(scratch, dst - scratch);
  if (n > 0) {
    LOG(INFO) << "Snappy: read less bytes than requested";
    return errors::OutOfRange("Snappy: Read less bytes than requested");
  }
  return s;
}

Status SnappyRandomAccessFile::BuildOffsetMap(uint64 offset,
                                              string* storage) const {
  while (raw_pos_ <= offset) {
    // Read chunk header
    uint8 chunk_type;
    uint32 chunk_size;
    Status s = ReadSnappyChunkHeader(snappy_pos_, &chunk_type, &chunk_size);
    if (errors::IsOutOfRange(s)) {
      // Now we should have:
      // snappy_pos_ = length of compressed data
      // raw_pos_ = length of uncompressed data
      return Status::OK();
    } else if (!s.ok()) {
      return s;
    }

    // Number of bytes this chunk will add to the uncompressed output
    size_t uncompressed_size = 0;
    if (chunk_type == kCompressedChunk) {
      StringPiece header;

      // 4 bytes for CRC-32, then 1 byte for uncompressed length varint
      CHECK_GE(chunk_size, 5u) << "compressed chunk too small";

      storage->resize(5);
      // +8: +4 to skip chunk header, +4 to skip CRC-32C
      s = src_->Read(snappy_pos_ + 8, 5, &header, &(*storage)[0]);
      if (!s.ok() && !errors::IsOutOfRange(s)) {
        return s;
      }
      if (!port::Snappy_GetUncompressedLength(header.data(), header.size(),
                                              &uncompressed_size)) {
        return errors::DataLoss(
            "Failed to read uncompressed length, or Snappy "
            "unavailable");
      }
    } else if (chunk_type == kUncompressedChunk) {
      CHECK_GE(chunk_size, 4u) << "uncompressed chunk too small";
      uncompressed_size = chunk_size - 4;  // -4 for CRC-32C
    } else if (chunk_type >= 0x02 && chunk_type <= 0x7f) {
      return errors::Unimplemented("Unknown Snappy chunk type: ", chunk_type);
    }

    // Advance to the next chunk
    raw_pos_ += uncompressed_size;
    if (uncompressed_size != 0) {
      raw_to_snappy_pos_[raw_pos_] = snappy_pos_;
    }
    snappy_pos_ += chunk_size + 4;
  }
  return Status::OK();
}

Status SnappyRandomAccessFile::ReadSnappyChunkHeader(uint64 offset,
                                                     uint8* chunk_type,
                                                     uint32* chunk_size) const {
  // Read chunk header
  char storage[4];
  StringPiece header;
  Status s = src_->Read(offset, 4, &header, storage);
  if (!s.ok()) {
    return s;
  }

  *chunk_type = header[0];
  *chunk_size =
      ((static_cast<uint32>(static_cast<unsigned char>(header[1]))) |
       (static_cast<uint32>(static_cast<unsigned char>(header[2])) << 8) |
       (static_cast<uint32>(static_cast<unsigned char>(header[3])) << 16));
  return s;
}

}  // namespace io
}  // namespace tensorflow
