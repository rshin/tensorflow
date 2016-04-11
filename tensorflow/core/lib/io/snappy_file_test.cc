#include "tensorflow/core/lib/io/snappy_file.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(SnappyFile, WriteBuffering) {
  Env* env = Env::Default();

  {
    string fname = testing::TmpDir() + "/buffered.sz";
    WritableFile* file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));
    io::SnappyWritableFile snappy_file(file);

    for (int i = 0; i < 10000; i++) {
      snappy_file.Append("aaaaaaaaaa");
    }
    snappy_file.Close();
  }

  {
    string fname = testing::TmpDir() + "/unbuffered.sz";
    WritableFile* file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));
    io::SnappyWritableFile snappy_file(file);

    string buffer;
    for (int i = 0; i < 10000; i++) {
      buffer.append("aaaaaaaaaa");
    }
    snappy_file.Append(buffer);
    snappy_file.Close();
  }

  string snappy_buffered, snappy_unbuffered;
  TF_CHECK_OK(ReadFileToString(env, testing::TmpDir() + "/buffered.sz", &snappy_buffered));
  TF_CHECK_OK(ReadFileToString(env, testing::TmpDir() + "/unbuffered.sz", &snappy_unbuffered));

  EXPECT_EQ(snappy_buffered, snappy_unbuffered);
}

TEST(SnappyFile, RoundTrip) {
  Env* env = Env::Default();

  string fname = testing::TmpDir() + "/numbers.sz";
  {
    WritableFile* file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));
    io::SnappyWritableFile snappy_file(file);

    char dst[4];
    for (int i = 0; i < 1234567; i++) {
      core::EncodeFixed32(dst, i);
      TF_CHECK_OK(snappy_file.Append({dst, 4}));
    }
    TF_CHECK_OK(snappy_file.Close());
  }

  {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    io::SnappyRandomAccessFile snappy_file(file);

    std::vector<int> ints(1234567);
    std::iota(ints.begin(), ints.end(), 0);
    std::random_shuffle(ints.begin(), ints.end());
    ints[0] = 0;

    char scratch[4];
    StringPiece result;
    for (int i = 0; i < 12345; i++) {
      TF_CHECK_OK(snappy_file.Read(ints[i] * 4, 4, &result, scratch));
      ASSERT_EQ(ints[i], core::DecodeFixed32(scratch));
    }
  }
}

TEST(SnappyFile, Offset) {
  Env* env = Env::Default();

  string fname = testing::TmpDir() + "/letters.sz";
  {
    WritableFile* file;
    TF_CHECK_OK(env->NewWritableFile(fname, &file));
    io::SnappyWritableFile snappy_file(file);

    for (int i = 0; i < 1234567; i++) {
      TF_CHECK_OK(snappy_file.Append("abcdefghijklmnopqrstuvw"));;
    }
    TF_CHECK_OK(snappy_file.Close());
  }

  {
    RandomAccessFile* file;
    TF_CHECK_OK(env->NewRandomAccessFile(fname, &file));
    io::SnappyRandomAccessFile snappy_file(file);

    std::vector<int> ints(1234567 - 23);
    std::iota(ints.begin(), ints.end(), 0);
    std::random_shuffle(ints.begin(), ints.end());
    ints[0] = 0;

    char scratch[23];
    StringPiece result;
    const char reference[] = "abcdefghijklmnopqrstuvwabcdefghijklmnopqrstuvw";
    for (int i = 0; i < 12345; i++) {
      TF_CHECK_OK(snappy_file.Read(ints[i], 23, &result, scratch));
      ASSERT_EQ(memcmp(&reference[ints[i] % 23], result.data(), 23), 0);
    }
  }

}

}  // namespace tensorflow
