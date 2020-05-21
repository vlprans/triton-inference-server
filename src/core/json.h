// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

// This header can be used both within Triton server and externally
// (i.e. in source that interacts only via TRITONSERVER API). Status
// is handled differently in these two cases.
#if defined(TRITONJSON_INTERNAL_STATUS)

#include "src/core/status.h"
#define TRITONJSON_STATUSTYPE Status
#define TRITONJSON_STATUSRETURN(M) return Status(Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS Status::Success

#elif defined(TRITONJSON_TRITONSERVER_STATUS)

#include "src/core/tritonserver.h"
#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr

#else

#error "Must set TRITONJSON_INTERNAL_STATUS or TRITONJSON_TRITONSERVER_STATUS"

#endif

namespace nvidia { namespace inferenceserver {

#define TRITONJSON_DECL_COMMON_METHODS                                         \
  /* Add a value to an object. It is assumed that 'name' can be used           \
     by reference, it is the caller's responsibility to make sure the          \
     lifetime of 'name' extends at least as long as the object. */             \
  TRITONJSON_STATUSTYPE Add(const char* name, TritonJson::Value& value);       \
                                                                               \
  /* Add a copy of a string to an object. It is assumed that 'name' can be     \
     used by reference, it is the caller's responsibility to make sure the     \
     lifetime of 'name' extends at least as long as the object. */             \
  TRITONJSON_STATUSTYPE Add(const char* name, const std::string& value);       \
                                                                               \
  /* Add a copy of a string to an object. It is assumed that 'name' can be     \
     used by reference, it is the caller's responsibility to make sure the     \
     lifetime of 'name' extends at least as long as the object. */             \
  TRITONJSON_STATUSTYPE Add(const char* name, const char* value);              \
                                                                               \
  /* Add a signed int to an object. It is assumed that 'name' can be           \
     used by reference, it is the caller's responsibility to make sure the     \
     lifetime of 'name' extends at least as long as the object. */             \
  TRITONJSON_STATUSTYPE Add(const char* name, const int64_t value);            \
                                                                               \
  /* Add an unsigned int to an object. It is assumed that 'name' can be        \
     used by reference, it is the caller's responsibility to make sure the     \
     lifetime of 'name' extends at least as long as the object. */             \
  TRITONJSON_STATUSTYPE Add(const char* name, const uint64_t value);           \
                                                                               \
  /* Add a string value to an object. It is assumed that 'name' and 'value'    \
     can be used by reference, it is the caller's responsibility to make sure  \
     the lifetime of 'name' and 'value' extends at least as long as the        \
     object. */                                                                \
  TRITONJSON_STATUSTYPE AddStringRef(const char* name, const char* value);     \
                                                                               \
  /* Append a value to an array. */                                            \
  TRITONJSON_STATUSTYPE Append(TritonJson::Value& value);                      \
                                                                               \
  /* Append a copy of a string to an array. */                                 \
  TRITONJSON_STATUSTYPE Append(const std::string& value);                      \
                                                                               \
  /* Append a copy of a string to an array. */                                 \
  TRITONJSON_STATUSTYPE Append(const char* value);                             \
                                                                               \
  /* Append a signed int to an array. */                                       \
  TRITONJSON_STATUSTYPE Append(const int64_t value);                           \
                                                                               \
  /* Append an unsigned int to an array. */                                    \
  TRITONJSON_STATUSTYPE Append(const uint64_t value);                          \
                                                                               \
  /* Append a string value to an array. It is assumed that 'value' can be used \
     by reference, it is the caller's responsibility to make sure the          \
     lifetime of 'value' extends at least as long as the object. */            \
  TRITONJSON_STATUSTYPE AppendStringRef(const char* value);

#define TRITONJSON_DEFINE_COMMON_METHODS(CLS)                                  \
  TRITONJSON_STATUSTYPE CLS::Add(const char* name, TritonJson::Value& value)   \
  {                                                                            \
    if (!value_.IsObject()) {                                                  \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to add member to non-object"));           \
    }                                                                          \
    value_.AddMember(                                                          \
        rapidjson::Value(rapidjson::StringRef(name)).Move(), value.value_,     \
        Allocator());                                                          \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Add(const char* name, const std::string& value)   \
  {                                                                            \
    if (!value_.IsObject()) {                                                  \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to add member to non-object"));           \
    }                                                                          \
    rapidjson::Document::AllocatorType& allocator = Allocator();               \
    value_.AddMember(                                                          \
        rapidjson::Value(rapidjson::StringRef(name)).Move(),                   \
        rapidjson::Value(value.c_str(), value.size(), allocator).Move(),       \
        allocator);                                                            \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Add(const char* name, const char* value)          \
  {                                                                            \
    if (!value_.IsObject()) {                                                  \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to add member to non-object"));           \
    }                                                                          \
    rapidjson::Document::AllocatorType& allocator = Allocator();               \
    value_.AddMember(                                                          \
        rapidjson::Value(rapidjson::StringRef(name)).Move(),                   \
        rapidjson::Value(value, allocator).Move(), allocator);                 \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Add(const char* name, const int64_t value)        \
  {                                                                            \
    if (!value_.IsObject()) {                                                  \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to add member to non-object"));           \
    }                                                                          \
    value_.AddMember(                                                          \
        rapidjson::Value(rapidjson::StringRef(name)).Move(),                   \
        rapidjson::Value(value).Move(), Allocator());                          \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Add(const char* name, const uint64_t value)       \
  {                                                                            \
    if (!value_.IsObject()) {                                                  \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to add member to non-object"));           \
    }                                                                          \
    value_.AddMember(                                                          \
        rapidjson::Value(rapidjson::StringRef(name)).Move(),                   \
        rapidjson::Value(value).Move(), Allocator());                          \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::AddStringRef(const char* name, const char* value) \
  {                                                                            \
    if (!value_.IsObject()) {                                                  \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to add member to non-object"));           \
    }                                                                          \
    value_.AddMember(                                                          \
        rapidjson::Value(rapidjson::StringRef(name)).Move(),                   \
        rapidjson::StringRef(value), Allocator());                             \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Append(TritonJson::Value& value)                  \
  {                                                                            \
    if (!value_.IsArray()) {                                                   \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to append to non-array"));                \
    }                                                                          \
    value_.PushBack(value.value_, Allocator());                                \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Append(const std::string& value)                  \
  {                                                                            \
    if (!value_.IsArray()) {                                                   \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to append to non-array"));                \
    }                                                                          \
    rapidjson::Document::AllocatorType& allocator = Allocator();               \
    value_.PushBack(                                                           \
        rapidjson::Value(value.c_str(), value.size(), allocator).Move(),       \
        allocator);                                                            \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Append(const char* value)                         \
  {                                                                            \
    if (!value_.IsArray()) {                                                   \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to append to non-array"));                \
    }                                                                          \
    rapidjson::Document::AllocatorType& allocator = Allocator();               \
    value_.PushBack(rapidjson::Value(value, allocator).Move(), allocator);     \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Append(const int64_t value)                       \
  {                                                                            \
    if (!value_.IsArray()) {                                                   \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to append to non-array"));                \
    }                                                                          \
                                                                               \
    value_.PushBack(rapidjson::Value(value).Move(), Allocator());              \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::Append(const uint64_t value)                      \
  {                                                                            \
    if (!value_.IsArray()) {                                                   \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to append to non-array"));                \
    }                                                                          \
                                                                               \
    value_.PushBack(rapidjson::Value(value).Move(), Allocator());              \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }                                                                            \
                                                                               \
  TRITONJSON_STATUSTYPE CLS::AppendStringRef(const char* value)                \
  {                                                                            \
    if (!value_.IsArray()) {                                                   \
      TRITONJSON_STATUSRETURN(                                                 \
          std::string("JSON, attempt to append to non-array"));                \
    }                                                                          \
    value_.PushBack(rapidjson::StringRef(value), Allocator());                 \
    return TRITONJSON_STATUSSUCCESS;                                           \
  }

//
// A JSON parser/writer. Currently based on rapidjson but the intent
// is to provide an abstraction for JSON functions that make it easy
// to substiture a different JSON parser. Specifically for rapidjson
// the class is also designed to provide safe access and error
// reporting to avoid the cases where rapidjson would just abort the
// entire application (!).
//
class TritonJson {
 public:
  class Document;
  class Value;

  enum class ValueType {
    NULLVAL = rapidjson::kNullType,
    OBJECT = rapidjson::kObjectType,
    ARRAY = rapidjson::kArrayType,
  };

  //
  // Buffer filled by document when writing JSON representation.
  //
  class WriteBuffer {
   public:
    void Clear() { buffer_.Clear(); }
    const char* Base() const { return buffer_.GetString(); }
    size_t Size() const { return buffer_.GetSize(); }

   private:
    friend class Document;
    rapidjson::StringBuffer buffer_;
  };

  //
  // Top-level value representing the entire document.
  //
  class Document {
   public:
    explicit Document(ValueType type = ValueType::NULLVAL)
        : value_(static_cast<rapidjson::Type>(type))
    {
    }

    // Parse JSON into document.
    TRITONJSON_STATUSTYPE Parse(const char* base, const size_t size)
    {
      value_.Parse(base, size);
      if (value_.HasParseError()) {
        TRITONJSON_STATUSRETURN(std::string(
            "failed to parse the request JSON buffer: " +
            std::string(GetParseError_En(value_.GetParseError())) + " at " +
            std::to_string(value_.GetErrorOffset())));
      }
      return TRITONJSON_STATUSSUCCESS;
    }

    // Parse JSON into document.
    TRITONJSON_STATUSTYPE Parse(const std::string& json)
    {
      return Parse(json.data(), json.size());
    }

    // Write JSON representation for document into a 'buffer'.
    TRITONJSON_STATUSTYPE Write(WriteBuffer* buffer) const
    {
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer->buffer_);
      value_.Accept(writer);
      return TRITONJSON_STATUSSUCCESS;
    }

    TRITONJSON_DECL_COMMON_METHODS

   private:
    friend class Value;
    rapidjson::Document::AllocatorType& Allocator()
    {
      return value_.GetAllocator();
    }
    rapidjson::Document value_;
  };

  //
  // Represents a value within a document.
  //
  class Value {
   public:
    explicit Value(TritonJson::Document& document, ValueType type)
        : document_(&document), value_(static_cast<rapidjson::Type>(type))
    {
    }

    TRITONJSON_DECL_COMMON_METHODS

   private:
    friend class Document;
    rapidjson::Document::AllocatorType& Allocator()
    {
      return document_->Allocator();
    }

    // The document that will contain this value.
    TritonJson::Document* document_;
    rapidjson::Value value_;
  };
};

TRITONJSON_DEFINE_COMMON_METHODS(TritonJson::Document)
TRITONJSON_DEFINE_COMMON_METHODS(TritonJson::Value)

}}  // namespace nvidia::inferenceserver
