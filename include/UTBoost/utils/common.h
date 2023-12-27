/*!
 * Licensed under the MIT license.
 * See LICENSE file in the project root for full license information.
 * Created by Junjie Gao on 2023/3/1.
 */

#ifndef UTBOOST_INCLUDE_UTBOOST_UTILS_COMMON_H_
#define UTBOOST_INCLUDE_UTBOOST_UTILS_COMMON_H_

#include <limits>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>

#include "UTBoost/utils/omp_wrapper.h"
#include "UTBoost/utils/log_wrapper.h"


namespace UTBoost {

#define TIMER_START(_X) auto _X##_start = std::chrono::steady_clock::now(), _X##_stop = _X##_start
#define TIMER_STOP(_X) _X##_stop = std::chrono::steady_clock::now()
#define TIMER_SEC(_X)                                                                              \
    (0.000001 *                                                                                    \
     std::chrono::duration_cast<std::chrono::microseconds>(_X##_stop - _X##_start).count())

inline static std::vector<std::string> Compact(const std::vector<std::string> &tokens){
  std::vector<std::string> compacted;
  for(const auto & token : tokens) {
    if (!token.empty()) {
      compacted.push_back(token);
    }
  }
  return compacted;
}

inline static std::vector<std::string> Split(const std::string &str, char delim, const bool trim_empty = false){
  size_t pos, last_pos = 0, len;
  std::vector<std::string> tokens;

  while(true) {
    pos = str.find(delim, last_pos);
    if (pos == std::string::npos) {
      pos = str.size();
    }

    len = pos - last_pos;
    if ( !trim_empty || len != 0) {
      tokens.push_back(str.substr(last_pos, len));
    }

    if (pos == str.size()) {
      break;
    } else {
      last_pos = pos + 1;
    }
  }

  return tokens;
}

inline static std::string Join(const std::vector<std::string> &tokens, const std::string &delim, const bool trim_empty = false){
  if(trim_empty) {
    return Join(Compact(tokens), delim, false);
  } else {
    std::stringstream ss;
    for(size_t i=0; i<tokens.size()-1; ++i) {
      ss << tokens[i] << delim;
    }
    ss << tokens[tokens.size()-1];

    return ss.str();
  }
}

inline static const char* SkipSpaceAndTab(const char* p) {
  while (*p == ' ' || *p == '\t') {
    ++p;
  }
  return p;
}

/*!
 * Imbues the stream with the C locale.
 */
static void C_stringstream(std::stringstream &ss) {
  ss.imbue(std::locale::classic());
}

template<typename T>
inline static std::string Join(const std::vector<T>& strs, const char* delimiter, const bool force_C_locale = false) {
  if (strs.empty()) {
    return std::string("");
  }
  std::stringstream str_buf;
  if (force_C_locale) {
    C_stringstream(str_buf);
  }
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << strs[0];
  for (size_t i = 1; i < strs.size(); ++i) {
    str_buf << delimiter;
    str_buf << strs[i];
  }
  return str_buf.str();
}

inline static std::string Repeat(const std::string &str, unsigned int times){
  std::stringstream ss;
  for(unsigned int i=0; i<times; ++i) {
    ss << str;
  }
  return ss.str();
}

inline static bool StartsWith(const std::string& str, const std::string prefix) {
  if (str.substr(0, prefix.size()) == prefix) {
    return true;
  } else {
    return false;
  }
}

inline static std::string ToUpper(const std::string &str){
  std::string s(str);
  std::transform(s.begin(), s.end(), s.begin(), toupper);
  return s;
}

template <typename T>
inline static std::vector<T*> Vector2Ptr(std::vector<std::vector<T>>& data) {
  std::vector<T*> ptr(data.size());

  for (int i = 0; i < static_cast<int>(data.size()); ++i) {
    ptr[i] = data[i].data();
  }
  return ptr;
}

template<typename T>
inline std::vector<const T*> ConstPtrInVectorWrapper(std::vector<std::unique_ptr<T>>& input) {

  std::vector<const T*> ret;
  for (size_t i = 0; i < input.size(); ++i) {

    ret.push_back(input.at(i).get());
  }
  return ret;
}

template<typename T>
inline static size_t ArgMax(const std::vector<T> &array) {

  if (array.empty()) { return 0; }

  size_t arg_max = 0;

  for (size_t i = 0; i < array.size(); ++i) {
    if (array[i] > array[arg_max]) {
      arg_max = i;
    }

  }
  return arg_max;
}

template<typename _Iter> inline
static typename std::iterator_traits<_Iter>::value_type* IteratorValType(_Iter) {
  return (0);
}

template<typename _RanIt, typename _Pr, typename _VTRanIt> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred, _VTRanIt*) {
  size_t len = _Last - _First;
  const size_t kMinInnerLen = 1024;
  int num_threads = OMP_GET_NUM_THREADS();
  if (len <= kMinInnerLen || num_threads <= 1) {
    std::sort(_First, _Last, _Pred);
    return;
  }
  size_t inner_size = (len + num_threads - 1) / num_threads;
  inner_size = std::max(inner_size, kMinInnerLen);
  num_threads = static_cast<int>((len + inner_size - 1) / inner_size);
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < num_threads; ++i) {
    size_t left = inner_size*i;
    size_t right = left + inner_size;
    right = std::min(right, len);
    if (right > left) {
      std::sort(_First + left, _First + right, _Pred);
    }
  }
  // Buffer for merge.
  std::vector<_VTRanIt> temp_buf(len);
  _RanIt buf = temp_buf.begin();
  size_t s = inner_size;
  // Recursive merge
  while (s < len) {
    int loop_size = static_cast<int>((len + s * 2 - 1) / (s * 2));
#pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < loop_size; ++i) {
      size_t left = i * 2 * s;
      size_t mid = left + s;
      size_t right = mid + s;
      right = std::min(len, right);
      if (mid >= right) { continue; }
      std::copy(_First + left, _First + mid, buf + left);
      std::merge(buf + left, buf + mid, _First + mid, _First + right, _First + left, _Pred);
    }
    s *= 2;
  }
}

template<typename _RanIt, typename _Pr> inline
static void ParallelSort(_RanIt _First, _RanIt _Last, _Pr _Pred) {
  return ParallelSort(_First, _Last, _Pred, IteratorValType(_First));
}

template<typename T, typename T2>
inline static std::vector<T2> ArrayCast(const std::vector<T>& arr) {
  std::vector<T2> ret;
  for (size_t i = 0; i < arr.size(); ++i) {
    ret.push_back(static_cast<T2>(arr[i]));
  }
  return ret;
}

template<typename T>
inline static std::string ArrayToString(const std::vector<T>& arr, char delimiter) {
  if (arr.empty()) {
    return std::string("");
  }
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << arr[0];
  for (size_t i = 1; i < arr.size(); ++i) {
    str_buf << delimiter;
    str_buf << arr[i];
  }
  return str_buf.str();
}

template<typename T>
inline static std::string ArrayToString(const std::vector<T>& arr, size_t n, char delimiter) {
  if (arr.empty() || n == 0) {
    return std::string("");
  }
  std::stringstream str_buf;
  str_buf << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  str_buf << arr[0];
  for (size_t i = 1; i < std::min(n, arr.size()); ++i) {
    str_buf << delimiter;
    str_buf << arr[i];
  }
  return str_buf.str();
}

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter, size_t n) {
  std::vector<std::string> strs = Split(str, delimiter);
  if (strs.size() != n) {
    Log::Error("StringToArray error, size doesn't match.");
  }
  std::vector<T> ret(n);
  if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    for (size_t i = 0; i < n; ++i) {
      ret[i] = static_cast<T>(std::stod(strs[i]));
    }
  } else {
    for (size_t i = 0; i < n; ++i) {
      ret[i] = static_cast<T>(std::stol(strs[i]));
    }
  }
  return ret;
}

template<typename T>
inline static std::vector<T> StringToArray(const std::string& str, char delimiter) {
  std::vector<std::string> strs = Split(str, delimiter);
  std::vector<T> ret;
  if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
    for (const auto& s : strs) {
      ret.push_back(static_cast<T>(std::stod(s)));
    }
  } else {
    for (const auto& s : strs) {
      ret.push_back(static_cast<T>(std::stol(s)));
    }
  }
  return ret;
}

inline static float AvoidInf(float x) {
  if (std::isnan(x)) {
    return 0.0f;
  } else if (x >= 1e38) {
    return 1e38f;
  } else if (x <= -1e38) {
    return -1e38f;
  } else {
    return x;
  }
}

template<typename T>
inline static const char* Atoi(const char* p, T* out) {
  int sign;
  T value;
  while (*p == ' ') {
    ++p;
  }
  sign = 1;
  if (*p == '-') {
    sign = -1;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  for (value = 0; *p >= '0' && *p <= '9'; ++p) {
    value = value * 10 + (*p - '0');
  }
  *out = static_cast<T>(sign * value);
  while (*p == ' ') {
    ++p;
  }
  return p;
}

template<typename T>
inline static double Square(T base) {
  return base*base;
}

template<typename T>
inline static double Pow(T base, int power) {
  if (power < 0) {
    return 1.0 / Pow(base, -power);
  } else if (power == 0) {
    return 1;
  } else if (power % 2 == 0) {
    return Pow(base*base, power / 2);
  } else if (power % 3 == 0) {
    return Pow(base*base*base, power / 3);
  } else {
    return base * Pow(base, power - 1);
  }
}


inline static char tolower(char in) {
  if (in <= 'Z' && in >= 'A')
    return in - ('Z' - 'z');
  return in;
}

inline static const char* Atof(const char* p, double* out) {
  int frac;
  double sign, value, scale;
  *out = NAN;
  // Skip leading white space, if any.
  while (*p == ' ') {
    ++p;
  }
  // Get sign, if any.
  sign = 1.0;
  if (*p == '-') {
    sign = -1.0;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  // is a number
  if ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E') {
    // Get digits before decimal point or exponent, if any.
    for (value = 0.0; *p >= '0' && *p <= '9'; ++p) {
      value = value * 10.0 + (*p - '0');
    }

    // Get digits after decimal point, if any.
    if (*p == '.') {
      double right = 0.0;
      int nn = 0;
      ++p;
      while (*p >= '0' && *p <= '9') {
        right = (*p - '0') + right * 10.0;
        ++nn;
        ++p;
      }
      value += right / Pow(10.0, nn);
    }

    // Handle exponent, if any.
    frac = 0;
    scale = 1.0;
    if ((*p == 'e') || (*p == 'E')) {
      uint32_t expon;
      // Get sign of exponent, if any.
      ++p;
      if (*p == '-') {
        frac = 1;
        ++p;
      } else if (*p == '+') {
        ++p;
      }
      // Get digits of exponent, if any.
      for (expon = 0; *p >= '0' && *p <= '9'; ++p) {
        expon = expon * 10 + (*p - '0');
      }
      if (expon > 308) expon = 308;
      // Calculate scaling factor.
      while (expon >= 50) { scale *= 1E50; expon -= 50; }
      while (expon >= 8) { scale *= 1E8;  expon -= 8; }
      while (expon > 0) { scale *= 10.0; expon -= 1; }
    }
    // Return signed and scaled floating point result.
    *out = sign * (frac ? (value / scale) : (value * scale));
  } else {
    size_t cnt = 0;
    while (*(p + cnt) != '\0' && *(p + cnt) != ' '
        && *(p + cnt) != '\t' && *(p + cnt) != ','
        && *(p + cnt) != '\n' && *(p + cnt) != '\r'
        && *(p + cnt) != ':') {
      ++cnt;
    }
    if (cnt > 0) {
      std::string tmp_str(p, cnt);
      std::transform(tmp_str.begin(), tmp_str.end(), tmp_str.begin(), tolower);
      if (tmp_str == std::string("na") || tmp_str == std::string("nan") ||
          tmp_str == std::string("null")) {
        *out = NAN;
      } else if (tmp_str == std::string("inf") || tmp_str == std::string("infinity")) {
        *out = sign * 1e308;
      } else {
        Log::Error("Unknown token %s in data file", tmp_str.c_str());
      }
      p += cnt;
    }
  }

  while (*p == ' ') {
    ++p;
  }

  return p;
}


inline static size_t GetLine(const char* str) {
  auto start = str;
  while (*str != '\0' && *str != '\n' && *str != '\r') {
    ++str;
  }
  return str - start;
}

inline static const char* SkipNewLine(const char* str) {
  if (*str == '\r') {
    ++str;
  }
  if (*str == '\n') {
    ++str;
  }
  return str;
}

template<typename T>
inline static void Clip(T &base, T low, T high) {
  if (base < low) base = low;
  if (base > high) base = high;
}

template<typename T>
inline static double Trapz(std::vector<std::pair<T, T>> &xy) {
  double ret = 0.0f;
  for (int i = 0; i < xy.size() - 1; ++i) {
    double diff = static_cast<double>(xy[i + 1].first - xy[i].first);
    if (std::abs(diff) < 1e-10) continue;
    ret += static_cast<double>(xy[i + 1].second + xy[i].second) * diff / 2.0f;
  }
  return ret;
}

}

#endif //UTBOOST_INCLUDE_UTBOOST_UTILS_COMMON_H_
