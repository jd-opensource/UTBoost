#ifndef UTBOOST_INCLUDE_UTBOOST_UTILS_OMP_WRAPPER_H_
#define UTBOOST_INCLUDE_UTBOOST_UTILS_OMP_WRAPPER_H_

#ifdef _OPENMP

#include "UTBoost/utils/log_wrapper.h"

#include <omp.h>

#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
#pragma warning(disable : 4068)  // disable unknown pragma warning
#endif

/*!
 * \brief Set the number of threads.
 *        If num_threads is less than 0, use maximum number of threads.
 * \param num_threads number of thread
 */
inline void OMP_SET_NUM_THREADS(int num_threads) {
  omp_set_num_threads(num_threads > 0 ? num_threads: omp_get_max_threads());
}

/*!
 * \brief Gets the number of threads in serial regions.
 * \return number of threads
 */
inline int OMP_GET_NUM_THREADS() {
  int n_threads = 1;
#pragma omp parallel
#pragma omp master
  { n_threads = omp_get_num_threads(); }
  return n_threads;
}

/*!
 * \brief Get current thread number in parallel region.
 * \return thread number
 */
inline int OMP_GET_THREAD_NUM() { return omp_get_thread_num(); }


class ThreadExceptionHelper {
 public:
  ThreadExceptionHelper() {
    ex_ptr_ = nullptr;
  }

  ~ThreadExceptionHelper() {
    ReThrow();
  }
  void ReThrow() {
    if (ex_ptr_ != nullptr) {
      std::rethrow_exception(ex_ptr_);
    }
  }
  void CaptureException() {
    // only catch first exception.
    if (ex_ptr_ != nullptr) { return; }
    std::unique_lock<std::mutex> guard(lock_);
    if (ex_ptr_ != nullptr) { return; }
    ex_ptr_ = std::current_exception();
  }

 private:
  std::exception_ptr ex_ptr_;
  std::mutex lock_;
};


#define OMP_INIT_EX() ThreadExceptionHelper omp_except_helper
#define OMP_LOOP_EX_BEGIN() try {
#define OMP_LOOP_EX_END()                 \
  }                                       \
  catch (...) {                           \
    omp_except_helper.CaptureException(); \
  }
#define OMP_THROW_EX() omp_except_helper.ReThrow()

#else
/**
 * To be compatible when OpenMp is unavailable.
 */
inline int OMP_GET_THREAD_NUM() { return 0; }
inline int OMP_GET_NUM_THREADS() { return 1; }
inline void OMP_SET_NUM_THREADS(int num_threads) {}

#endif // _OPENMP

#endif //UTBOOST_INCLUDE_UTBOOST_UTILS_OMP_WRAPPER_H_
