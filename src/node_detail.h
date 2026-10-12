#pragma once

#include "constants.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <node.h>

namespace genetic {
namespace detail {

static constexpr float MIN_VAL = 0.001f;

inline bool is_terminal(node::type t) {
  return t == node::type::variable || t == node::type::constant;
}

inline bool is_nonterminal(node::type t) { return !is_terminal(t); }

inline int arity(node::type t) {
  if (node::type::unary_begin <= t && t <= node::type::unary_end) {
    return 1;
  }
  if (node::type::binary_begin <= t && t <= node::type::binary_end) {
    return 2;
  }
  return 0;
}

// `data` assumed to be stored in col-major format
inline int evaluate_node(const node &n, const float *data, const uint64_t stride,
                    const uint64_t idx, const int batch_size, const float *in_a, const float *in_b, float *out) {
  if (n.t == node::type::constant) {
    for (int i = 0; i < batch_size; ++i) {
      out[i] = n.u.val;
    }
    return 0;
  } else if (n.t == node::type::variable) {
    for (int i = 0; i < batch_size; ++i) {
      out[i] = data[stride * n.u.fid + idx + i];
    }
    return 0;
  } else {
    // note: keep the case statements in alphabetical order under each category
    // of operators.
    switch (n.t) {
    // binary operators
    case node::type::add:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = in_a[i] + in_b[i];
      }
      return 2;
    case node::type::atan2:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = atan2f(in_a[i], in_b[i]);
      }
      return 2;
    case node::type::div:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = fabsf(in_b[i]) < MIN_VAL ? 1.0f : (in_a[i] / in_b[i]);
      }
      return 2;
    case node::type::fdim:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = fdimf(in_a[i], in_b[i]);
      }
      return 2;
    case node::type::gt:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = in_a[i] > in_b[i] ? 1.0f : 0.0f;
      }
      return 2;
    case node::type::max:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = fmaxf(in_a[i], in_b[i]);
      }
      return 2;
    case node::type::min:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = fminf(in_a[i], in_b[i]);
      }
      return 2;
    case node::type::mul:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = in_a[i] * in_b[i];
      }
      return 2;
    case node::type::pow:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = powf(in_a[i], in_b[i]);
      }
      return 2;
    case node::type::sub:
      for (int i = 0; i < batch_size; ++i) {
        out[i - 2 * ROW_BATCH_SIZE] = in_a[i] - in_b[i];
      }
      return 2;
    // unary operators
    case node::type::abs:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = fabsf(in_a[i]);
      }
      return 1;
    case node::type::acos:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = acosf(in_a[i]);
      }
      return 1;
    case node::type::acosh:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = acoshf(in_a[i]);
      }
      return 1;
    case node::type::asin:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = asinf(in_a[i]);
      }
      return 1;
    case node::type::asinh:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = asinhf(in_a[i]);
      }
      return 1;
    case node::type::atan:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = atanf(in_a[i]);
      }
      return 1;
    case node::type::atanh:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = atanhf(in_a[i]);
      }
      return 1;
    case node::type::cbrt:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = cbrtf(in_a[i]);
      }
      return 1;
    case node::type::cos:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = cosf(in_a[i]);
      }
      return 1;
    case node::type::cosh:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = coshf(in_a[i]);
      }
      return 1;
    case node::type::cube:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = in_a[i] * in_a[i] * in_a[i];
      }
      return 1;
    case node::type::exp:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = expf(in_a[i]);
      }
      return 1;
    case node::type::inv:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = fabsf(in_a[i]) < MIN_VAL ? 0.f : 1.f / in_a[i];
      }
      return 1;
    case node::type::log:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = fabsf(in_a[i]) < MIN_VAL ? 0.f : logf(fabsf(in_a[i]));
      }
      return 1;
    case node::type::neg:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = -in_a[i];
      }
      return 1;
    case node::type::rcbrt:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = static_cast<float>(1.0) / cbrtf(in_a[i]);
      }
      return 1;
    case node::type::rsqrt:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = static_cast<float>(1.0) / sqrtf(fabsf(in_a[i]));
      }
      return 1;
    case node::type::sin:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = sinf(in_a[i]);
      }
      return 1;
    case node::type::sinh:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = sinhf(in_a[i]);
      }
      return 1;
    case node::type::sq:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = in_a[i] * in_a[i];
      }
      return 1;
    case node::type::sqrt:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = sqrtf(fabsf(in_a[i]));
      }
      return 1;
    case node::type::tan:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = tanf(in_a[i]);
      }
      return 1;
    case node::type::tanh:
      for (int i = 0; i < batch_size; ++i) {
        out[i - ROW_BATCH_SIZE] = tanhf(in_a[i]);
      }
      return 1;
    // shouldn't reach here!
    default:
      for (int i = 0; i < batch_size; ++i) {
        out[i] = 0.f;
      }
      return 0;
    };
  }
}

} // namespace detail
} // namespace genetic
