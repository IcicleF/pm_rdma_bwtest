#pragma once

#if defined(__GNUC__) && __GNUC__ < 7
#include <experimental/optional>
#define _rlib_optional std::experimental::optional
#define _rlib_none std::experimental::nullopt
#else
#include <optional>
#define _rlib_optional std::optional
#define _rlib_none std::nullopt
#endif

namespace rdmaio {
template <typename T> using Option = _rlib_optional<T>;
inline constexpr auto None = _rlib_none;
}
