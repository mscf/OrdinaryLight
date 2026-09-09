#include <cwchar>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <climits>
template<size_t N> inline int wcscpy_s(wchar_t (&dst)[N], const wchar_t* src) {
    if (!src || std::wcslen(src) >= N) return 1;
    std::wcscpy(dst, src); return 0;
}
#ifndef _WIN32
#define __declspec(x)
#endif
#include <locale>
#define _countof(a) (sizeof(a)/sizeof((a)[0]))
