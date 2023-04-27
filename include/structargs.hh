#pragma once

#if not defined(_STRUCTARGS_HH)
#define _STRUCTARGS_HH

#include <string> // std::string, std::stoul, std::getline
#include <typeinfo>
#include <iostream> // std::cin, std::cout

#define xstr(s) str(s)
#define str(s) #s

#define PP_NARG(...) \
    PP_NARG_(__VA_ARGS__, PP_RSEQ_N())
#define PP_NARG_(...) \
    PP_ARG_N(__VA_ARGS__)
#define PP_ARG_N(                                     \
    _1, _2, _3, _4, _5, _6, _7, _8, _9, _10,          \
    _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, \
    _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, \
    _31, _32, _33, _34, _35, _36, _37, _38, _39, _40, \
    _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, \
    _51, _52, _53, _54, _55, _56, _57, _58, _59, _60, \
    _61, _62, _63, N, ...) N
#define PP_RSEQ_N()                             \
    63, 62, 61, 60,                             \
        59, 58, 57, 56, 55, 54, 53, 52, 51, 50, \
        49, 48, 47, 46, 45, 44, 43, 42, 41, 40, \
        39, 38, 37, 36, 35, 34, 33, 32, 31, 30, \
        29, 28, 27, 26, 25, 24, 23, 22, 21, 20, \
        19, 18, 17, 16, 15, 14, 13, 12, 11, 10, \
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0

#define PARENS ()
#define EXPAND(arg) EXPAND1(EXPAND1(EXPAND1(EXPAND1(arg))))
#define EXPAND1(arg) EXPAND2(EXPAND2(EXPAND2(EXPAND2(arg))))
#define EXPAND2(arg) EXPAND3(EXPAND3(EXPAND3(EXPAND3(arg))))
#define EXPAND3(arg) EXPAND4(EXPAND4(EXPAND4(EXPAND4(arg))))
#define EXPAND4(arg) arg

#define FOR_EACH(macro, ...) __VA_OPT__(EXPAND(FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define FOR_EACH_HELPER(macro, a1, ...) macro(a1) __VA_OPT__(FOR_EACH_AGAIN PARENS(macro, __VA_ARGS__))
#define FOR_EACH_AGAIN() FOR_EACH_HELPER

#define FOR_EACHATTR_PACK(macro, ...) __VA_OPT__(EXPAND(FOR_EACH_HELPER_PACK(macro, __VA_ARGS__)))
#define FOR_EACH_HELPER_PACK(macro, a, a1, a2, ...) macro(a, a1, a2) __VA_OPT__(FOR_EACH_AGAINPACK PARENS(macro, __VA_ARGS__))
#define FOR_EACH_AGAINPACK() FOR_EACH_HELPER_PACK


#define typer(type, name, value) type name = value;
#define TEMPLATESTRUCT(name, type, ...)  \
    template <typename ARG_TYPE = type>  \
    struct name                          \
    {                                    \
        FOR_EACHATTR_PACK(typer, __VA_ARGS__) \
    };



#define CONVERT_OR_DIE(type, val, func...)                       \
    type return_value;                                           \
    try                                                          \
    {                                                            \
        return_value = func;                                     \
    }                                                            \
    catch (...)                                                  \
    {                                                            \
        std::cerr << "Incorect inputs parameters\n"              \
                  << "Error: '" << #val << " " << #func          \
                  << "' [" __FILE__ << ":" << __LINE__ << "] \t" \
                  << std::endl;                                  \
        throw;                                                   \
        exit(1);                                                 \
    }                                                            \
    return return_value;

#define CONVERT(val, type) to_##type(val)
#define TOCONVERT(source_arg, type, val) source_arg = CONVERT(val, type);

#define ARGTOSTRUCT(...) \
    FOR_EACHATTR_PACK(TOCONVERT, __VA_ARGS__)

typedef long double ldouble;
typedef long long llong;

inline ldouble to_ldouble(const char *val)
{
    CONVERT_OR_DIE(ldouble, val, std::stold(val))
}

inline llong to_llong(const char *val)
{
    CONVERT_OR_DIE(long, val, std::stoll(val))
}

inline double to_double(const char *val)
{
    CONVERT_OR_DIE(double, val, std::stod(val))
}

inline float to_float(const char *val)
{
    CONVERT_OR_DIE(float, val, std::stof(val))
}

inline long to_long(const char *val)
{
    CONVERT_OR_DIE(long, val, std::stol(val))
}

inline int to_int(const char *val)
{
    CONVERT_OR_DIE(int, val, std::stoi(val))
}

inline ulong to_ulong(const char *val)
{
    CONVERT_OR_DIE(ulong, val, std::stoul(val))
}

#endif /*_STRUCTARGS_HH*/