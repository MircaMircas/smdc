
#include <stdarg.h>
#include "libultra_internal.h"
#include "printf.h"
#include <string.h>
char *proutSprintf(char *dst, const char *src, size_t count);

void n64_memcpy(void* dst, const void* src, size_t size);

char *proutSprintf(char *dst, const char *src, size_t count) {
    return (char *) n64_memcpy((u8 *) dst, (u8 *) src, count) + count;
}
