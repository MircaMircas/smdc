#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <ultra64.h>

#ifdef __SSE4_1__
#include <immintrin.h>
#define HAS_SSE41 1
#define HAS_NEON 0
#elif __ARM_NEON
#include <arm_neon.h>
#define HAS_SSE41 0
#define HAS_NEON 1
#else
#define HAS_SSE41 0
#define HAS_NEON 0
#endif

//#pragma GCC optimize ("unroll-loops")
#define MEM_BARRIER() asm volatile("" : : : "memory");
#define MEM_BARRIER_PREF(ptr) asm volatile("pref @%0" : : "r"((ptr)) : "memory")

#if HAS_SSE41
#define LOADLH(l, h) _mm_castpd_si128(_mm_loadh_pd(_mm_load_sd((const double *)(l)), (const double *)(h)))
#endif

void n64_memcpy(void* dst, const void* src, size_t size);

#define ROUND_UP_32(v) (((v) + 31) & ~31)
#define ROUND_UP_16(v) (((v) + 15) & ~15)
#define ROUND_UP_8(v) (((v) + 7) & ~7)

#define BUF_U8(a) (rspa.buf.as_u8 + (a))
#define BUF_S16(a) (rspa.buf.as_s16 + (a) / sizeof(int16_t))

static struct  __attribute__((aligned(32)))  {
    union  __attribute__((aligned(32))) {
        int16_t __attribute__((aligned(32))) as_s16[2512 / sizeof(int16_t)];
        uint8_t __attribute__((aligned(32))) as_u8[2512];
    } buf;
    uint16_t in;
    uint16_t out;
    uint16_t nbytes;

    int16_t vol[2];

    uint16_t dry_right;
    uint16_t wet_left;
    uint16_t wet_right;

    int16_t target[2];
    int32_t rate[2];

    int16_t vol_dry;
    int16_t vol_wet;

    ADPCM_STATE *adpcm_loop_state;

    float __attribute__((aligned(32))) adpcm_table[8][2][8];
/*     union {
        int16_t as_s16[2512 / sizeof(int16_t)];
        uint8_t as_u8[2512];
    } buf; */
} rspa;
static float __attribute__((aligned(32))) resample_table[64][4] = {
    {
        (f32) 3129,
        (f32) 26285,
        (f32) 3398,
        (f32) -33,
    },
    {
        (f32) 2873,
        (f32) 26262,
        (f32) 3679,
        (f32) -40,
    },
    {
        (f32) 2628,
        (f32) 26217,
        (f32) 3971,
        (f32) -48,
    },
    {
        (f32) 2394,
        (f32) 26150,
        (f32) 4276,
        (f32) -56,
    },
    {
        (f32) 2173,
        (f32) 26061,
        (f32) 4592,
        (f32) -65,
    },
    {
        (f32) 1963,
        (f32) 25950,
        (f32) 4920,
        (f32) -74,
    },
    {
        (f32) 1764,
        (f32) 25817,
        (f32) 5260,
        (f32) -84,
    },
    {
        (f32) 1576,
        (f32) 25663,
        (f32) 5611,
        (f32) -95,
    },
    {
        (f32) 1399,
        (f32) 25487,
        (f32) 5974,
        (f32) -106,
    },
    {
        (f32) 1233,
        (f32) 25291,
        (f32) 6347,
        (f32) -118,
    },
    {
        (f32) 1077,
        (f32) 25075,
        (f32) 6732,
        (f32) -130,
    },
    {
        (f32) 932,
        (f32) 24838,
        (f32) 7127,
        (f32) -143,
    },
    {
        (f32) 796,
        (f32) 24583,
        (f32) 7532,
        (f32) -156,
    },
    {
        (f32) 671,
        (f32) 24309,
        (f32) 7947,
        (f32) -170,
    },
    {
        (f32) 554,
        (f32) 24016,
        (f32) 8371,
        (f32) -184,
    },
    {
        (f32) 446,
        (f32) 23706,
        (f32) 8804,
        (f32) -198,
    },
    {
        (f32) 347,
        (f32) 23379,
        (f32) 9246,
        (f32) -212,
    },
    {
        (f32) 257,
        (f32) 23036,
        (f32) 9696,
        (f32) -226,
    },
    {
        (f32) 174,
        (f32) 22678,
        (f32) 10153,
        (f32) -240,
    },
    {
        (f32) 99,
        (f32) 22304,
        (f32) 10618,
        (f32) -254,
    },
    {
        (f32) 31,
        (f32) 21917,
        (f32) 11088,
        (f32) -268,
    },
    {
        (f32) -30,
        (f32) 21517,
        (f32) 11564,
        (f32) -280,
    },
    {
        (f32) -84,
        (f32) 21104,
        (f32) 12045,
        (f32) -293,
    },
    {
        (f32) -132,
        (f32) 20679,
        (f32) 12531,
        (f32) -304,
    },
    {
        (f32) -173,
        (f32) 20244,
        (f32) 13020,
        (f32) -314,
    },
    {
        (f32) -210,
        (f32) 19799,
        (f32) 13512,
        (f32) -323,
    },
    {
        (f32) -241,
        (f32) 19345,
        (f32) 14006,
        (f32) -330,
    },
    {
        (f32) -267,
        (f32) 18882,
        (f32) 14501,
        (f32) -336,
    },
    {
        (f32) -289,
        (f32) 18413,
        (f32) 14997,
        (f32) -340,
    },
    {
        (f32) -306,
        (f32) 17937,
        (f32) 15493,
        (f32) -341,
    },
    {
        (f32) -320,
        (f32) 17456,
        (f32) 15988,
        (f32) -340,
    },
    {
        (f32) -330,
        (f32) 16970,
        (f32) 16480,
        (f32) -337,
    },
    {
        (f32) -337,
        (f32) 16480,
        (f32) 16970,
        (f32) -330,
    },
    {
        (f32) -340,
        (f32) 15988,
        (f32) 17456,
        (f32) -320,
    },
    {
        (f32) -341,
        (f32) 15493,
        (f32) 17937,
        (f32) -306,
    },
    {
        (f32) -340,
        (f32) 14997,
        (f32) 18413,
        (f32) -289,
    },
    {
        (f32) -336,
        (f32) 14501,
        (f32) 18882,
        (f32) -267,
    },
    {
        (f32) -330,
        (f32) 14006,
        (f32) 19345,
        (f32) -241,
    },
    {
        (f32) -323,
        (f32) 13512,
        (f32) 19799,
        (f32) -210,
    },
    {
        (f32) -314,
        (f32) 13020,
        (f32) 20244,
        (f32) -173,
    },
    {
        (f32) -304,
        (f32) 12531,
        (f32) 20679,
        (f32) -132,
    },
    {
        (f32) -293,
        (f32) 12045,
        (f32) 21104,
        (f32) -84,
    },
    {
        (f32) -280,
        (f32) 11564,
        (f32) 21517,
        (f32) -30,
    },
    {
        (f32) -268,
        (f32) 11088,
        (f32) 21917,
        (f32) 31,
    },
    {
        (f32) -254,
        (f32) 10618,
        (f32) 22304,
        (f32) 99,
    },
    {
        (f32) -240,
        (f32) 10153,
        (f32) 22678,
        (f32) 174,
    },
    {
        (f32) -226,
        (f32) 9696,
        (f32) 23036,
        (f32) 257,
    },
    {
        (f32) -212,
        (f32) 9246,
        (f32) 23379,
        (f32) 347,
    },
    {
        (f32) -198,
        (f32) 8804,
        (f32) 23706,
        (f32) 446,
    },
    {
        (f32) -184,
        (f32) 8371,
        (f32) 24016,
        (f32) 554,
    },
    {
        (f32) -170,
        (f32) 7947,
        (f32) 24309,
        (f32) 671,
    },
    {
        (f32) -156,
        (f32) 7532,
        (f32) 24583,
        (f32) 796,
    },
    {
        (f32) -143,
        (f32) 7127,
        (f32) 24838,
        (f32) 932,
    },
    {
        (f32) -130,
        (f32) 6732,
        (f32) 25075,
        (f32) 1077,
    },
    {
        (f32) -118,
        (f32) 6347,
        (f32) 25291,
        (f32) 1233,
    },
    {
        (f32) -106,
        (f32) 5974,
        (f32) 25487,
        (f32) 1399,
    },
    {
        (f32) -95,
        (f32) 5611,
        (f32) 25663,
        (f32) 1576,
    },
    {
        (f32) -84,
        (f32) 5260,
        (f32) 25817,
        (f32) 1764,
    },
    {
        (f32) -74,
        (f32) 4920,
        (f32) 25950,
        (f32) 1963,
    },
    {
        (f32) -65,
        (f32) 4592,
        (f32) 26061,
        (f32) 2173,
    },
    {
        (f32) -56,
        (f32) 4276,
        (f32) 26150,
        (f32) 2394,
    },
    {
        (f32) -48,
        (f32) 3971,
        (f32) 26217,
        (f32) 2628,
    },
    {
        (f32) -40,
        (f32) 3679,
        (f32) 26262,
        (f32) 2873,
    },
    {
        (f32) -33,
        (f32) 3398,
        (f32) 26285,
        (f32) 3129,
    },
};

static inline int16_t clamp16(int32_t v) {
    if (v < -0x8000) {
        return -0x8000;
    } else if (v > 0x7fff) {
        return 0x7fff;
    }
    return (int16_t)v;
}

static inline int32_t clamp32(int64_t v) {
    if (v < -0x7fffffff - 1) {
        return -0x7fffffff - 1;
    } else if (v > 0x7fffffff) {
        return 0x7fffffff;
    }
    return (int32_t)v;
}

//void memcpy4(void *dest, const void *src, size_t count);
#if defined(TARGET_PSP)
void memcpy_vfpu( void* dst, const void* src, size_t size )
{
    //less than 16bytes or there is no 32bit alignment -> not worth optimizing
	if( ((u32)src&0x3) != ((u32)dst&0x3) && (size<16) )
    {
        memcpy( dst, src, size );
        return;
    }

    u8* src8 = (u8*)src;
    u8* dst8 = (u8*)dst;

	// Align dst to 4 bytes or just resume if already done
	while( ((u32)dst8&0x3)!=0 )
	{
		*dst8++ = *src8++;
		size--;
	}

	u32 *dst32=(u32*)dst8;
	u32 *src32=(u32*)src8;

	// Align dst to 16 bytes or just resume if already done
	while( ((u32)dst32&0xF)!=0 )
	{
		*dst32++ = *src32++;
		size -= 4;
	}

	dst8=(u8*)dst32;
	src8=(u8*)src32;

	if( ((u32)src8&0xF)==0 )	//Both src and dst are 16byte aligned
	{
		while (size>63)
		{
			asm(".set	push\n"					// save assembler option
				".set	noreorder\n"			// suppress reordering
				"lv.q c000, 0(%1)\n"
				"lv.q c010, 16(%1)\n"
				"lv.q c020, 32(%1)\n"
				"lv.q c030, 48(%1)\n"
				"sv.q c000, 0(%0)\n"
				"sv.q c010, 16(%0)\n"
				"sv.q c020, 32(%0)\n"
				"sv.q c030, 48(%0)\n"
				"addiu  %2, %2, -64\n"			//size -= 64;
				"addiu	%1, %1, 64\n"			//dst8 += 64;
				"addiu	%0, %0, 64\n"			//src8 += 64;
				".set	pop\n"					// restore assembler option
				:"+r"(dst8),"+r"(src8),"+r"(size)
				:
				:"memory"
				);
		}

		while (size>15)
		{
			asm(".set	push\n"					// save assembler option
				".set	noreorder\n"			// suppress reordering
				"lv.q c000, 0(%1)\n"
				"sv.q c000, 0(%0)\n"
				"addiu  %2, %2, -16\n"			//size -= 16;
				"addiu	%1, %1, 16\n"			//dst8 += 16;
				"addiu	%0, %0, 16\n"			//src8 += 16;
				".set	pop\n"					// restore assembler option
				:"+r"(dst8),"+r"(src8),"+r"(size)
				:
				:"memory"
				);
		}
	}
	else 	//At least src is 4byte and dst is 16byte aligned
    {
		while (size>63)
		{
			asm(".set	push\n"					// save assembler option
				".set	noreorder\n"			// suppress reordering
				"ulv.q c000, 0(%1)\n"
				"ulv.q c010, 16(%1)\n"
				"ulv.q c020, 32(%1)\n"
				"ulv.q c030, 48(%1)\n"
				"sv.q c000, 0(%0)\n"
				"sv.q c010, 16(%0)\n"
				"sv.q c020, 32(%0)\n"
				"sv.q c030, 48(%0)\n"
				"addiu  %2, %2, -64\n"			//size -= 64;
				"addiu	%1, %1, 64\n"			//dst8 += 64;
				"addiu	%0, %0, 64\n"			//src8 += 64;
				".set	pop\n"					// restore assembler option
				:"+r"(dst8),"+r"(src8),"+r"(size)
				:
				:"memory"
				);
		}

		while (size>15)
		{
			asm(".set	push\n"					// save assembler option
				".set	noreorder\n"			// suppress reordering
				"ulv.q c000, 0(%1)\n"
				"sv.q c000, 0(%0)\n"
				"addiu  %2, %2, -16\n"			//size -= 16;
				"addiu	%1, %1, 16\n"			//dst8 += 16;
				"addiu	%0, %0, 16\n"			//src8 += 16;
				".set	pop\n"					// restore assembler option
				:"+r"(dst8),"+r"(src8),"+r"(size)
				:
				:"memory"
				);
		}
    }

	// Most copies are completed with the VFPU, so fast out
	if (size == 0)
		return;

	dst32=(u32*)dst8;
	src32=(u32*)src8;

	//Copy remaning 32bit...
	while( size>3 )
	{
		*dst32++ = *src32++;
		size -= 4;
	}

	dst8=(u8*)dst32;
	src8=(u8*)src32;

	//Copy remaning bytes if any...
	while( size>0 )
    {
        *dst8++ = *src8++;
        size--;
    }
}

void __attribute__((noinline))  memcpy_vfpu_simple(void *dst, void *src, size_t size) 
{
    __asm__ volatile (
    "loop:\n"
        "beqz %2, loop_end\n"
        "lv.q C200, 0(%1)\n"
        "sv.q C200, 0(%0)\n"
        "addiu %0, %0, 16\n"
        "addiu %1, %1, 16\n"
        "addiu %2, %2, -16\n"
        "b loop\n"
    "loop_end:\n" ::
    "r"(dst), "r"( src), "r"(size));
}
#else
void memcpy_vfpu( void* dst, const void* src, unsigned int size )
{
	memcpy( dst, src, size );
}
#endif
/* 
void memcpy4(void *dest, const void *src, size_t count)
{
	unsigned long *tmp = (unsigned long *) dest;
	unsigned long *s = (unsigned long *) src;
	count = count/4;

	while (count--)
		*tmp++ = *s++;
} */

void aClearBufferImpl(uint16_t addr, int nbytes) {
    nbytes = ROUND_UP_16(nbytes);
    memset(rspa.buf.as_u8 + addr, 0, nbytes);
}

void aLoadBufferImpl(const void *source_addr) {
    n64_memcpy(rspa.buf.as_u8 + rspa.in, source_addr, ROUND_UP_8(rspa.nbytes));
}

void aSaveBufferImpl(int16_t *dest_addr) {
    n64_memcpy(dest_addr, rspa.buf.as_s16 + rspa.out / sizeof(int16_t), ROUND_UP_8(rspa.nbytes));
}

#define recip8192 0.00012207f
#define recip2048 0.00048828f
#define recip2560 0.00039062f

void aLoadADPCMImpl(int num_entries_times_16, const int16_t* book_source_addr) {
    float* aptr = (float*) rspa.adpcm_table;
    short tmp[8];

    __builtin_prefetch(book_source_addr);

    for (int i = 0; i < num_entries_times_16 / 2; i += 8) {
        __builtin_prefetch(&aptr[i]);

        tmp[0] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 0]);
        tmp[1] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 1]);
        tmp[2] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 2]);
        tmp[3] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 3]);
        tmp[4] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 4]);
        tmp[5] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 5]);
        tmp[6] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 6]);
        tmp[7] = (short) /* __builtin_bswap16 */((uint16_t) book_source_addr[i + 7]);

        MEM_BARRIER_PREF(&book_source_addr[i + 8]);

        aptr[i + 0] = recip2048 * (f32) (s32) tmp[0];
        aptr[i + 1] = recip2048 * (f32) (s32) tmp[1];
        aptr[i + 2] = recip2048 * (f32) (s32) tmp[2];
        aptr[i + 3] = recip2048 * (f32) (s32) tmp[3];
        aptr[i + 4] = recip2048 * (f32) (s32) tmp[4];
        aptr[i + 5] = recip2048 * (f32) (s32) tmp[5];
        aptr[i + 6] = recip2048 * (f32) (s32) tmp[6];
        aptr[i + 7] = recip2048 * (f32) (s32) tmp[7];
    }
}

void aSetBufferImpl(uint8_t flags, uint16_t in, uint16_t out, uint16_t nbytes) {
    if (flags & A_AUX) {
        rspa.dry_right = in;
        rspa.wet_left = out;
        rspa.wet_right = nbytes;
    } else {
        rspa.in = in;
        rspa.out = out;
        rspa.nbytes = nbytes;
    }
}

void aSetVolumeImpl(uint8_t flags, int16_t v, int16_t t, int16_t r) {
    if (flags & A_AUX) {
        rspa.vol_dry = v;
        rspa.vol_wet = r;
    } else if (flags & A_VOL) {
        if (flags & A_LEFT) {
            rspa.vol[0] = v;
        } else {
            rspa.vol[1] = v;
        }
    } else {
        if (flags & A_LEFT) {
            rspa.target[0] = v;
            rspa.rate[0] = (int32_t)((uint16_t)t << 16 | ((uint16_t)r));
        } else {
            rspa.target[1] = v;
            rspa.rate[1] = (int32_t)((uint16_t)t << 16 | ((uint16_t)r));
        }
    }
}
#if 0
void aInterleaveImpl(uint16_t left, uint16_t right) {
    int count = ROUND_UP_16(rspa.nbytes) / sizeof(int16_t) / 8;
    int16_t *l = rspa.buf.as_s16 + left / sizeof(int16_t);
    int16_t *r = rspa.buf.as_s16 + right / sizeof(int16_t);
    int16_t *d = rspa.buf.as_s16 + rspa.out / sizeof(int16_t);
    while (count > 0) {
        int16_t l0 = *l++;
        int16_t l1 = *l++;
        int16_t l2 = *l++;
        int16_t l3 = *l++;
        int16_t l4 = *l++;
        int16_t l5 = *l++;
        int16_t l6 = *l++;
        int16_t l7 = *l++;
        int16_t r0 = *r++;
        int16_t r1 = *r++;
        int16_t r2 = *r++;
        int16_t r3 = *r++;
        int16_t r4 = *r++;
        int16_t r5 = *r++;
        int16_t r6 = *r++;
        int16_t r7 = *r++;
        *d++ = l0;
        *d++ = r0;
        *d++ = l1;
        *d++ = r1;
        *d++ = l2;
        *d++ = r2;
        *d++ = l3;
        *d++ = r3;
        *d++ = l4;
        *d++ = r4;
        *d++ = l5;
        *d++ = r5;
        *d++ = l6;
        *d++ = r6;
        *d++ = l7;
        *d++ = r7;
        --count;
    }
}
#endif
void aInterleaveImpl(uint16_t left, uint16_t right) {
    int count = ROUND_UP_16(rspa.nbytes) / sizeof(int16_t) / 4;
    int16_t* l = BUF_S16(left);
    int16_t* r = BUF_S16(right);

    //int16_t* d = BUF_S16(rspa.out);
    int32_t* d = (int32_t*)(((uintptr_t)BUF_S16(rspa.out)+3) & ~3);

    __builtin_prefetch(r);

    while (count > 0) {
        __builtin_prefetch(r+16);
        int32_t lr0 = (*r++ << 16) | (*l++ & 0xffff);
        int32_t lr1 = (*r++ << 16) | (*l++ & 0xffff);
        int32_t lr2 = (*r++ << 16) | (*l++ & 0xffff);
        int32_t lr3 = (*r++ << 16) | (*l++ & 0xffff);
#if 1
            asm volatile("": : : "memory");
#endif
        *d++ = lr0;
        *d++ = lr1;
        *d++ = lr2;
        *d++ = lr3;

        --count;
    }
}
void aDMEMMoveImpl(uint16_t in_addr, uint16_t out_addr, int nbytes) {
    nbytes = ROUND_UP_16(nbytes);
    memmove(rspa.buf.as_u8 + out_addr, rspa.buf.as_u8 + in_addr, nbytes);
}

void aSetLoopImpl(ADPCM_STATE *adpcm_loop_state) {
    rspa.adpcm_loop_state = adpcm_loop_state;
}
#include <kos.h>


#include "sh4zam.h"

inline static void shz_xmtrx_load_3x4_rows(const shz_vec4_t* r1, const shz_vec4_t* r2, const shz_vec4_t* r3) {
    asm volatile(R"(
        pref    @%0
        frchg

        fldi0   fr12
        fldi0   fr13
        fldi0   fr14
        fldi1   fr15

        pref    @%1
        fmov.s  @%0+, fr0
        fmov.s  @%0+, fr1
        fmov.s  @%0+, fr2
        fmov.s  @%0,  fr3

        pref    @%2
        fmov.s  @%1+, fr4
        fmov.s  @%1+, fr5
        fmov.s  @%1+, fr6
        fmov.s  @%1,  fr7

        fmov.s  @%2+, fr8
        fmov.s  @%2+, fr9
        fmov.s  @%2+, fr10
        fmov.s  @%2,  fr11

        frchg
    )"
                 : "+&r"(r1), "+&r"(r2), "+&r"(r3));
}

SHZ_FORCE_INLINE void shz_copy_16_shorts(void* restrict dst, const void* restrict src) {
    asm volatile(R"(
        mov.w   @%[s]+, r0
        mov.w   @%[s]+, r1
        mov.w   @%[s]+, r2
        mov.w   @%[s]+, r3
        mov.w   @%[s]+, r4
        mov.w   @%[s]+, r5
        mov.w   @%[s]+, r6
        mov.w   @%[s]+, r7
        add     #16, %[d]
        mov.w   r7, @-%[d]
        mov.w   r6, @-%[d]
        mov.w   r5, @-%[d]
        mov.w   r4, @-%[d]
        mov.w   r3, @-%[d]
        mov.w   r2, @-%[d]
        mov.w   r1, @-%[d]
        mov.w   r0, @-%[d]
        mov.w   @%[s]+, r0
        mov.w   @%[s]+, r1
        mov.w   @%[s]+, r2
        mov.w   @%[s]+, r3
        mov.w   @%[s]+, r4
        mov.w   @%[s]+, r5
        mov.w   @%[s]+, r6
        mov.w   @%[s]+, r7
        add     #32, %[d]
        mov.w   r7, @-%[d]
        mov.w   r6, @-%[d]
        mov.w   r5, @-%[d]
        mov.w   r4, @-%[d]
        mov.w   r3, @-%[d]
        mov.w   r2, @-%[d]
        mov.w   r1, @-%[d]
        mov.w   r0, @-%[d]
    )"
                 : [d] "+r"(dst), [s] "+r"(src)
                 :
                 : "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "memory");
}

SHZ_FORCE_INLINE void shz_zero_16_shorts(void* dst) {
    asm volatile(R"(
        xor     r0, r0
        add     #32 %0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
        mov.w   r0, @-%0
    )"
                 :
                 : "r"(dst)
                 : "r0", "memory");
}

static inline s16 clamp16f(float v) {
    // v *= recip2048;
    s32 sv = (s32)v;
    if (sv < -32768) {
        return -32768;
    } else if (sv > 32767) {
        return 32767;
    }
    return (s16)sv;
}

static inline float shift_to_float_multiplier(uint8_t shift) {
    const static float
        __attribute__((aligned(32))) shift_to_float[16] = { 1.0f,    2.0f,    4.0f,     8.0f,    16.0f,   32.0f,
                                                            64.0f,   128.0f,  256.0f,   512.0f,  1024.0f, 2048.0f,
                                                            4096.0f, 8192.0f, 16364.0f, 32768.0f };
    return shift_to_float[shift];
}

static const float __attribute__((aligned(32))) nyblls_as_floats[256][2] = {
    { 0.0f, 0.0f },   { 0.0f, 1.0f },   { 0.0f, 2.0f },   { 0.0f, 3.0f },   { 0.0f, 4.0f },   { 0.0f, 5.0f },
    { 0.0f, 6.0f },   { 0.0f, 7.0f },   { 0.0f, -8.0f },  { 0.0f, -7.0f },  { 0.0f, -6.0f },  { 0.0f, -5.0f },
    { 0.0f, -4.0f },  { 0.0f, -3.0f },  { 0.0f, -2.0f },  { 0.0f, -1.0f },  { 1.0f, 0.0f },   { 1.0f, 1.0f },
    { 1.0f, 2.0f },   { 1.0f, 3.0f },   { 1.0f, 4.0f },   { 1.0f, 5.0f },   { 1.0f, 6.0f },   { 1.0f, 7.0f },
    { 1.0f, -8.0f },  { 1.0f, -7.0f },  { 1.0f, -6.0f },  { 1.0f, -5.0f },  { 1.0f, -4.0f },  { 1.0f, -3.0f },
    { 1.0f, -2.0f },  { 1.0f, -1.0f },  { 2.0f, 0.0f },   { 2.0f, 1.0f },   { 2.0f, 2.0f },   { 2.0f, 3.0f },
    { 2.0f, 4.0f },   { 2.0f, 5.0f },   { 2.0f, 6.0f },   { 2.0f, 7.0f },   { 2.0f, -8.0f },  { 2.0f, -7.0f },
    { 2.0f, -6.0f },  { 2.0f, -5.0f },  { 2.0f, -4.0f },  { 2.0f, -3.0f },  { 2.0f, -2.0f },  { 2.0f, -1.0f },
    { 3.0f, 0.0f },   { 3.0f, 1.0f },   { 3.0f, 2.0f },   { 3.0f, 3.0f },   { 3.0f, 4.0f },   { 3.0f, 5.0f },
    { 3.0f, 6.0f },   { 3.0f, 7.0f },   { 3.0f, -8.0f },  { 3.0f, -7.0f },  { 3.0f, -6.0f },  { 3.0f, -5.0f },
    { 3.0f, -4.0f },  { 3.0f, -3.0f },  { 3.0f, -2.0f },  { 3.0f, -1.0f },  { 4.0f, 0.0f },   { 4.0f, 1.0f },
    { 4.0f, 2.0f },   { 4.0f, 3.0f },   { 4.0f, 4.0f },   { 4.0f, 5.0f },   { 4.0f, 6.0f },   { 4.0f, 7.0f },
    { 4.0f, -8.0f },  { 4.0f, -7.0f },  { 4.0f, -6.0f },  { 4.0f, -5.0f },  { 4.0f, -4.0f },  { 4.0f, -3.0f },
    { 4.0f, -2.0f },  { 4.0f, -1.0f },  { 5.0f, 0.0f },   { 5.0f, 1.0f },   { 5.0f, 2.0f },   { 5.0f, 3.0f },
    { 5.0f, 4.0f },   { 5.0f, 5.0f },   { 5.0f, 6.0f },   { 5.0f, 7.0f },   { 5.0f, -8.0f },  { 5.0f, -7.0f },
    { 5.0f, -6.0f },  { 5.0f, -5.0f },  { 5.0f, -4.0f },  { 5.0f, -3.0f },  { 5.0f, -2.0f },  { 5.0f, -1.0f },
    { 6.0f, 0.0f },   { 6.0f, 1.0f },   { 6.0f, 2.0f },   { 6.0f, 3.0f },   { 6.0f, 4.0f },   { 6.0f, 5.0f },
    { 6.0f, 6.0f },   { 6.0f, 7.0f },   { 6.0f, -8.0f },  { 6.0f, -7.0f },  { 6.0f, -6.0f },  { 6.0f, -5.0f },
    { 6.0f, -4.0f },  { 6.0f, -3.0f },  { 6.0f, -2.0f },  { 6.0f, -1.0f },  { 7.0f, 0.0f },   { 7.0f, 1.0f },
    { 7.0f, 2.0f },   { 7.0f, 3.0f },   { 7.0f, 4.0f },   { 7.0f, 5.0f },   { 7.0f, 6.0f },   { 7.0f, 7.0f },
    { 7.0f, -8.0f },  { 7.0f, -7.0f },  { 7.0f, -6.0f },  { 7.0f, -5.0f },  { 7.0f, -4.0f },  { 7.0f, -3.0f },
    { 7.0f, -2.0f },  { 7.0f, -1.0f },  { -8.0f, 0.0f },  { -8.0f, 1.0f },  { -8.0f, 2.0f },  { -8.0f, 3.0f },
    { -8.0f, 4.0f },  { -8.0f, 5.0f },  { -8.0f, 6.0f },  { -8.0f, 7.0f },  { -8.0f, -8.0f }, { -8.0f, -7.0f },
    { -8.0f, -6.0f }, { -8.0f, -5.0f }, { -8.0f, -4.0f }, { -8.0f, -3.0f }, { -8.0f, -2.0f }, { -8.0f, -1.0f },
    { -7.0f, 0.0f },  { -7.0f, 1.0f },  { -7.0f, 2.0f },  { -7.0f, 3.0f },  { -7.0f, 4.0f },  { -7.0f, 5.0f },
    { -7.0f, 6.0f },  { -7.0f, 7.0f },  { -7.0f, -8.0f }, { -7.0f, -7.0f }, { -7.0f, -6.0f }, { -7.0f, -5.0f },
    { -7.0f, -4.0f }, { -7.0f, -3.0f }, { -7.0f, -2.0f }, { -7.0f, -1.0f }, { -6.0f, 0.0f },  { -6.0f, 1.0f },
    { -6.0f, 2.0f },  { -6.0f, 3.0f },  { -6.0f, 4.0f },  { -6.0f, 5.0f },  { -6.0f, 6.0f },  { -6.0f, 7.0f },
    { -6.0f, -8.0f }, { -6.0f, -7.0f }, { -6.0f, -6.0f }, { -6.0f, -5.0f }, { -6.0f, -4.0f }, { -6.0f, -3.0f },
    { -6.0f, -2.0f }, { -6.0f, -1.0f }, { -5.0f, 0.0f },  { -5.0f, 1.0f },  { -5.0f, 2.0f },  { -5.0f, 3.0f },
    { -5.0f, 4.0f },  { -5.0f, 5.0f },  { -5.0f, 6.0f },  { -5.0f, 7.0f },  { -5.0f, -8.0f }, { -5.0f, -7.0f },
    { -5.0f, -6.0f }, { -5.0f, -5.0f }, { -5.0f, -4.0f }, { -5.0f, -3.0f }, { -5.0f, -2.0f }, { -5.0f, -1.0f },
    { -4.0f, 0.0f },  { -4.0f, 1.0f },  { -4.0f, 2.0f },  { -4.0f, 3.0f },  { -4.0f, 4.0f },  { -4.0f, 5.0f },
    { -4.0f, 6.0f },  { -4.0f, 7.0f },  { -4.0f, -8.0f }, { -4.0f, -7.0f }, { -4.0f, -6.0f }, { -4.0f, -5.0f },
    { -4.0f, -4.0f }, { -4.0f, -3.0f }, { -4.0f, -2.0f }, { -4.0f, -1.0f }, { -3.0f, 0.0f },  { -3.0f, 1.0f },
    { -3.0f, 2.0f },  { -3.0f, 3.0f },  { -3.0f, 4.0f },  { -3.0f, 5.0f },  { -3.0f, 6.0f },  { -3.0f, 7.0f },
    { -3.0f, -8.0f }, { -3.0f, -7.0f }, { -3.0f, -6.0f }, { -3.0f, -5.0f }, { -3.0f, -4.0f }, { -3.0f, -3.0f },
    { -3.0f, -2.0f }, { -3.0f, -1.0f }, { -2.0f, 0.0f },  { -2.0f, 1.0f },  { -2.0f, 2.0f },  { -2.0f, 3.0f },
    { -2.0f, 4.0f },  { -2.0f, 5.0f },  { -2.0f, 6.0f },  { -2.0f, 7.0f },  { -2.0f, -8.0f }, { -2.0f, -7.0f },
    { -2.0f, -6.0f }, { -2.0f, -5.0f }, { -2.0f, -4.0f }, { -2.0f, -3.0f }, { -2.0f, -2.0f }, { -2.0f, -1.0f },
    { -1.0f, 0.0f },  { -1.0f, 1.0f },  { -1.0f, 2.0f },  { -1.0f, 3.0f },  { -1.0f, 4.0f },  { -1.0f, 5.0f },
    { -1.0f, 6.0f },  { -1.0f, 7.0f },  { -1.0f, -8.0f }, { -1.0f, -7.0f }, { -1.0f, -6.0f }, { -1.0f, -5.0f },
    { -1.0f, -4.0f }, { -1.0f, -3.0f }, { -1.0f, -2.0f }, { -1.0f, -1.0f }
};

static inline void extend_nyblls_to_floats(uint8_t nybll, float* fp1, float* fp2) {
    const float* fpair = nyblls_as_floats[nybll];
    *fp1 = fpair[0];
    *fp2 = fpair[1];
}

void aADPCMdecImpl(uint8_t flags, ADPCM_STATE state) {
    int16_t* out = BUF_S16(rspa.out);
    MEM_BARRIER_PREF(out);
    uint8_t* in = BUF_U8(rspa.in);
    int nbytes = ROUND_UP_32(rspa.nbytes);
    if (flags & A_INIT) {
        shz_zero_16_shorts(out);
    } else if (flags & A_LOOP) {
        shz_copy_16_shorts(out, rspa.adpcm_loop_state);
        for (int i=0;i<16;i++) {
            out[i] = __builtin_bswap16(out[i]);
        }
    } else {
        shz_copy_16_shorts(out, state);
    }
    MEM_BARRIER_PREF(in);
    out += 16;
    float prev1 = out[-1];
    float prev2 = out[-2];

    while (nbytes > 0) {
        const uint8_t si_in = *in++;
        const uint8_t next = *in++;
        MEM_BARRIER_PREF(nyblls_as_floats[next]);
        const uint8_t in_array[2][4] = {
            { next, *in++, *in++, *in++ },
            { *in++, *in++, *in++, *in++ }
        };
        const unsigned table_index = si_in & 0xf; // should be in 0..7
        const float(*tbl)[8] = rspa.adpcm_table[table_index];
        const float shift = shift_to_float_multiplier(si_in >> 4); // should be in 0..12 or 0..14
        float instr[2][8];

        for(int i = 0; i < 2; ++i) {
            {
                MEM_BARRIER_PREF(nyblls_as_floats[in_array[i][1]]);
                extend_nyblls_to_floats(in_array[i][0], &instr[i][0], &instr[i][1]);
                instr[i][0] *= shift;
                instr[i][1] *= shift;
                MEM_BARRIER_PREF(nyblls_as_floats[in_array[i][2]]);
                extend_nyblls_to_floats(in_array[i][1], &instr[i][2], &instr[i][3]);
                instr[i][2] *= shift;
                instr[i][3] *= shift;
            }
            {
                MEM_BARRIER_PREF(nyblls_as_floats[in_array[i][3]]);
                extend_nyblls_to_floats(in_array[i][2], &instr[i][4], &instr[i][5]);
                instr[i][4] *= shift;
                instr[i][5] *= shift;
                MEM_BARRIER_PREF(&tbl[i][0]);
                extend_nyblls_to_floats(in_array[i][3], &instr[i][6], &instr[i][7]);
                instr[i][6] *= shift;
                instr[i][7] *= shift;
            }
        }
        MEM_BARRIER_PREF(in);

        for (size_t i = 0; i < 2; i++) {
            const float *ins = instr[i];
            shz_vec4_t acc_vec[2];
            float *accf = (float *)acc_vec;
            const shz_vec4_t in_vec = { .x = prev2, .y = prev1, .z = 1.0f };

            shz_xmtrx_load_3x4_rows((const shz_vec4_t*)&tbl[0][0], (const shz_vec4_t*)&tbl[1][0], (const shz_vec4_t*)&ins[0]);
            acc_vec[0] = shz_xmtrx_trans_vec4(in_vec);
            shz_xmtrx_load_3x4_rows((const shz_vec4_t*)&tbl[0][4], (const shz_vec4_t*)&tbl[1][4], (const shz_vec4_t*)&ins[4]);
            acc_vec[1] = shz_xmtrx_trans_vec4(in_vec);

            {
                register float fone asm("fr8")  = 1.0f;
                register float ins0 asm("fr9")  = ins[0];
                register float ins1 asm("fr10") = ins[1];
                register float ins2 asm("fr11") = ins[2];
                accf[2] = shz_dot8f(fone, ins0, ins1, ins2, accf[2], tbl[1][1], tbl[1][0], 0.0f);
                accf[7] = shz_dot8f(fone, ins0, ins1, ins2, accf[7], tbl[1][6], tbl[1][5], tbl[1][4]);
                accf[1] += (tbl[1][0] * ins0);
                shz_xmtrx_load_4x4_cols((const shz_vec4_t*)&accf[3], (const shz_vec4_t*)&tbl[1][2], (const shz_vec4_t*)&tbl[1][1], (const shz_vec4_t*)&tbl[1][0]);
                *(SHZ_ALIASING shz_vec4_t*)&accf[3] =
                    shz_xmtrx_trans_vec4((shz_vec4_t) { .x = fone, .y = ins0, .z = ins1, .w = ins2 });
            }
            {
                register float ins3 asm("fr8")  = ins[3];
                register float ins4 asm("fr9")  = ins[4];
                register float ins5 asm("fr10") = ins[5];
                register float ins6 asm("fr11") = ins[6];
                accf[7] += shz_dot8f(ins3, ins4, ins5, ins6, tbl[1][3], tbl[1][2], tbl[1][1], tbl[1][0]);
                accf[6] += shz_dot8f(ins3, ins4, ins5, ins6, tbl[1][2], tbl[1][1], tbl[1][0], 0.0f);
                accf[5] += (tbl[1][1] * ins3) + (tbl[1][0] * ins4);
                accf[4] += (tbl[1][0] * ins3);
            }

            for (size_t j = 0; j < 6; ++j)
                *out++ = clamp16f(accf[j]);

            prev2  = clamp16f(accf[6]);
            *out++ = prev2;
            prev1  = clamp16f(accf[7]);
            *out++ = prev1;
        }
        MEM_BARRIER_PREF(out);
        nbytes -= 16 * sizeof(int16_t);
    }

    shz_copy_16_shorts(state, (out - 16));
}

#if 0
/*@Note: Decent Slowdown */
void aResampleImpl(uint8_t flags, uint16_t pitch, RESAMPLE_STATE state) {
    int16_t tmp[16];
    int16_t *in_initial = rspa.buf.as_s16 + rspa.in / sizeof(int16_t);
    int16_t *in = in_initial;
    int16_t *out = rspa.buf.as_s16 + rspa.out / sizeof(int16_t);
    int nbytes = ROUND_UP_16(rspa.nbytes);
    uint32_t pitch_accumulator;
    int i;
#if !HAS_SSE41 && !HAS_NEON
    int16_t *tbl;
    int32_t sample;
#endif
    if (flags & A_INIT) {
        memset(tmp, 0, 5 * sizeof(int16_t));
    } else {
        memcpy(tmp, state, 16 * sizeof(int16_t));
    }
    if (flags & 2) {
        memcpy(in - 8, tmp + 8, 8 * sizeof(int16_t));
        in -= tmp[5] / sizeof(int16_t);
    }
    in -= 4;
    pitch_accumulator = (uint16_t)tmp[4];
    memcpy(in, tmp, 4 * sizeof(int16_t));

#if HAS_SSE41
    __m128i multiples = _mm_setr_epi16(0, 2, 4, 6, 8, 10, 12, 14);
    __m128i pitchvec = _mm_set1_epi16((int16_t)pitch);
    __m128i pitchvec_8_steps = _mm_set1_epi32((pitch << 1) * 8);
    __m128i pitchacclo_vec = _mm_set1_epi32((uint16_t)pitch_accumulator);
    __m128i pl = _mm_mullo_epi16(multiples, pitchvec);
    __m128i ph = _mm_mulhi_epu16(multiples, pitchvec);
    __m128i acc_a = _mm_add_epi32(_mm_unpacklo_epi16(pl, ph), pitchacclo_vec);
    __m128i acc_b = _mm_add_epi32(_mm_unpackhi_epi16(pl, ph), pitchacclo_vec);

    do {
        __m128i tbl_positions = _mm_srli_epi16(_mm_packus_epi32(
            _mm_and_si128(acc_a, _mm_set1_epi32(0xffff)),
            _mm_and_si128(acc_b, _mm_set1_epi32(0xffff))), 10);

        __m128i in_positions = _mm_packus_epi32(_mm_srli_epi32(acc_a, 16), _mm_srli_epi32(acc_b, 16));
        __m128i tbl_entries[4];
        __m128i samples[4];

        /*for (i = 0; i < 4; i++) {
            tbl_entries[i] = _mm_castpd_si128(_mm_loadh_pd(_mm_load_sd(
                (const double *)resample_table[_mm_extract_epi16(tbl_positions, 2 * i)]),
                (const double *)resample_table[_mm_extract_epi16(tbl_positions, 2 * i + 1)]));

            samples[i] = _mm_castpd_si128(_mm_loadh_pd(_mm_load_sd(
                (const double *)&in[_mm_extract_epi16(in_positions, 2 * i)]),
                (const double *)&in[_mm_extract_epi16(in_positions, 2 * i + 1)]));

            samples[i] = _mm_mulhrs_epi16(samples[i], tbl_entries[i]);
        }*/
        tbl_entries[0] = LOADLH(resample_table[_mm_extract_epi16(tbl_positions, 0)], resample_table[_mm_extract_epi16(tbl_positions, 1)]);
        tbl_entries[1] = LOADLH(resample_table[_mm_extract_epi16(tbl_positions, 2)], resample_table[_mm_extract_epi16(tbl_positions, 3)]);
        tbl_entries[2] = LOADLH(resample_table[_mm_extract_epi16(tbl_positions, 4)], resample_table[_mm_extract_epi16(tbl_positions, 5)]);
        tbl_entries[3] = LOADLH(resample_table[_mm_extract_epi16(tbl_positions, 6)], resample_table[_mm_extract_epi16(tbl_positions, 7)]);
        samples[0] = LOADLH(&in[_mm_extract_epi16(in_positions, 0)], &in[_mm_extract_epi16(in_positions, 1)]);
        samples[1] = LOADLH(&in[_mm_extract_epi16(in_positions, 2)], &in[_mm_extract_epi16(in_positions, 3)]);
        samples[2] = LOADLH(&in[_mm_extract_epi16(in_positions, 4)], &in[_mm_extract_epi16(in_positions, 5)]);
        samples[3] = LOADLH(&in[_mm_extract_epi16(in_positions, 6)], &in[_mm_extract_epi16(in_positions, 7)]);
        samples[0] = _mm_mulhrs_epi16(samples[0], tbl_entries[0]);
        samples[1] = _mm_mulhrs_epi16(samples[1], tbl_entries[1]);
        samples[2] = _mm_mulhrs_epi16(samples[2], tbl_entries[2]);
        samples[3] = _mm_mulhrs_epi16(samples[3], tbl_entries[3]);

        _mm_storeu_si128((__m128i *)out, _mm_hadds_epi16(_mm_hadds_epi16(samples[0], samples[1]), _mm_hadds_epi16(samples[2], samples[3])));

        acc_a = _mm_add_epi32(acc_a, pitchvec_8_steps);
        acc_b = _mm_add_epi32(acc_b, pitchvec_8_steps);
        out += 8;
        nbytes -= 8 * sizeof(int16_t);
    } while (nbytes > 0);
    in += (uint16_t)_mm_extract_epi16(acc_a, 1);
    pitch_accumulator = (uint16_t)_mm_extract_epi16(acc_a, 0);
#elif HAS_NEON
    static const uint16_t multiples_data[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    uint16x8_t multiples = vld1q_u16(multiples_data);
    uint32x4_t pitchvec_8_steps = vdupq_n_u32((pitch << 1) * 8);
    uint32x4_t pitchacclo_vec = vdupq_n_u32((uint16_t)pitch_accumulator);
    uint32x4_t acc_a = vmlal_n_u16(pitchacclo_vec, vget_low_u16(multiples), pitch);
    uint32x4_t acc_b = vmlal_n_u16(pitchacclo_vec, vget_high_u16(multiples), pitch);

    do {
        uint16x8x2_t unzipped = vuzpq_u16(vreinterpretq_u16_u32(acc_a), vreinterpretq_u16_u32(acc_b));
        uint16x8_t tbl_positions = vshrq_n_u16(unzipped.val[0], 10);
        uint16x8_t in_positions = unzipped.val[1];
        int16x8_t tbl_entries[4];
        int16x8_t samples[4];
        int16x8x2_t unzipped1;
        int16x8x2_t unzipped2;

        tbl_entries[0] = vcombine_s16(vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 0)]), vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 1)]));
        tbl_entries[1] = vcombine_s16(vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 2)]), vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 3)]));
        tbl_entries[2] = vcombine_s16(vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 4)]), vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 5)]));
        tbl_entries[3] = vcombine_s16(vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 6)]), vld1_s16(resample_table[vgetq_lane_u16(tbl_positions, 7)]));
        samples[0] = vcombine_s16(vld1_s16(&in[vgetq_lane_u16(in_positions, 0)]), vld1_s16(&in[vgetq_lane_u16(in_positions, 1)]));
        samples[1] = vcombine_s16(vld1_s16(&in[vgetq_lane_u16(in_positions, 2)]), vld1_s16(&in[vgetq_lane_u16(in_positions, 3)]));
        samples[2] = vcombine_s16(vld1_s16(&in[vgetq_lane_u16(in_positions, 4)]), vld1_s16(&in[vgetq_lane_u16(in_positions, 5)]));
        samples[3] = vcombine_s16(vld1_s16(&in[vgetq_lane_u16(in_positions, 6)]), vld1_s16(&in[vgetq_lane_u16(in_positions, 7)]));
        samples[0] = vqrdmulhq_s16(samples[0], tbl_entries[0]);
        samples[1] = vqrdmulhq_s16(samples[1], tbl_entries[1]);
        samples[2] = vqrdmulhq_s16(samples[2], tbl_entries[2]);
        samples[3] = vqrdmulhq_s16(samples[3], tbl_entries[3]);

        unzipped1 = vuzpq_s16(samples[0], samples[1]);
        unzipped2 = vuzpq_s16(samples[2], samples[3]);
        samples[0] = vqaddq_s16(unzipped1.val[0], unzipped1.val[1]);
        samples[1] = vqaddq_s16(unzipped2.val[0], unzipped2.val[1]);
        unzipped1 = vuzpq_s16(samples[0], samples[1]);
        samples[0] = vqaddq_s16(unzipped1.val[0], unzipped1.val[1]);

        vst1q_s16(out, samples[0]);

        acc_a = vaddq_u32(acc_a, pitchvec_8_steps);
        acc_b = vaddq_u32(acc_b, pitchvec_8_steps);
        out += 8;
        nbytes -= 8 * sizeof(int16_t);
    } while (nbytes > 0);
    in += vgetq_lane_u16(vreinterpretq_u16_u32(acc_a), 1);
    pitch_accumulator = vgetq_lane_u16(vreinterpretq_u16_u32(acc_a), 0);
#else
    do {
        for (i = 0; i < 8; i++) {
            tbl = resample_table[pitch_accumulator * 64 >> 16];
            sample = ((in[0] * tbl[0] + 0x4000) >> 15) +
                     ((in[1] * tbl[1] + 0x4000) >> 15) +
                     ((in[2] * tbl[2] + 0x4000) >> 15) +
                     ((in[3] * tbl[3] + 0x4000) >> 15);
            *out++ = clamp16(sample);

            pitch_accumulator += (pitch << 1);
            in += pitch_accumulator >> 16;
            pitch_accumulator %= 0x10000;
        }
        nbytes -= 8 * sizeof(int16_t);
    } while (nbytes > 0);
#endif

    state[4] = (int16_t)pitch_accumulator;
    memcpy(state, in, 4 * sizeof(int16_t));
    i = (in - in_initial + 4) & 7;
    in -= i;
    if (i != 0) {
        i = -8 - i;
    }
    state[5] = i;
    memcpy(state + 8, in, 8 * sizeof(int16_t));
}
#endif

void aResampleImpl(uint8_t flags, uint16_t pitch, RESAMPLE_STATE state) {
    int16_t __attribute__((aligned(32))) tmp[32] = { 0 };
    int16_t* in_initial = BUF_S16(rspa.in);
    int16_t* in = in_initial;
    MEM_BARRIER_PREF(in);
    int16_t* out = BUF_S16(rspa.out);
    int nbytes = ROUND_UP_16(rspa.nbytes);
    uint32_t pitch_accumulator = 0;
    int i = 0;
    float* tbl_f = NULL;
    float sample_f = 0;
    size_t l;

    int16_t *dp, *sp;
    int32_t *wdp, *wsp;

    if (!(flags & A_INIT)) {
        dp = tmp;
        sp = state;

        wdp = (int32_t *)dp;
        wsp = (int32_t *)sp;

        if ((((uintptr_t)wdp | (uintptr_t)wsp) & 3) == 0) {
            for (l = 0; l < 8; l++)
                *wdp++ = *wsp++;
        } else {
            for (l = 0; l < 16; l++)
                *dp++ = *sp++;
        }
    }

    in -= 4;
    pitch_accumulator = (uint16_t) tmp[4];
    tbl_f = resample_table[pitch_accumulator >> 10];
    __builtin_prefetch(tbl_f);

    dp = in;
    sp = tmp;
    for (l = 0; l < 4; l++)
        *dp++ = *sp++;

    do {
        __builtin_prefetch(out);
        for (i = 0; i < 8; i++) {

            float in_f[4] = { (float) (int) in[0], (float) (int) in[1], (float) (int) in[2], (float) (int) in[3] };

            sample_f =
                shz_dot8f(in_f[0], in_f[1], in_f[2], in_f[3], tbl_f[0], tbl_f[1], tbl_f[2], tbl_f[3]) * 0.00003052f;

            MEM_BARRIER();
            pitch_accumulator += (pitch << 1);
            in += pitch_accumulator >> 16;
            MEM_BARRIER_PREF(in);
            pitch_accumulator %= 0x10000;
            MEM_BARRIER();
            *out++ = clamp16f((sample_f));
            MEM_BARRIER();
            tbl_f = resample_table[pitch_accumulator >> 10];
            MEM_BARRIER_PREF(tbl_f);
        }
        nbytes -= 8 * sizeof(int16_t);
    } while (nbytes > 0);

    state[4] = (int16_t) pitch_accumulator;
    dp = (int16_t*) (state);
    sp = in;
    for (l = 0; l < 4; l++)
        *dp++ = *sp++;

    i = (in - in_initial + 4) & 7;
    in -= i;
    if (i != 0) {
        i = -8 - i;
    }
    state[5] = i;
    dp = (int16_t*) (state + 8);
    sp = in;
    for (l = 0; l < 8; l++)
        *dp++ = *sp++;
}

/*@Note: Much Slowdown */
void aEnvMixerImpl(uint8_t flags, ENVMIX_STATE state) {
    int16_t *in = rspa.buf.as_s16 + rspa.in / sizeof(int16_t);
    int16_t *dry[2] = {rspa.buf.as_s16 + rspa.out / sizeof(int16_t), rspa.buf.as_s16 + rspa.dry_right / sizeof(int16_t)};
    int16_t *wet[2] = {rspa.buf.as_s16 + rspa.wet_left / sizeof(int16_t), rspa.buf.as_s16 + rspa.wet_right / sizeof(int16_t)};
    int nbytes = ROUND_UP_16(rspa.nbytes);

#if HAS_SSE41
    __m128 vols[2][2];
    __m128i dry_factor;
    __m128i wet_factor;
    __m128 target[2];
    __m128 rate[2];
    __m128i in_loaded;
    __m128i vol_s16;
    bool increasing[2];

    int c;

    if (flags & A_INIT) {
        float vol_init[2] = {rspa.vol[0], rspa.vol[1]};
        float rate_float[2] = {(float)rspa.rate[0] * (1.0f / 65536.0f), (float)rspa.rate[1] * (1.0f / 65536.0f)};
        float step_diff[2] = {vol_init[0] * (rate_float[0] - 1.0f), vol_init[1] * (rate_float[1] - 1.0f)};

        for (c = 0; c < 2; c++) {
            vols[c][0] = _mm_add_ps(
                _mm_set_ps1(vol_init[c]),
                _mm_mul_ps(_mm_set1_ps(step_diff[c]), _mm_setr_ps(1.0f / 8.0f, 2.0f / 8.0f, 3.0f / 8.0f, 4.0f / 8.0f)));
            vols[c][1] = _mm_add_ps(
                _mm_set_ps1(vol_init[c]),
                _mm_mul_ps(_mm_set1_ps(step_diff[c]), _mm_setr_ps(5.0f / 8.0f, 6.0f / 8.0f, 7.0f / 8.0f, 8.0f / 8.0f)));

            increasing[c] = rate_float[c] >= 1.0f;
            target[c] = _mm_set1_ps(rspa.target[c]);
            rate[c] = _mm_set1_ps(rate_float[c]);
        }

        dry_factor = _mm_set1_epi16(rspa.vol_dry);
        wet_factor = _mm_set1_epi16(rspa.vol_wet);

        memcpy(state + 32, &rate_float[0], 4);
        memcpy(state + 34, &rate_float[1], 4);
        state[36] = rspa.target[0];
        state[37] = rspa.target[1];
        state[38] = rspa.vol_dry;
        state[39] = rspa.vol_wet;
    } else {
        float floats[2];
        vols[0][0] = _mm_loadu_ps((const float *)state);
        vols[0][1] = _mm_loadu_ps((const float *)(state + 8));
        vols[1][0] = _mm_loadu_ps((const float *)(state + 16));
        vols[1][1] = _mm_loadu_ps((const float *)(state + 24));
        memcpy(floats, state + 32, 8);
        rate[0] = _mm_set1_ps(floats[0]);
        rate[1] = _mm_set1_ps(floats[1]);
        increasing[0] = floats[0] >= 1.0f;
        increasing[1] = floats[1] >= 1.0f;
        target[0] = _mm_set1_ps(state[36]);
        target[1] = _mm_set1_ps(state[37]);
        dry_factor = _mm_set1_epi16(state[38]);
        wet_factor = _mm_set1_epi16(state[39]);
    }
    do {
        in_loaded = _mm_loadu_si128((const __m128i *)in);
        in += 8;
        for (c = 0; c < 2; c++) {
            if (increasing[c]) {
                vols[c][0] = _mm_min_ps(vols[c][0], target[c]);
                vols[c][1] = _mm_min_ps(vols[c][1], target[c]);
            } else {
                vols[c][0] = _mm_max_ps(vols[c][0], target[c]);
                vols[c][1] = _mm_max_ps(vols[c][1], target[c]);
            }

            vol_s16 = _mm_packs_epi32(_mm_cvtps_epi32(vols[c][0]), _mm_cvtps_epi32(vols[c][1]));
            _mm_storeu_si128((__m128i *)dry[c],
                             _mm_adds_epi16(
                                 _mm_loadu_si128((const __m128i *)dry[c]),
                                 _mm_mulhrs_epi16(in_loaded, _mm_mulhrs_epi16(vol_s16, dry_factor))));
            dry[c] += 8;

            if (flags & A_AUX) {
                _mm_storeu_si128((__m128i *)wet[c],
                                 _mm_adds_epi16(
                                     _mm_loadu_si128((const __m128i *)wet[c]),
                                     _mm_mulhrs_epi16(in_loaded, _mm_mulhrs_epi16(vol_s16, wet_factor))));
                wet[c] += 8;
            }

            vols[c][0] = _mm_mul_ps(vols[c][0], rate[c]);
            vols[c][1] = _mm_mul_ps(vols[c][1], rate[c]);
        }

        nbytes -= 8 * sizeof(int16_t);
    } while (nbytes > 0);

    _mm_storeu_ps((float *)state, vols[0][0]);
    _mm_storeu_ps((float *)(state + 8), vols[0][1]);
    _mm_storeu_ps((float *)(state + 16), vols[1][0]);
    _mm_storeu_ps((float *)(state + 24), vols[1][1]);
#elif HAS_NEON
    float32x4_t vols[2][2];
    int16_t dry_factor;
    int16_t wet_factor;
    float32x4_t target[2];
    float rate[2];
    int16x8_t in_loaded;
    int16x8_t vol_s16;
    bool increasing[2];

    int c;

    if (flags & A_INIT) {
        float vol_init[2] = {rspa.vol[0], rspa.vol[1]};
        float rate_float[2] = {(float)rspa.rate[0] * (1.0f / 65536.0f), (float)rspa.rate[1] * (1.0f / 65536.0f)};
        float step_diff[2] = {vol_init[0] * (rate_float[0] - 1.0f), vol_init[1] * (rate_float[1] - 1.0f)};
        static const float step_dividers_data[2][4] = {{1.0f / 8.0f, 2.0f / 8.0f, 3.0f / 8.0f, 4.0f / 8.0f},
                                                      {5.0f / 8.0f, 6.0f / 8.0f, 7.0f / 8.0f, 8.0f / 8.0f}};
        float32x4_t step_dividers[2] = {vld1q_f32(step_dividers_data[0]), vld1q_f32(step_dividers_data[1])};

        for (c = 0; c < 2; c++) {
            vols[c][0] = vaddq_f32(vdupq_n_f32(vol_init[c]), vmulq_n_f32(step_dividers[0], step_diff[c]));
            vols[c][1] = vaddq_f32(vdupq_n_f32(vol_init[c]), vmulq_n_f32(step_dividers[1], step_diff[c]));
            increasing[c] = rate_float[c] >= 1.0f;
            target[c] = vdupq_n_f32(rspa.target[c]);
            rate[c] = rate_float[c];
        }

        dry_factor = rspa.vol_dry;
        wet_factor = rspa.vol_wet;

        memcpy(state + 32, &rate_float[0], 4);
        memcpy(state + 34, &rate_float[1], 4);
        state[36] = rspa.target[0];
        state[37] = rspa.target[1];
        state[38] = rspa.vol_dry;
        state[39] = rspa.vol_wet;
    } else {
        vols[0][0] = vreinterpretq_f32_s16(vld1q_s16(state));
        vols[0][1] = vreinterpretq_f32_s16(vld1q_s16(state + 8));
        vols[1][0] = vreinterpretq_f32_s16(vld1q_s16(state + 16));
        vols[1][1] = vreinterpretq_f32_s16(vld1q_s16(state + 24));
        memcpy(&rate[0], state + 32, 4);
        memcpy(&rate[1], state + 34, 4);
        increasing[0] = rate[0] >= 1.0f;
        increasing[1] = rate[1] >= 1.0f;
        target[0] = vdupq_n_f32(state[36]);
        target[1] = vdupq_n_f32(state[37]);
        dry_factor = state[38];
        wet_factor = state[39];
    }

    do {
        in_loaded = vld1q_s16(in);
        in += 8;
        for (c = 0; c < 2; c++) {
            if (increasing[c]) {
                vols[c][0] = vminq_f32(vols[c][0], target[c]);
                vols[c][1] = vminq_f32(vols[c][1], target[c]);
            } else {
                vols[c][0] = vmaxq_f32(vols[c][0], target[c]);
                vols[c][1] = vmaxq_f32(vols[c][1], target[c]);
            }

            vol_s16 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(vols[c][0])), vqmovn_s32(vcvtq_s32_f32(vols[c][1])));
            vst1q_s16(dry[c], vqaddq_s16(vld1q_s16(dry[c]), vqrdmulhq_s16(in_loaded, vqrdmulhq_n_s16(vol_s16, dry_factor))));
            dry[c] += 8;
            if (flags & A_AUX) {
                vst1q_s16(wet[c], vqaddq_s16(vld1q_s16(wet[c]), vqrdmulhq_s16(in_loaded, vqrdmulhq_n_s16(vol_s16, wet_factor))));
                wet[c] += 8;
            }
            vols[c][0] = vmulq_n_f32(vols[c][0], rate[c]);
            vols[c][1] = vmulq_n_f32(vols[c][1], rate[c]);
        }

        nbytes -= 8 * sizeof(int16_t);
    } while (nbytes > 0);

    vst1q_s16(state, vreinterpretq_s16_f32(vols[0][0]));
    vst1q_s16(state + 8, vreinterpretq_s16_f32(vols[0][1]));
    vst1q_s16(state + 16, vreinterpretq_s16_f32(vols[1][0]));
    vst1q_s16(state + 24, vreinterpretq_s16_f32(vols[1][1]));
#else
    int16_t target[2];
    int32_t rate[2];
    int16_t vol_dry, vol_wet;

    int32_t step_diff[2];
    int32_t vols[2][8];

    int c, i;

    if (flags & A_INIT) {
        target[0] = rspa.target[0];
        target[1] = rspa.target[1];
        rate[0] = rspa.rate[0];
        rate[1] = rspa.rate[1];
        vol_dry = rspa.vol_dry;
//        vol_wet = rspa.vol_wet;
        step_diff[0] = rspa.vol[0] * (rate[0] - 0x10000) / 8;
        step_diff[1] = rspa.vol[0] * (rate[1] - 0x10000) / 8;

        for (i = 0; i < 8; i++) {
            vols[0][i] = clamp32((int64_t)(rspa.vol[0] << 16) + step_diff[0] * (i + 1));
            vols[1][i] = clamp32((int64_t)(rspa.vol[1] << 16) + step_diff[1] * (i + 1));
        }
    } else {
        n64_memcpy(vols[0], state, 32);
        n64_memcpy(vols[1], state + 16, 32);
        target[0] = state[32];
        target[1] = state[35];
        rate[0] = (state[33] << 16) | (uint16_t)state[34];
        rate[1] = (state[36] << 16) | (uint16_t)state[37];
        vol_dry = state[38];
//        vol_wet = state[39];
    }

    do {
        for (c = 0; c < 2; c++) {
            for (i = 0; i < 8; i++) {
                if ((rate[c] >> 16) > 0) {
                    // Increasing volume
                    if ((vols[c][i] >> 16) > target[c]) {
                        vols[c][i] = target[c] << 16;
                    }
                } else {
                    // Decreasing volume
                    if ((vols[c][i] >> 16) < target[c]) {
                        vols[c][i] = target[c] << 16;
                    }
                }
                dry[c][i] = clamp16((dry[c][i] * 0x7fff + in[i] * (((vols[c][i] >> 16) * vol_dry + 0x4000) >> 15) + 0x4000) >> 15);
                    //(dry[c][i] + (in[i]* vol_dry * (vols[c][i] >> 16)));
                    //(dry[c][i] * 0x7fff + in[i] * (((vols[c][i] >> 16) * vol_dry + 0x4000) >> 15) + 0x4000) >> 15);
//                if (flags & A_AUX) {
//                    wet[c][i] = clamp16((wet[c][i] * 0x7fff + in[i] * (((vols[c][i] >> 16) * vol_wet + 0x4000) >> 15) + 0x4000) >> 15);
//                }
                vols[c][i] = clamp32((int64_t)vols[c][i] * rate[c] >> 16);
            }

            dry[c] += 8;
//            if (flags & A_AUX) {
//                wet[c] += 8;
//            }
        }

        nbytes -= 16;
        in += 8;
    } while (nbytes > 0);

    n64_memcpy(state, vols[0], 32);
    n64_memcpy(state + 16, vols[1], 32);
    state[32] = target[0];
    state[35] = target[1];
    state[33] = (int16_t)(rate[0] >> 16);
    state[34] = (int16_t)rate[0];
    state[36] = (int16_t)(rate[1] >> 16);
    state[37] = (int16_t)rate[1];
    state[38] = vol_dry;
    state[39] = vol_wet;
#endif
}

/*@Note: Yes Slowdown */
void aMixImpl(int16_t gain, uint16_t in_addr, uint16_t out_addr) {
    int nbytes = ROUND_UP_32(rspa.nbytes);
    int16_t *in = rspa.buf.as_s16 + in_addr / sizeof(int16_t);
    int16_t *out = rspa.buf.as_s16 + out_addr / sizeof(int16_t);
#if HAS_SSE41
    __m128i gain_vec = _mm_set1_epi16(gain);
#elif !HAS_NEON
    int i;
    int32_t sample;
#endif

#if !HAS_NEON
    if (gain == -0x8000) {
        while (nbytes > 0) {
#if HAS_SSE41
            __m128i out1, out2, in1, in2;
            out1 = _mm_loadu_si128((const __m128i *)out);
            out2 = _mm_loadu_si128((const __m128i *)(out + 8));
            in1 = _mm_loadu_si128((const __m128i *)in);
            in2 = _mm_loadu_si128((const __m128i *)(in + 8));

            out1 = _mm_subs_epi16(out1, in1);
            out2 = _mm_subs_epi16(out2, in2);

            _mm_storeu_si128((__m128i *)out, out1);
            _mm_storeu_si128((__m128i *)(out + 8), out2);

            out += 16;
            in += 16;
#else
            for (i = 0; i < 16; i++) {
                sample = *out - *in++;
                *out++ = clamp16(sample);
            }
#endif

            nbytes -= 16 * sizeof(int16_t);
        }
    }
#endif

    while (nbytes > 0) {
#if HAS_SSE41
        __m128i out1, out2, in1, in2;
        out1 = _mm_loadu_si128((const __m128i *)out);
        out2 = _mm_loadu_si128((const __m128i *)(out + 8));
        in1 = _mm_loadu_si128((const __m128i *)in);
        in2 = _mm_loadu_si128((const __m128i *)(in + 8));

        out1 = _mm_adds_epi16(out1, _mm_mulhrs_epi16(in1, gain_vec));
        out2 = _mm_adds_epi16(out2, _mm_mulhrs_epi16(in2, gain_vec));

        _mm_storeu_si128((__m128i *)out, out1);
        _mm_storeu_si128((__m128i *)(out + 8), out2);

        out += 16;
        in += 16;
#elif HAS_NEON
        int16x8_t out1, out2, in1, in2;
        out1 = vld1q_s16(out);
        out2 = vld1q_s16(out + 8);
        in1 = vld1q_s16(in);
        in2 = vld1q_s16(in + 8);

        out1 = vqaddq_s16(out1, vqrdmulhq_n_s16(in1, gain));
        out2 = vqaddq_s16(out2, vqrdmulhq_n_s16(in2, gain));

        vst1q_s16(out, out1);
        vst1q_s16(out + 8, out2);

        out += 16;
        in += 16;
#else
        for (i = 0; i < 16; i++) {
            sample = ((*out * 0x7fff + *in++ * gain) + 0x4000) >> 15;
            *out++ = clamp16(sample);
        }
#endif

        nbytes -= 16 * sizeof(int16_t);
    }
}
