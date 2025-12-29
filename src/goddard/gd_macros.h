#ifndef GD_MACROS_H
#define GD_MACROS_H

/**
 * @file gd_macros.h
 * 
 * Common macros that Goddard used throughout the Mario Head subsytem code.
 */
//57.29577950560105
#define DEG_PER_RAD 57.2957795f 
#define RAD_PER_DEG 0.01745329f
//(1.0 / DEG_PER_RAD)

#define ABS(val) (((val) < 0 ? (-(val)) : (val)))
#define SQ(val) ((val) * (val))
#define ALIGN(VAL_, ALIGNMENT_) (((VAL_) + ((ALIGNMENT_) - 1)) & ~((ALIGNMENT_) - 1))

#endif // GD_MACROS_H
