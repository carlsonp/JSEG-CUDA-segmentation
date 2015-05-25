
#ifndef _JSEG_KERNEL_H_
#define _JSEG_KERNEL_H_

#include "matrix.h"


// JSEG kernel thread
void JSEGOnDevice(const Matrix M, Matrix P, int window_size);

#endif // #ifndef _JSEG_KERNEL_H_
