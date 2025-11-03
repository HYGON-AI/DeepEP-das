/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

/* #undef DEBUG */
/* #undef PROFILE */
/* #undef USE_RO */
/* #undef USE_IPC */
#define USE_GDA
/* #undef USE_THREADS */
/* #undef USE_SHARED_CTX */
/* #undef USE_WF_COAL */
#define USE_HEAP_DEVICE_FINEGRAIN
/* #undef USE_HEAP_DEVICE_UNCACHED */
/* #undef USE_HEAP_DEVICE_COARSEGRAIN */
/* #undef USE_HEAP_MANAGED */
/* #undef USE_HEAP_HOST_HIP */
/* #undef USE_HEAP_HOST */
#define USE_ALLOC_DLMALLOC
/* #undef USE_ALLOC_POW2BINS */
/* #undef USE_FUNC_CALL */
/* #undef USE_SINGLE_NODE */
/* #undef USE_HDP_FLUSH */
/* #undef USE_HDP_FLUSH_HOST_SIDE */
/* #undef GDA_IONIC */
/* #undef GDA_BNXT */
#define GDA_MLX5
#define HAVE_EXTERNAL_MPI
