/* 由 tools/clean_generic.py 从 OpenBLAS 提取的 generic.c
 * 来源：30_dsymv_op\dsymv_k.c
 * 清洗：去掉 common.h 依赖，仅保留最小 typedef；函数体保持原样（未优化参考实现）。
 */
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

typedef long long BLASLONG;
typedef int blasint;

#include <stdio.h>

static __inline void SYMCOPY_L(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a12;
  double a21, a22;

  b1 = b;
  b2 = b;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 2;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 1 * m;
    b1 += 2 * m + 2;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 1 * m;
    b2 += 2 * m + 2;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);

      a22 = *(aa2 + 1);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb2 + 0) = a21;
      *(bb2 + 1) = a22;
      aa1 += 2;
      aa2 += 2;
      bb1 += 2;
      bb2 += 2;

      cc1 += 2 * m;
      cc2 += 2 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 aa1 += 2;
 aa2 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
 *(cc2 + 0) = a21;
 *(cc2 + 1) = a22;

 bb1 += 2;
 bb2 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;

 is --;
      }

      is = ((m - js - 2) & 1);

      if (is == 1){
 a11 = *(aa1 + 0);
 a12 = *(aa2 + 0);

 *(bb1 + 0) = a11;
 *(bb2 + 0) = a12;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
    }

  }
}

static __inline void SYMCOPY_U(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a12;
  double a21, a22;

  b1 = b;
  b2 = b;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 1 * m;
    b1 += 2 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 1 * m;
    b2 += 2;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 aa1 += 2;
 aa2 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
 *(cc2 + 0) = a21;
 *(cc2 + 1) = a22;

 bb1 += 2;
 bb2 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;
      }

      a11 = *(aa1 + 0);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a12;
      *(bb2 + 0) = a12;
      *(bb2 + 1) = a22;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 aa1 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(cc1 + 0) = a11;
 *(cc2 + 0) = a21;
 bb1 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;
      }

      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
    }
  }
}


static __inline void ZSYMCOPY_L(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 4;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m + 4;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4 * m + 4;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      a31 = *(aa1 + 2);
      a41 = *(aa1 + 3);

      a12 = *(aa2 + 2);
      a22 = *(aa2 + 3);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb1 + 2) = a31;
      *(bb1 + 3) = a41;

      *(bb2 + 0) = a31;
      *(bb2 + 1) = a41;
      *(bb2 + 2) = a12;
      *(bb2 + 3) = a22;

      aa1 += 4;
      aa2 += 4;
      bb1 += 4;
      bb2 += 4;

      cc1 += 4 * m;
      cc2 += 4 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;

 is --;
      }

      if (m & 1){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
    }

  }
}

static __inline void ZSYMCOPY_U(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);
      a32 = *(aa2 + 2);
      a42 = *(aa2 + 3);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb1 + 2) = a12;
      *(bb1 + 3) = a22;

      *(bb2 + 0) = a12;
      *(bb2 + 1) = a22;
      *(bb2 + 2) = a32;
      *(bb2 + 3) = a42;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);
 aa1 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 bb1 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
    }
  }
}

static __inline void ZHEMCOPY_L(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 4;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m + 4;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4 * m + 4;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a31 = *(aa1 + 2);
      a41 = *(aa1 + 3);

      a12 = *(aa2 + 2);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
      *(bb1 + 2) = a31;
      *(bb1 + 3) = a41;

      *(bb2 + 0) = a31;
      *(bb2 + 1) = -a41;
      *(bb2 + 2) = a12;
      *(bb2 + 3) = 0.;

      aa1 += 4;
      aa2 += 4;
      bb1 += 4;
      bb2 += 4;

      cc1 += 4 * m;
      cc2 += 4 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = -a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = -a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = -a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = -a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;

 is --;
      }

      if (m & 1){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = -a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = -a22;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
    }

  }
}

static __inline void ZHEMCOPY_U(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = -a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = -a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = -a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = -a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);
      a32 = *(aa2 + 2);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
      *(bb1 + 2) = a12;
      *(bb1 + 3) = -a22;

      *(bb2 + 0) = a12;
      *(bb2 + 1) = a22;
      *(bb2 + 2) = a32;
      *(bb2 + 3) = 0.;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);
 aa1 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = -a21;
 *(cc2 + 0) = a31;
 *(cc2 + 1) = -a41;
 bb1 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
    }
  }
}


static __inline void ZHEMCOPY_M(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 4;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m + 4;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4 * m + 4;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a31 = *(aa1 + 2);
      a41 = *(aa1 + 3);

      a12 = *(aa2 + 2);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
      *(bb1 + 2) = a31;
      *(bb1 + 3) = -a41;

      *(bb2 + 0) = a31;
      *(bb2 + 1) = a41;
      *(bb2 + 2) = a12;
      *(bb2 + 3) = 0.;

      aa1 += 4;
      aa2 += 4;
      bb1 += 4;
      bb2 += 4;

      cc1 += 4 * m;
      cc2 += 4 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = -a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = -a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = -a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = -a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;

 is --;
      }

      if (m & 1){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 *(bb1 + 0) = a11;
 *(bb1 + 1) = -a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = -a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
    }

  }
}

static __inline void ZHEMCOPY_V(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = -a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = -a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = -a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = -a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);
      a32 = *(aa2 + 2);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
      *(bb1 + 2) = a12;
      *(bb1 + 3) = a22;

      *(bb2 + 0) = a12;
      *(bb2 + 1) = -a22;
      *(bb2 + 2) = a32;
      *(bb2 + 3) = 0.;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);
 aa1 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = -a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = -a41;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 bb1 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = 0.;
    }
  }
}


static __inline void TRMCOPY_NL(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a12;
  double a21, a22;

  b1 = b;
  b2 = b;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 2;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 1 * m;
    b1 += 2 * m + 2;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 1 * m;
    b2 += 2 * m + 2;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);

      a22 = *(aa2 + 1);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb2 + 0) = a21;
      *(bb2 + 1) = a22;
      aa1 += 2;
      aa2 += 2;
      bb1 += 2;
      bb2 += 2;

      cc1 += 2 * m;
      cc2 += 2 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 aa1 += 2;
 aa2 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
 *(cc2 + 0) = a21;
 *(cc2 + 1) = a22;

 bb1 += 2;
 bb2 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;

 is --;
      }

      is = ((m - js - 2) & 1);

      if (is == 1){
 a11 = *(aa1 + 0);
 a12 = *(aa2 + 0);

 *(bb1 + 0) = a11;
 *(bb2 + 0) = a12;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
    }

  }
}

static __inline void TRMCOPY_TL(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a12;
  double a21, a22;

  b1 = b;
  b2 = b;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 2;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 1 * m;
    b1 += 2 * m + 2;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 1 * m;
    b2 += 2 * m + 2;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);

      a22 = *(aa2 + 1);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb2 + 0) = a21;
      *(bb2 + 1) = a22;
      aa1 += 2;
      aa2 += 2;
      bb1 += 2;
      bb2 += 2;

      cc1 += 2 * m;
      cc2 += 2 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 aa1 += 2;
 aa2 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
 *(cc2 + 0) = a21;
 *(cc2 + 1) = a22;

 bb1 += 2;
 bb2 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;

 is --;
      }

      is = ((m - js - 2) & 1);

      if (is == 1){
 a11 = *(aa1 + 0);
 a12 = *(aa2 + 0);

 *(bb1 + 0) = a11;
 *(bb2 + 0) = a12;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
    }

  }
}

static __inline void TRMCOPY_NU(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a12;
  double a21, a22;

  b1 = b;
  b2 = b;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 1 * m;
    b1 += 2 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 1 * m;
    b2 += 2;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 aa1 += 2;
 aa2 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
 *(cc2 + 0) = a21;
 *(cc2 + 1) = a22;

 bb1 += 2;
 bb2 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;
      }

      a11 = *(aa1 + 0);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a12;
      *(bb2 + 0) = a12;
      *(bb2 + 1) = a22;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 aa1 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(cc1 + 0) = a11;
 *(cc2 + 0) = a21;
 bb1 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;
      }

      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
    }
  }
}

static __inline void TRMCOPY_TU(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a12;
  double a21, a22;

  b1 = b;
  b2 = b;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 1 * m;
    b1 += 2 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 1 * m;
    b2 += 2;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 aa1 += 2;
 aa2 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a12;
 *(cc2 + 0) = a21;
 *(cc2 + 1) = a22;

 bb1 += 2;
 bb2 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;
      }

      a11 = *(aa1 + 0);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a12;
      *(bb2 + 0) = a12;
      *(bb2 + 1) = a22;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 aa1 += 2;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(cc1 + 0) = a11;
 *(cc2 + 0) = a21;
 bb1 += 2;

 cc1 += 2 * m;
 cc2 += 2 * m;
      }

      a11 = *(aa1 + 0);
      *(bb1 + 0) = a11;
    }
  }
}

static __inline void ZTRMCOPY_NL(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 4;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m + 4;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4 * m + 4;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      a31 = *(aa1 + 2);
      a41 = *(aa1 + 3);

      a12 = *(aa2 + 2);
      a22 = *(aa2 + 3);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb1 + 2) = a31;
      *(bb1 + 3) = a41;

      *(bb2 + 0) = a31;
      *(bb2 + 1) = a41;
      *(bb2 + 2) = a12;
      *(bb2 + 3) = a22;

      aa1 += 4;
      aa2 += 4;
      bb1 += 4;
      bb2 += 4;

      cc1 += 4 * m;
      cc2 += 4 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;

 is --;
      }

      if (m & 1){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
    }

  }
}

static __inline void ZTRMCOPY_TL(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda + 4;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m + 4;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4 * m + 4;

    if (m - js >= 2){

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      a31 = *(aa1 + 2);
      a41 = *(aa1 + 3);

      a12 = *(aa2 + 2);
      a22 = *(aa2 + 3);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb1 + 2) = a31;
      *(bb1 + 3) = a41;

      *(bb2 + 0) = a31;
      *(bb2 + 1) = a41;
      *(bb2 + 2) = a12;
      *(bb2 + 3) = a22;

      aa1 += 4;
      aa2 += 4;
      bb1 += 4;
      bb2 += 4;

      cc1 += 4 * m;
      cc2 += 4 * m;

      is = ((m - js - 2) >> 1);

      while (is > 0){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;

 is --;
      }

      if (m & 1){
 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;
      }
    }

    if (m - js == 1){
      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
    }

  }
}

static __inline void ZTRMCOPY_NU(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);
      a32 = *(aa2 + 2);
      a42 = *(aa2 + 3);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb1 + 2) = a12;
      *(bb1 + 3) = a22;

      *(bb2 + 0) = a12;
      *(bb2 + 1) = a22;
      *(bb2 + 2) = a32;
      *(bb2 + 3) = a42;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);
 aa1 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 bb1 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
    }
  }
}

static __inline void ZTRMCOPY_TU(BLASLONG m, double *a, BLASLONG lda, double *b){
  BLASLONG is, js;

  double *aa1, *aa2;
  double *b1, *b2;
  double *bb1, *bb2;
  double *cc1, *cc2;
  double a11, a21, a31, a41;
  double a12, a22, a32, a42;

  b1 = b;
  b2 = b;

  lda *= 2;

  for (js = 0; js < m; js += 2){

    aa1 = a + 0 * lda;
    aa2 = a + 1 * lda;
    a += 2 * lda;

    bb1 = b1 + 0 * m;
    bb2 = b1 + 2 * m;
    b1 += 4 * m;

    cc1 = b2 + 0 * m;
    cc2 = b2 + 2 * m;
    b2 += 4;

    if (m - js >= 2){

      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);

 a12 = *(aa2 + 0);
 a22 = *(aa2 + 1);
 a32 = *(aa2 + 2);
 a42 = *(aa2 + 3);

 aa1 += 4;
 aa2 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(bb2 + 0) = a12;
 *(bb2 + 1) = a22;
 *(bb2 + 2) = a32;
 *(bb2 + 3) = a42;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc1 + 2) = a12;
 *(cc1 + 3) = a22;

 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 *(cc2 + 2) = a32;
 *(cc2 + 3) = a42;

 bb1 += 4;
 bb2 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);

      a12 = *(aa2 + 0);
      a22 = *(aa2 + 1);
      a32 = *(aa2 + 2);
      a42 = *(aa2 + 3);

      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
      *(bb1 + 2) = a12;
      *(bb1 + 3) = a22;

      *(bb2 + 0) = a12;
      *(bb2 + 1) = a22;
      *(bb2 + 2) = a32;
      *(bb2 + 3) = a42;
    }

    if (m - js == 1){
      for (is = 0; is < js; is += 2){

 a11 = *(aa1 + 0);
 a21 = *(aa1 + 1);
 a31 = *(aa1 + 2);
 a41 = *(aa1 + 3);
 aa1 += 4;

 *(bb1 + 0) = a11;
 *(bb1 + 1) = a21;
 *(bb1 + 2) = a31;
 *(bb1 + 3) = a41;

 *(cc1 + 0) = a11;
 *(cc1 + 1) = a21;
 *(cc2 + 0) = a31;
 *(cc2 + 1) = a41;
 bb1 += 4;

 cc1 += 4 * m;
 cc2 += 4 * m;
      }

      a11 = *(aa1 + 0);
      a21 = *(aa1 + 1);
      *(bb1 + 0) = a11;
      *(bb1 + 1) = a21;
    }
  }
}

int dsymv_L(BLASLONG m, BLASLONG offset, double alpha, double *a, BLASLONG lda,
   double *x, BLASLONG incx, double *y, BLASLONG incy, double *buffer){

  BLASLONG is, min_i;
  double *X = x;
  double *Y = y;
  double *symbuffer = buffer;
  double *gemvbuffer = (double *)(((BLASLONG)buffer + 16 * 16 * sizeof(double) + 4095) & ~4095);
  double *bufferY = gemvbuffer;
  double *bufferX = gemvbuffer;

  if (incy != 1) {
    Y = bufferY;
    bufferX = (double *)(((BLASLONG)bufferY + m * sizeof(double) + 4095) & ~4095);
    gemvbuffer = bufferX;
    dcopy_k(m, y, incy, Y, 1);
  }

  if (incx != 1) {
    X = bufferX;
    gemvbuffer = (double *)(((BLASLONG)bufferX + m * sizeof(double) + 4095) & ~4095);
    dcopy_k(m, x, incx, X, 1);
  }





  for(is = 0; is < offset; is += 16){
    min_i = (offset - is>16? 16:offset - is);

    SYMCOPY_L(min_i, a + is + is * lda, lda, symbuffer);




    dgemv_n(min_i, min_i, 0, alpha,
    symbuffer, min_i,
    X + is, 1,
    Y + is, 1, gemvbuffer);


    if (m - is > min_i){
      dgemv_t(m - is - min_i, min_i, 0, alpha,
      a + (is + min_i) + is * lda, lda,
      X + (is + min_i), 1,
      Y + is, 1, gemvbuffer);

      dgemv_n(m - is - min_i, min_i, 0, alpha,
      a + (is + min_i) + is * lda, lda,
      X + is, 1,
      Y + (is + min_i), 1, gemvbuffer);
    }


  }

  if (incy != 1) {
    dcopy_k(m, Y, 1, y, incy);
  }

  return 0;
}
