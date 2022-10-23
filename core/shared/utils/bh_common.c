/*
 * Copyright (C) 2019 Intel Corporation.  All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "bh_common.h"


static uint8 *
align_ptr(const uint8 *p, uint32 b)
{
    uintptr_t v = (uintptr_t)p;
    uintptr_t m = b - 1;
    return (uint8 *)((v + m) & ~m);
}
/*
int
b_memcpy_aw(char *dest, unsigned int dlen, char *src, unsigned int plen)
{
    if(dest == NULL || src == NULL) {
        return -1;
    }

    if(dlen >= plen) {
        return -1;
    }

    char *pa = align_ptr(src, sizeof(unsigned int));
    char *pb = align_ptr((src + plen), sizeof(unsigned int)) - sizeof(unsigned int);

    char *p_pre = pa, *p_suf =pa, *p_read=pa;

    if (pa > src) {
        p_pre = pa - sizeof(unsigned int);
    }
    if (pa < pb) {
        p_suf = pb;
    }

    unsigned int pre_offset = 0;
    unsigned int pre_size = 0;
    if (p_pre != p_read) {
        pre_offset = src - p_pre;
        if (src + plen > p_read) {
            pre_size = p_read - src;
        }
        else {
            pre_size = plen;
        }
    }

    unsigned int read_size = 0;
    unsigned int suf_offset = 0;
    unsigned int suf_size = 0;
    if (p_suf != p_read) {
        read_size = p_suf - p_read;
        suf_size = src + plen - p_suf;
    }
    else {
        if (src + plen > pa) {
            read_size = src + plen - pa;
        }
    }
    if((pre_size + read_size + suf_size) != plen) {
        return -1;
    }

    // copy pre segment
    unsigned int buff;
    char* pbuff = &buff;
    buff = (*(unsigned int*)p_pre);
    bh_memcpy_s(dest, pre_size, pbuff + pre_offset,
                pre_size);

    // copy segment
    if (read_size < 4) {
        buff = (*(unsigned int*)p_read);
        bh_memcpy_s(dest + pre_size, read_size, pbuff, read_size);
    }
    else {
        unsigned int* des = (unsigned int*)(dest + pre_size);
        unsigned int* src = (unsigned int*)p_read;
        for(int i = 0;i < read_size/4;i++)
            *des++ = *src++;
    }

    // copy suffix segment
    buff = (*(unsigned int*)p_suf);
    bh_memcpy_s(dest + pre_size + read_size, suf_size,
                pbuff, suf_size);

    return 0;
}
*/

uint8 *
b_memcpy_aw(uint8 *dest, size_t dlen, uint8 *p, size_t plen)
{
  bh_assert(p);
  bh_assert(dlen >= plen);

  uint8 *pa = align_ptr(p, sizeof(uint32));
  uint8 *pb = align_ptr((p + plen), sizeof(uint32)) - sizeof(uint32);

  uint8 *p_pre_read = pa;
  uint8 *p_suf_read = pa;
  uint8 *p_read = pa;

  if (pa > p) {
      p_pre_read = pa - sizeof(uint32);
  }
  if (pa < pb) {
      p_suf_read = pb;
  }

  uint32 pre_read_valid_offset = 0;
  uint32 pre_read_valid_size = 0;
  if (p_pre_read != p_read) {
      pre_read_valid_offset = p - p_pre_read;
      if (p + plen > p_read) {
          pre_read_valid_size = p_read - p;
      }
      else {
          pre_read_valid_size = plen;
      }
  }

  uint32 read_size = 0;
  uint32 suf_read_valid_offset = 0;
  uint32 suf_read_valid_size = 0;
  if (p_suf_read != p_read) {
      read_size = p_suf_read - p_read;
      suf_read_valid_size = p + plen - p_suf_read;
  }
  else {
      if (p + plen > pa) {
          read_size = p + plen - pa;
      }
  }
  bh_assert((pre_read_valid_size + read_size + suf_read_valid_size) == plen);

  // copy pre segment
  uint32 buff;
  uint8* pbuff = &buff;
  buff = (*(uint32*)p_pre_read);
  bh_memcpy_s(dest, pre_read_valid_size, pbuff + pre_read_valid_offset,
              pre_read_valid_size);

  // copy segment
  if (read_size < 4) {
      buff = (*(uint32*)p_read);
      bh_memcpy_s(dest + pre_read_valid_size, read_size, pbuff, read_size);
  }
  else {
      uint32* des = (uint32*)(dest + pre_read_valid_size);
      uint32* src = (uint32*)p_read;
      for(int i = 0;i < read_size/4;i++)
          *des++ = *src++;
  }

  // copy suffix segment
  buff = (*(uint32*)p_suf_read);
  bh_memcpy_s(dest + pre_read_valid_size + read_size, suf_read_valid_size,
              pbuff, suf_read_valid_size);

  return dest;
}

int
b_memcpy_s(void *s1, unsigned int s1max, const void *s2, unsigned int n)
{
    char *dest = (char *)s1;
    char *src = (char *)s2;
    if (n == 0) {
        return 0;
    }

    if (s1 == NULL) {
        return -1;
    }
    if (s2 == NULL || n > s1max) {
        memset(dest, 0, s1max);
        return -1;
    }
    memcpy(dest, src, n);
    return 0;
}

int
b_memmove_s(void *s1, unsigned int s1max, const void *s2, unsigned int n)
{
    char *dest = (char *)s1;
    char *src = (char *)s2;
    if (n == 0) {
        return 0;
    }

    if (s1 == NULL) {
        return -1;
    }
    if (s2 == NULL || n > s1max) {
        memset(dest, 0, s1max);
        return -1;
    }
    memmove(dest, src, n);
    return 0;
}

int
b_strcat_s(char *s1, unsigned int s1max, const char *s2)
{
    if (NULL == s1 || NULL == s2 || s1max < (strlen(s1) + strlen(s2) + 1)) {
        return -1;
    }

    memcpy(s1 + strlen(s1), s2, strlen(s2) + 1);
    return 0;
}

int
b_strcpy_s(char *s1, unsigned int s1max, const char *s2)
{
    if (NULL == s1 || NULL == s2 || s1max < (strlen(s2) + 1)) {
        return -1;
    }

    memcpy(s1, s2, strlen(s2) + 1);
    return 0;
}

char *
bh_strdup(const char *s)
{
    unsigned int size;
    char *s1 = NULL;

    if (s) {
        size = (unsigned int)(strlen(s) + 1);
        if ((s1 = BH_MALLOC(size)))
            bh_memcpy_s(s1, size, s, size);
    }
    return s1;
}

char *
wa_strdup(const char *s)
{
    unsigned int size;
    char *s1 = NULL;

    if (s) {
        size = (unsigned int)(strlen(s) + 1);
        if ((s1 = WA_MALLOC(size)))
            bh_memcpy_s(s1, size, s, size);
    }
    return s1;
}
