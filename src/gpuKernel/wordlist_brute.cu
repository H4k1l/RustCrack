// Copyright 2025 Hakil
// Licensed under the Apache License, Version 2.0

// -- hashfnct -- //

// md5

#include <cstdio>
#define MD5_BLOCK_SIZE 16
typedef unsigned char BYTE;
#ifndef WORD32_DEFINED
#define WORD32_DEFINED
typedef unsigned int WORD32;
#endif
typedef struct {
    BYTE data[64];
    WORD32 datalen;
    unsigned long long bitlen;
    WORD32 state[4];
} CUDA_MD5_CTX;
#define ROTLEFT(a,b) ((a << b) | (a >> (32-b)))
#define F(x,y,z) ((x & y) | (~x & z))
#define G(x,y,z) ((x & z) | (y & ~z))
#define H(x,y,z) (x ^ y ^ z)
#define I(x,y,z) (y ^ (x | ~z))
#define FF(a,b,c,d,m,s,t) { a += F(b,c,d) + m + t; a = b + ROTLEFT(a,s); }
#define GG(a,b,c,d,m,s,t) { a += G(b,c,d) + m + t; a = b + ROTLEFT(a,s); }
#define HH(a,b,c,d,m,s,t) { a += H(b,c,d) + m + t; a = b + ROTLEFT(a,s); }
#define II(a,b,c,d,m,s,t) { a += I(b,c,d) + m + t; a = b + ROTLEFT(a,s); }

__device__ void cuda_md5_transform(CUDA_MD5_CTX *ctx, const BYTE data[])
{
    WORD32 a, b, c, d, m[16], i, j;

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j]) + (data[j + 1] << 8) + (data[j + 2] << 16) + (data[j + 3] << 24);

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];

    FF(a,b,c,d,m[0],  7,0xd76aa478); FF(d,a,b,c,m[1], 12,0xe8c7b756); FF(c,d,a,b,m[2], 17,0x242070db); FF(b,c,d,a,m[3], 22,0xc1bdceee);
    FF(a,b,c,d,m[4],  7,0xf57c0faf); FF(d,a,b,c,m[5], 12,0x4787c62a); FF(c,d,a,b,m[6], 17,0xa8304613); FF(b,c,d,a,m[7], 22,0xfd469501);
    FF(a,b,c,d,m[8],  7,0x698098d8); FF(d,a,b,c,m[9], 12,0x8b44f7af); FF(c,d,a,b,m[10],17,0xffff5bb1); FF(b,c,d,a,m[11],22,0x895cd7be);
    FF(a,b,c,d,m[12], 7,0x6b901122); FF(d,a,b,c,m[13],12,0xfd987193); FF(c,d,a,b,m[14],17,0xa679438e); FF(b,c,d,a,m[15],22,0x49b40821);

    GG(a,b,c,d,m[1],  5,0xf61e2562); GG(d,a,b,c,m[6],  9,0xc040b340); GG(c,d,a,b,m[11],14,0x265e5a51); GG(b,c,d,a,m[0], 20,0xe9b6c7aa);
    GG(a,b,c,d,m[5],  5,0xd62f105d); GG(d,a,b,c,m[10], 9,0x02441453); GG(c,d,a,b,m[15],14,0xd8a1e681); GG(b,c,d,a,m[4], 20,0xe7d3fbc8);
    GG(a,b,c,d,m[9],  5,0x21e1cde6); GG(d,a,b,c,m[14], 9,0xc33707d6); GG(c,d,a,b,m[3], 14,0xf4d50d87); GG(b,c,d,a,m[8], 20,0x455a14ed);
    GG(a,b,c,d,m[13], 5,0xa9e3e905); GG(d,a,b,c,m[2],  9,0xfcefa3f8); GG(c,d,a,b,m[7], 14,0x676f02d9); GG(b,c,d,a,m[12],20,0x8d2a4c8a);

    HH(a,b,c,d,m[5],  4,0xfffa3942); HH(d,a,b,c,m[8], 11,0x8771f681); HH(c,d,a,b,m[11],16,0x6d9d6122); HH(b,c,d,a,m[14],23,0xfde5380c);
    HH(a,b,c,d,m[1],  4,0xa4beea44); HH(d,a,b,c,m[4], 11,0x4bdecfa9); HH(c,d,a,b,m[7], 16,0xf6bb4b60); HH(b,c,d,a,m[10],23,0xbebfbc70);
    HH(a,b,c,d,m[13], 4,0x289b7ec6); HH(d,a,b,c,m[0], 11,0xeaa127fa); HH(c,d,a,b,m[3], 16,0xd4ef3085); HH(b,c,d,a,m[6], 23,0x04881d05);
    HH(a,b,c,d,m[9],  4,0xd9d4d039); HH(d,a,b,c,m[12],11,0xe6db99e5); HH(c,d,a,b,m[15],16,0x1fa27cf8); HH(b,c,d,a,m[2], 23,0xc4ac5665);

    II(a,b,c,d,m[0],  6,0xf4292244); II(d,a,b,c,m[7], 10,0x432aff97); II(c,d,a,b,m[14],15,0xab9423a7); II(b,c,d,a,m[5], 21,0xfc93a039);
    II(a,b,c,d,m[12], 6,0x655b59c3); II(d,a,b,c,m[3], 10,0x8f0ccc92); II(c,d,a,b,m[10],15,0xffeff47d); II(b,c,d,a,m[1], 21,0x85845dd1);
    II(a,b,c,d,m[8],  6,0x6fa87e4f); II(d,a,b,c,m[15],10,0xfe2ce6e0); II(c,d,a,b,m[6], 15,0xa3014314); II(b,c,d,a,m[13],21,0x4e0811a1);
    II(a,b,c,d,m[4],  6,0xf7537e82); II(d,a,b,c,m[11],10,0xbd3af235); II(c,d,a,b,m[2], 15,0x2ad7d2bb); II(b,c,d,a,m[9], 21,0xeb86d391);

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
}

__device__ void cuda_md5_init(CUDA_MD5_CTX *ctx)
{
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0x67452301; ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE; ctx->state[3] = 0x10325476;
}

__device__ void cuda_md5_update(CUDA_MD5_CTX *ctx, const BYTE data[], size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            cuda_md5_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_md5_final(CUDA_MD5_CTX *ctx, BYTE hash[])
{
    size_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        cuda_md5_transform(ctx, ctx->data);
        for (i = 0; i < 56; i++) ctx->data[i] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[56] = ctx->bitlen;       ctx->data[57] = ctx->bitlen >> 8;
    ctx->data[58] = ctx->bitlen >> 16; ctx->data[59] = ctx->bitlen >> 24;
    ctx->data[60] = ctx->bitlen >> 32; ctx->data[61] = ctx->bitlen >> 40;
    ctx->data[62] = ctx->bitlen >> 48; ctx->data[63] = ctx->bitlen >> 56;
    cuda_md5_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i]      = (ctx->state[0] >> (i * 8)) & 0x000000ff;
        hash[i + 4]  = (ctx->state[1] >> (i * 8)) & 0x000000ff;
        hash[i + 8]  = (ctx->state[2] >> (i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (i * 8)) & 0x000000ff;
    }
}

__device__ void compute_md5_cuda_single(char* word, int word_length, char* output_hash)
{

    CUDA_MD5_CTX ctx;
    BYTE hash[MD5_BLOCK_SIZE];
    
    cuda_md5_init(&ctx);
    
    cuda_md5_update(&ctx, (BYTE*)word, word_length);
    
    cuda_md5_final(&ctx, hash);

    for (int i = 0; i < MD5_BLOCK_SIZE; i++) {
        output_hash[i * 2]     = "0123456789abcdef"[hash[i] >> 4];
        output_hash[i * 2 + 1] = "0123456789abcdef"[hash[i] & 0x0F];
    }
    output_hash[32] = '\0';

}

#undef ROTRIGHT
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
#undef F
#undef G
#undef H
#undef I
#undef CH
#undef MAJ

// sha1

#define SHA1_BLOCK_SIZE 20
typedef unsigned char BYTE;
typedef unsigned int  WORD;
typedef struct {
    BYTE data[64];
    WORD datalen;
    unsigned long long bitlen;
    WORD state[5];
    WORD k[4];
} CUDA_SHA1_CTX;

#define ROTLEFT(a,b) ((a << b) | (a >> (32-b)))

__device__ void cuda_sha1_transform(CUDA_SHA1_CTX *ctx, const BYTE data[])
{
    WORD a, b, c, d, e, i, j, t, m[80];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) + (data[j + 1] << 16) + (data[j + 2] << 8) + (data[j + 3]);
    for (; i < 80; ++i) {
        m[i] = (m[i - 3] ^ m[i - 8] ^ m[i - 14] ^ m[i - 16]);
        m[i] = (m[i] << 1) | (m[i] >> 31);
    }

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3]; e = ctx->state[4];

    for (i = 0; i < 20; ++i) {
        t = ROTLEFT(a, 5) + ((b & c) ^ (~b & d)) + e + ctx->k[0] + m[i];
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = t;
    }
    for (; i < 40; ++i) {
        t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[1] + m[i];
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = t;
    }
    for (; i < 60; ++i) {
        t = ROTLEFT(a, 5) + ((b & c) ^ (b & d) ^ (c & d)) + e + ctx->k[2] + m[i];
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = t;
    }
    for (; i < 80; ++i) {
        t = ROTLEFT(a, 5) + (b ^ c ^ d) + e + ctx->k[3] + m[i];
        e = d; d = c; c = ROTLEFT(b, 30); b = a; a = t;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d; ctx->state[4] += e;
}

__device__ void cuda_sha1_init(CUDA_SHA1_CTX *ctx)
{
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0x67452301; ctx->state[1] = 0xEFCDAB89;
    ctx->state[2] = 0x98BADCFE; ctx->state[3] = 0x10325476; ctx->state[4] = 0xc3d2e1f0;
    ctx->k[0] = 0x5a827999; ctx->k[1] = 0x6ed9eba1; ctx->k[2] = 0x8f1bbcdc; ctx->k[3] = 0xca62c1d6;
}

__device__ void cuda_sha1_update(CUDA_SHA1_CTX *ctx, const BYTE data[], size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            cuda_sha1_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha1_final(CUDA_SHA1_CTX *ctx, BYTE hash[])
{
    size_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        cuda_sha1_transform(ctx, ctx->data);
        for (i = 0; i < 56; i++) ctx->data[i] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;       ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16; ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32; ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48; ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha1_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
    }
}

__device__ void compute_sha1_cuda_single(char* word, int word_length, char* output_hash)
{
    CUDA_SHA1_CTX ctx;
    BYTE hash[SHA1_BLOCK_SIZE];
    
    cuda_sha1_init(&ctx);
    
    cuda_sha1_update(&ctx, (BYTE*)word, word_length);
    
    cuda_sha1_final(&ctx, hash);

    for (int i = 0; i < SHA1_BLOCK_SIZE; i++) {
        output_hash[i * 2]     = "0123456789abcdef"[hash[i] >> 4];
        output_hash[i * 2 + 1] = "0123456789abcdef"[hash[i] & 0x0F];
    }
    output_hash[40] = '\0';
}
#undef ROTRIGHT
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
#undef F
#undef G
#undef H
#undef I
#undef CH
#undef MAJ

// sha224

#define SHA224_BLOCK_SIZE 28
typedef unsigned char BYTE;
#ifndef WORD32_DEFINED
#define WORD32_DEFINED
typedef unsigned int WORD32;  
#endif

typedef struct {
    BYTE data[64];
    WORD32 datalen;
    unsigned long long bitlen;
    WORD32 state[8];
} CUDA_SHA224_CTX;

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

__constant__ WORD32 k224[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ void cuda_sha224_transform(CUDA_SHA224_CTX *ctx, const BYTE data[])
{
    WORD32 a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k224[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

__device__ void cuda_sha224_init(CUDA_SHA224_CTX *ctx)
{
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0xc1059ed8; ctx->state[1] = 0x367cd507; ctx->state[2] = 0x3070dd17; ctx->state[3] = 0xf70e5939;
    ctx->state[4] = 0xffc00b31; ctx->state[5] = 0x68581511; ctx->state[6] = 0x64f98fa7; ctx->state[7] = 0xbefa4fa4;
}

__device__ void cuda_sha224_update(CUDA_SHA224_CTX *ctx, const BYTE data[], size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            cuda_sha224_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha224_final(CUDA_SHA224_CTX *ctx, BYTE hash[])
{
    size_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        cuda_sha224_transform(ctx, ctx->data);
        for (i = 0; i < 56; i++) ctx->data[i] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;       ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16; ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32; ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48; ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha224_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
    }
}

__device__ void compute_sha224_cuda_single(char* word, int word_length, char* output_hash)
{
    CUDA_SHA224_CTX ctx;
    BYTE hash[SHA224_BLOCK_SIZE];
    
    cuda_sha224_init(&ctx);
    cuda_sha224_update(&ctx, (BYTE*)word, word_length);
    cuda_sha224_final(&ctx, hash);

    for (int i = 0; i < SHA224_BLOCK_SIZE; i++) {
        output_hash[i * 2]     = "0123456789abcdef"[hash[i] >> 4];
        output_hash[i * 2 + 1] = "0123456789abcdef"[hash[i] & 0x0F];
    }
    output_hash[56] = '\0';
}
#undef ROTRIGHT
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
#undef F
#undef G
#undef H
#undef I
#undef CH
#undef MAJ

// sha256

#define SHA256_BLOCK_SIZE 32
typedef unsigned char BYTE;
#ifndef WORD32_DEFINED
#define WORD32_DEFINED
typedef unsigned int WORD32;
#endif

typedef struct {
    BYTE data[64];
    WORD32 datalen;
    unsigned long long bitlen;
    WORD32 state[8];
} CUDA_SHA256_CTX;

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

__constant__ WORD32 k256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__device__ void cuda_sha256_transform(CUDA_SHA256_CTX *ctx, const BYTE data[])
{
    WORD32 a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
    for (; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k256[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

__device__ void cuda_sha256_init(CUDA_SHA256_CTX *ctx)
{
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667; ctx->state[1] = 0xbb67ae85; ctx->state[2] = 0x3c6ef372; ctx->state[3] = 0xa54ff53a;
    ctx->state[4] = 0x510e527f; ctx->state[5] = 0x9b05688c; ctx->state[6] = 0x1f83d9ab; ctx->state[7] = 0x5be0cd19;
}

__device__ void cuda_sha256_update(CUDA_SHA256_CTX *ctx, const BYTE data[], size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 64) {
            cuda_sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha256_final(CUDA_SHA256_CTX *ctx, BYTE hash[])
{
    size_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) ctx->data[i++] = 0x00;
        cuda_sha256_transform(ctx, ctx->data);
        for (i = 0; i < 56; i++) ctx->data[i] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = ctx->bitlen;       ctx->data[62] = ctx->bitlen >> 8;
    ctx->data[61] = ctx->bitlen >> 16; ctx->data[60] = ctx->bitlen >> 24;
    ctx->data[59] = ctx->bitlen >> 32; ctx->data[58] = ctx->bitlen >> 40;
    ctx->data[57] = ctx->bitlen >> 48; ctx->data[56] = ctx->bitlen >> 56;
    cuda_sha256_transform(ctx, ctx->data);

    for (i = 0; i < 4; ++i) {
        hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
        hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
    }
}

__device__ void compute_sha256_cuda_single(char* word, int word_length, char* output_hash)
{
    CUDA_SHA256_CTX ctx;
    BYTE hash[SHA256_BLOCK_SIZE];
    
    cuda_sha256_init(&ctx);

    cuda_sha256_update(&ctx, (BYTE*)word, word_length);
    
    cuda_sha256_final(&ctx, hash);

    for (int i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output_hash[i * 2]     = "0123456789abcdef"[hash[i] >> 4];
        output_hash[i * 2 + 1] = "0123456789abcdef"[hash[i] & 0x0F];
    }
    output_hash[64] = '\0';
}
#undef ROTRIGHT
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
#undef F
#undef G
#undef H
#undef I
#undef CH
#undef MAJ

// sha384

#define SHA384_BLOCK_SIZE 48
typedef unsigned char BYTE;
#ifndef WORD64_DEFINED
#define WORD64_DEFINED
typedef unsigned long long WORD64;  
#endif

typedef struct {
    BYTE data[128];
    WORD64 datalen;
    WORD64 bitlen;
    WORD64 state[8];
} CUDA_SHA384_CTX;

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (64-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,28) ^ ROTRIGHT(x,34) ^ ROTRIGHT(x,39))
#define EP1(x) (ROTRIGHT(x,14) ^ ROTRIGHT(x,18) ^ ROTRIGHT(x,41))
#define SIG0(x) (ROTRIGHT(x,1) ^ ROTRIGHT(x,8) ^ ((x) >> 7))
#define SIG1(x) (ROTRIGHT(x,19) ^ ROTRIGHT(x,61) ^ ((x) >> 6))

__constant__ WORD64 k384[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

__device__ void cuda_sha384_transform(CUDA_SHA384_CTX *ctx, const BYTE data[])
{
    WORD64 a, b, c, d, e, f, g, h, i, j, t1, t2, m[80];

    for (i = 0, j = 0; i < 16; ++i, j += 8) {
        m[i] = ((WORD64)data[j] << 56) | ((WORD64)data[j + 1] << 48) | ((WORD64)data[j + 2] << 40) | 
               ((WORD64)data[j + 3] << 32) | ((WORD64)data[j + 4] << 24) | ((WORD64)data[j + 5] << 16) |
               ((WORD64)data[j + 6] << 8) | ((WORD64)data[j + 7]);
    }
    for (; i < 80; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (i = 0; i < 80; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k384[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

__device__ void cuda_sha384_init(CUDA_SHA384_CTX *ctx)
{
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0xcbbb9d5dc1059ed8ULL; ctx->state[1] = 0x629a292a367cd507ULL;
    ctx->state[2] = 0x9159015a3070dd17ULL; ctx->state[3] = 0x152fecd8f70e5939ULL;
    ctx->state[4] = 0x67332667ffc00b31ULL; ctx->state[5] = 0x8eb44a8768581511ULL;
    ctx->state[6] = 0xdb0c2e0d64f98fa7ULL; ctx->state[7] = 0x47b5481dbefa4fa4ULL;
}

__device__ void cuda_sha384_update(CUDA_SHA384_CTX *ctx, const BYTE data[], size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 128) {
            cuda_sha384_transform(ctx, ctx->data);
            ctx->bitlen += 1024;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha384_final(CUDA_SHA384_CTX *ctx, BYTE hash[])
{
    size_t i = ctx->datalen;

    if (ctx->datalen < 112) {
        ctx->data[i++] = 0x80;
        while (i < 112) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 128) ctx->data[i++] = 0x00;
        cuda_sha384_transform(ctx, ctx->data);
        for (i = 0; i < 112; i++) ctx->data[i] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[127] = ctx->bitlen;       ctx->data[126] = ctx->bitlen >> 8;
    ctx->data[125] = ctx->bitlen >> 16; ctx->data[124] = ctx->bitlen >> 24;
    ctx->data[123] = ctx->bitlen >> 32; ctx->data[122] = ctx->bitlen >> 40;
    ctx->data[121] = ctx->bitlen >> 48; ctx->data[120] = ctx->bitlen >> 56;
    for (i = 119; i >= 112; i--) ctx->data[i] = 0;
    cuda_sha384_transform(ctx, ctx->data);

    for (i = 0; i < 8; ++i) {
        hash[i]      = (ctx->state[0] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 8]  = (ctx->state[1] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 16] = (ctx->state[2] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 24] = (ctx->state[3] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 32] = (ctx->state[4] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 40] = (ctx->state[5] >> (56 - i * 8)) & 0x00000000000000ffULL;
    }
}

__device__ void compute_sha384_cuda_single(char* word, int word_length, char* output_hash)
{
    CUDA_SHA384_CTX ctx;
    BYTE hash[SHA384_BLOCK_SIZE];
    
    cuda_sha384_init(&ctx);
    cuda_sha384_update(&ctx, (BYTE*)word, word_length);
    cuda_sha384_final(&ctx, hash);

    for (int i = 0; i < SHA384_BLOCK_SIZE; i++) {
        output_hash[i * 2]     = "0123456789abcdef"[hash[i] >> 4];
        output_hash[i * 2 + 1] = "0123456789abcdef"[hash[i] & 0x0F];
    }
    output_hash[96] = '\0';
}
#undef ROTRIGHT
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
#undef F
#undef G
#undef H
#undef I
#undef CH
#undef MAJ

// sha512

#define SHA512_BLOCK_SIZE 64
typedef unsigned char BYTE;
#ifndef WORD64_DEFINED
#define WORD64_DEFINED
typedef unsigned long long WORD64;
#endif

typedef struct {
    BYTE data[128];
    WORD64 datalen;
    WORD64 bitlen;
    WORD64 state[8];
} CUDA_SHA512_CTX;

#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (64-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,28) ^ ROTRIGHT(x,34) ^ ROTRIGHT(x,39))
#define EP1(x) (ROTRIGHT(x,14) ^ ROTRIGHT(x,18) ^ ROTRIGHT(x,41))
#define SIG0(x) (ROTRIGHT(x,1) ^ ROTRIGHT(x,8) ^ ((x) >> 7))
#define SIG1(x) (ROTRIGHT(x,19) ^ ROTRIGHT(x,61) ^ ((x) >> 6))

__constant__ WORD64 k[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

__device__ void cuda_sha512_transform(CUDA_SHA512_CTX *ctx, const BYTE data[])
{
    WORD64 a, b, c, d, e, f, g, h, i, j, t1, t2, m[80];

    for (i = 0, j = 0; i < 16; ++i, j += 8) {
        m[i] = ((WORD64)data[j] << 56) | ((WORD64)data[j + 1] << 48) | ((WORD64)data[j + 2] << 40) | 
               ((WORD64)data[j + 3] << 32) | ((WORD64)data[j + 4] << 24) | ((WORD64)data[j + 5] << 16) |
               ((WORD64)data[j + 6] << 8) | ((WORD64)data[j + 7]);
    }
    for (; i < 80; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

    a = ctx->state[0]; b = ctx->state[1]; c = ctx->state[2]; d = ctx->state[3];
    e = ctx->state[4]; f = ctx->state[5]; g = ctx->state[6]; h = ctx->state[7];

    for (i = 0; i < 80; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    ctx->state[0] += a; ctx->state[1] += b; ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f; ctx->state[6] += g; ctx->state[7] += h;
}

__device__ void cuda_sha512_init(CUDA_SHA512_CTX *ctx)
{
    ctx->datalen = 0; ctx->bitlen = 0;
    ctx->state[0] = 0x6a09e667f3bcc908ULL; ctx->state[1] = 0xbb67ae8584caa73bULL;
    ctx->state[2] = 0x3c6ef372fe94f82bULL; ctx->state[3] = 0xa54ff53a5f1d36f1ULL;
    ctx->state[4] = 0x510e527fade682d1ULL; ctx->state[5] = 0x9b05688c2b3e6c1fULL;
    ctx->state[6] = 0x1f83d9abfb41bd6bULL; ctx->state[7] = 0x5be0cd19137e2179ULL;
}

__device__ void cuda_sha512_update(CUDA_SHA512_CTX *ctx, const BYTE data[], size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen++;
        if (ctx->datalen == 128) {
            cuda_sha512_transform(ctx, ctx->data);
            ctx->bitlen += 1024;
            ctx->datalen = 0;
        }
    }
}

__device__ void cuda_sha512_final(CUDA_SHA512_CTX *ctx, BYTE hash[])
{
    size_t i = ctx->datalen;

    if (ctx->datalen < 112) {
        ctx->data[i++] = 0x80;
        while (i < 112) ctx->data[i++] = 0x00;
    } else {
        ctx->data[i++] = 0x80;
        while (i < 128) ctx->data[i++] = 0x00;
        cuda_sha512_transform(ctx, ctx->data);
        for (i = 0; i < 112; i++) ctx->data[i] = 0;
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[127] = ctx->bitlen;       ctx->data[126] = ctx->bitlen >> 8;
    ctx->data[125] = ctx->bitlen >> 16; ctx->data[124] = ctx->bitlen >> 24;
    ctx->data[123] = ctx->bitlen >> 32; ctx->data[122] = ctx->bitlen >> 40;
    ctx->data[121] = ctx->bitlen >> 48; ctx->data[120] = ctx->bitlen >> 56;
    for (i = 119; i >= 112; i--) ctx->data[i] = 0;
    cuda_sha512_transform(ctx, ctx->data);

    for (i = 0; i < 8; ++i) {
        hash[i]      = (ctx->state[0] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 8]  = (ctx->state[1] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 16] = (ctx->state[2] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 24] = (ctx->state[3] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 32] = (ctx->state[4] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 40] = (ctx->state[5] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 48] = (ctx->state[6] >> (56 - i * 8)) & 0x00000000000000ffULL;
        hash[i + 56] = (ctx->state[7] >> (56 - i * 8)) & 0x00000000000000ffULL;
    }
}
__device__ void compute_sha512_cuda_single(char* word, int word_length, char* output_hash)
{
    CUDA_SHA512_CTX ctx;
    BYTE hash[SHA512_BLOCK_SIZE];
    
    cuda_sha512_init(&ctx);
    
    cuda_sha512_update(&ctx, (BYTE*)word, word_length);
    
    cuda_sha512_final(&ctx, hash);

    for (int i = 0; i < SHA512_BLOCK_SIZE; i++) {
        output_hash[i * 2]     = "0123456789abcdef"[hash[i] >> 4];
        output_hash[i * 2 + 1] = "0123456789abcdef"[hash[i] & 0x0F];
    }
    output_hash[128] = '\0';
}
#undef ROTRIGHT
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
#undef F
#undef G
#undef H
#undef I
#undef CH
#undef MAJ

// -- code -- //
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

__device__ int global_match_count = 0; // used for output_file management

extern "C" __global__ void crack_word(char* wordlist, int* wordOffst, int* lenghts, char* hash,  unsigned char* hashtype, int tothashes, int n, bool verbose, bool nofile, char* output_file) {
    int i =  blockIdx.x * blockDim.x + threadIdx.x; // calculate the ID of the single thread
    if (i < n) {
        int indx = 0;
        char hash_word_cmp[129]; // size is 129 to accommodate the longest hash output (SHA-512)
        char* word = &wordlist[wordOffst[i]]; // get the word from the wordlist
        int inplen = lenghts[i]; // get the lenght of the word
    
        bool* match = new bool[tothashes]; // create the array of bool for the check
        int last_hashlen; // the last hash len, is used to avoid unnecessary recalculations
        int hashlen = 1; // set to 1 so it's different in the first iteration
        for (int k = 0; k < tothashes; k++) {
            switch (hashtype[k]) {
                case 0:
                    // md5 
                    hashlen = 32;
                    // conversion
                    if (last_hashlen != hashlen) // recompute the hash only if the length has changed
                        compute_md5_cuda_single(word, inplen, hash_word_cmp);
                    break;
                case 1:
                    // sha-1
                    hashlen = 40;
                    // conversion
                    if (last_hashlen != hashlen)
                        compute_sha1_cuda_single(word, inplen, hash_word_cmp);
                    break;
                case 2:
                    //  sha-224
                    hashlen = 56;
                    // conversion
                    if (last_hashlen != hashlen)
                        compute_sha224_cuda_single(word, inplen, hash_word_cmp);
                    break;
                case 3:
                    //  sha-256
                    hashlen = 64;
                    // conversion
                    if (last_hashlen != hashlen)
                        compute_sha256_cuda_single(word, inplen, hash_word_cmp);
                    break;
                case 4:
                    // sha-384
                    hashlen = 96;
                    // conversion
                    if (last_hashlen != hashlen)
                        compute_sha384_cuda_single(word, inplen, hash_word_cmp);
                    break;
                case 5:
                    // sha-512
                    hashlen = 128;
                    // conversion
                    if (last_hashlen != hashlen)
                        compute_sha512_cuda_single(word, inplen, hash_word_cmp);
                    break;
            }
            last_hashlen = hashlen;
            char* nHash = &hash[indx]; // copy the hash to inspect
            match[k] = true;
            for (int h = 0; h < hashlen; h++){
                if (hash_word_cmp[h] != nHash[h]){ // if there is an inconsistency, the match fails
                    match[k] = false;
                    break;
                }
            }
            if (match[k]) {
                printf("FOUND MATCH: '%s' -> '%s'\n", hash_word_cmp, word);
                if (!nofile) { // if not disabled, insert the entries in the buffer, in the format: <\0hash\0word\0>
                    int matchID = atomicAdd(&global_match_count, 1);
                    int rec_size = hashlen + inplen + 4;
                    int offset = matchID * rec_size;
                    output_file[offset] = '\0';

                    for (int l = 0; l < hashlen && hash_word_cmp[l] != '\0'; l++){
                        output_file[offset+1+l] = hash_word_cmp[l];
                    }

                    output_file[(offset + hashlen + 1)] = '\0';

                    for (int l = 0; l < inplen && word[l] != '\0'; l++){
                        output_file[offset+1+hashlen+1+l] = word[l];
                    }
                    output_file[(offset+1+hashlen+1+inplen)] = '\0';
                }
            }
            else if (!match[k] && verbose) {
                printf("dont work: '%s'\n", hash_word_cmp);
            }
            indx += hashlen;
        }
        delete[] match;

    }
    
}
