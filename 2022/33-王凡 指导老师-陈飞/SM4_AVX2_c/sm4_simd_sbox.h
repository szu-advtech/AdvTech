#pragma once
// sm4_simd_sbox.h

#ifndef SM4_SIMD_SBOX_H
#define SM4_SIMD_SBOX_H

#include <stdint.h>

/**
 * @brief SM4 ¬÷√‹‘ø
 */
typedef uint32_t* SM4_Key;


int SM4_KeyInit(uint8_t* key, SM4_Key* sm4_key);


void SM4_Encrypt_x8(uint8_t* plaintext, uint8_t* ciphertext, SM4_Key sm4_key);


void SM4_Decrypt_x8(uint8_t* ciphertext, uint8_t* plaintext, SM4_Key sm4_key);

void SM4_KeyDelete(SM4_Key sm4_key);

#endif
