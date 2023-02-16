package org.yaodong.util;

import cn.hutool.crypto.SecureUtil;
import cn.hutool.crypto.asymmetric.KeyType;
import cn.hutool.crypto.asymmetric.RSA;
import java.security.PrivateKey;
import java.security.PublicKey;

import cn.hutool.crypto.symmetric.DES;
import org.testng.util.Strings;

/**
 * 基于Hutool工具类的加密解密类
 *
 * @author yaodong
 */
public class DeEnCoderHutoolUtil {
    //创建私钥
    private static RSA rsa = new RSA();
    //获得私钥
    private static PrivateKey privateKey = rsa.getPrivateKey();
    //获得公钥
    private static PublicKey publicKey = rsa.getPublicKey();

    /**
     * function RSA加密通用方法：对称加密解密
     * @param originalContent：明文
     * @return 密文
     */
    public static String rsaEncrypt(String originalContent){
        // 当明文为空时
        if(Strings.isNullOrEmpty(originalContent))
            return null;
        //公钥加密
        return rsa.encryptBase64(originalContent, KeyType.PublicKey);
    }

    /**
     * function RSA解密通用方法
     * @param cipherContent 密文
     * @return 明文
     */
    public static String rsaDecrypt(String cipherContent){
        // 当密文为空时
        if(Strings.isNullOrEmpty(cipherContent))
            return null;
        // 私钥解密
        return rsa.decryptStr(cipherContent, KeyType.PrivateKey);
    }

    /**
     * function: DES 加密通用方法：对称加密
     * @param originalContent 明文
     * @param key 密钥
     * @return 密文
     */
    public static String desEncript(String originalContent, String key){
        // 1.当明文或加密密钥为空时
        if(Strings.isNullOrEmpty(originalContent)||Strings.isNullOrEmpty(key))
            return null;

        // 2.构建密文
        DES des = SecureUtil.des(key.getBytes());

        // 3.加密
        return des.encryptHex(originalContent);
    }

    /**
     * DES解密通用方法：对称解密
     * @param cipherContent 密文
     * @param key 密钥
     * @return 明文
     */
    public static String desDecript(String cipherContent, String key){
        // 1.当密文或密钥为空时
        if(Strings.isNullOrEmpty(cipherContent)||Strings.isNullOrEmpty(key))
            return null;
        // 2. 构建
        DES des = SecureUtil.des(key.getBytes());
        // 3. 解密
        return des.decryptStr(cipherContent);
    }
}
