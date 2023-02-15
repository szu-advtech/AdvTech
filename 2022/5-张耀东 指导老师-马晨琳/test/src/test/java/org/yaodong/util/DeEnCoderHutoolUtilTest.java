package org.yaodong.util;

import cn.hutool.crypto.SecureUtil;
import cn.hutool.crypto.symmetric.SymmetricAlgorithm;
import org.junit.Test;

/**
 * 基于Hutool工具类的加密解密类的单元测试
 * @author yaodong
 */
public class DeEnCoderHutoolUtilTest {
    @Test
    public void testDesEncrypt(){
        String originalContent = "hello world",
                key = new String(SecureUtil.generateKey(SymmetricAlgorithm.DES.getValue()).getEncoded());
        String s = DeEnCoderHutoolUtil.desEncript(originalContent, key);
        System.out.println(s);
    }

    @Test
    public void testDesDecrypt(){
        String originalContent = "hello world",
                key = new String(SecureUtil.generateKey(SymmetricAlgorithm.DES.getValue()).getEncoded());
//        System.out.printf("key: %s\n", key);
        String cipherContent = DeEnCoderHutoolUtil.desEncript(originalContent, key);
        System.out.printf("密文：%s\n",cipherContent);
        String s = DeEnCoderHutoolUtil.desDecript(cipherContent, key);
        System.out.printf("明文：%s\n", s);
    }
}
