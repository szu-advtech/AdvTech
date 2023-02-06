package org.yaodong.util;

import org.junit.Test;

public class SHAUtilTest {

    @Test
    public void testGetSHA256(){
        String originalString = "你好，世界";
        String encryptString = SHAUtil.getSHA256BasedMD(originalString);
        System.out.println(encryptString);
    }

    @Test
    public void testSha256BasedHutool(){
        String originalString = "hello world";
        String encryptString = SHAUtil.sha256BasedHutool(originalString);
        System.out.println(encryptString);
    }
}
