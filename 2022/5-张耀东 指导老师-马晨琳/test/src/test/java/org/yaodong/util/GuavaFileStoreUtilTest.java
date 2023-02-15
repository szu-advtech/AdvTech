package org.yaodong.util;

import org.junit.Test;

public class GuavaFileStoreUtilTest {
    @Test
    public void test(){
        GuavaFileStoreUtil.writeIntoBlockFile();
    }

    @Test
    public void test1(){
        String content = "123\n";
        String targetFileName = ".//blockchain-" + System.currentTimeMillis() + ".log";
        GuavaFileStoreUtil.appendToTargetFile(targetFileName, content);
        content = "456\n";
        GuavaFileStoreUtil.appendToTargetFileByGuava(targetFileName, content);
    }
}
