package org.yaodong.util;

import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
/**
 * MerkleTree测试
 * @author yaodong
 */
public class MerkleTreeTest {
    @Test
    public void testMerkleTree(){
        List<String>  contents = null;
        contents = Arrays.asList("hello", "world", "嘿嘿", "磁悬浮","五个核桃");
        String hash = new MerkleTree(contents).getRoot().getHash();
        System.out.println("hash: " + hash);
//        String name = new MerkleTree(contents).getRoot().getName();
//        System.out.println(name);

        new MerkleTree(contents).traverseTreeNode();
    }
}
