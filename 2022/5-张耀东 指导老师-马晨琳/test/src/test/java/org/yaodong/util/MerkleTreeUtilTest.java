package org.yaodong.util;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

public class MerkleTreeUtilTest {
    @Test
    public void testGetMerkleNodeList(){
        List<String> contentList = null;
        contentList = Arrays.asList("hello", "world", "嘿","大宝");
        List<String> list = MerkleTreeUtil.getMerkleNodeList(contentList);
        System.out.println(list.size());
        System.out.println(list);
    }
    @Test
    public void testGetTreeNodeHash(){
        List<String> contentList = null;
        contentList = Arrays.asList("hello","world", "嘿");
        String treeNodeHash = MerkleTreeUtil.getTreeNodeHash(contentList);
        System.out.println(treeNodeHash);
    }
}
