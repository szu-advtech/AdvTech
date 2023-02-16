package org.yaodong.util;

import java.util.ArrayList;
import java.util.List;

/**
 * 简化的Merkle树根节点hash计算
 * @author : yaodong
 */
public class MerkleTreeUtil {

    /**
     * function：按照Merkle树思想计算根节点的值
     * @param hashList：Merkle树节点列表
     * @return Merkle根节点的hash值
     */
    public static String getTreeNodeHash(List<String> hashList){
        //判空
        if(hashList == null || hashList.size() == 0)
            return null;
        while(hashList.size() != 1)
            hashList = getMerkleNodeList(hashList);
        return hashList.get(0);
    }

    /**
     * 计算根节点hash值
     * @param contentList 所有节点的hash值列表
     * @return 最新的Merkle树列表
     */
    public static List<String> getMerkleNodeList(List<String> contentList){
        List<String> merkleNodeList = new ArrayList<>();
        //判空
        if(contentList == null || contentList.size() == 0)
            return merkleNodeList;

        //index：索引位置
        int index = 0, length = contentList.size();

        while(index < length){
            //获取左孩子节点数据
            String left = contentList.get(index++);

            //获取右孩子节点数据
            String right = "";
            if(index < length)
                right = contentList.get(index++);

            //计算左右孩子节点的父节点的hash值
            String sha2HexValue = SHAUtil.sha256BasedHutool(left + right);
            merkleNodeList.add(sha2HexValue);
        }
        return merkleNodeList;
    }

}
