package org.yaodong.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
/**
 * Merkle树构建、生成根节点哈希值的工具类
 * @author yaodong
 */
public class MerkleTree {
    // 1. TreeNode list
    private List<TreeNode> list;
    // 2. merkle根节点
    private TreeNode root;

    public MerkleTree(List<String> contents){
        createMerkleTree(contents);
    }
    public void createMerkleTree(List<String> contents){
        //判空
        if(contents == null || contents.size() == 0)
            return;

        // 1. 初始化
        list = new ArrayList<>();

        // 2. 根据数据创建叶子节点
        List<TreeNode> leafList = createLeafList(contents);
        list.addAll(leafList);

        // 3. 创建父节点
        List<TreeNode> parents = createParentList(leafList);
        list.addAll(parents);

        // 4. 循环创建各级父节点直至根节点
        while (parents.size() > 1){
            List<TreeNode> temp = createParentList(parents);
            list.addAll(temp);
            parents = temp;
        }

        // 5. 将迭代完的根节点返回root
        root = parents.get(0);
    }

    /**
     * 创建父节点列表
     * @param leafList
     * @return
     */
    private List<TreeNode> createParentList(List<TreeNode> leafList){
        List<TreeNode> parents = new ArrayList<>();

        //判空
        if(leafList == null || leafList.size() == 0)
            return parents;

        int len = leafList.size();
        for(int i=0; i < len-1; i+=2){
            TreeNode parent = createParentNode(leafList.get(i), leafList.get(i+1));
            parents.add(parent);
        }

        // 奇数个节点时，单独处理最后一个节点
        if(len % 2 != 0){
            TreeNode parent = createParentNode(leafList.get(len-1), null);
            parents.add(parent);
        }
        return  parents;
    }

    /**
     * 创建父节点
     * @param left
     * @param right
     * @return
     */
    private TreeNode createParentNode(TreeNode left, TreeNode right){
        TreeNode parent = new TreeNode();

        parent.setLeft(left);
        parent.setRight(right);

        // 如果right为空，则父节点的hash值为left的hash值
        String hash = left.getHash();
        if(right != null)
            hash = SHAUtil.sha256BasedHutool(left.getHash() + right.getHash());
        //hash字段和data字段的值一样
        parent.setData(hash);
        parent.setHash(hash);

        if(right != null)
            parent.setName("节点" + left.getName() + "和" + right.getName() + "的父节点");
        else
            parent.setName("继承节点" + left.getName() + "成为父节点");
        return parent;
    }

    private List<TreeNode> createLeafList(List<String> contents){
        List<TreeNode> leafList = new ArrayList<>();

        //判空
        if(contents == null || contents.size() == 0)
            return leafList;

        //根据data创建叶节点
        for (String content : contents) {
            TreeNode node = new TreeNode(content);
            leafList.add(node);
        }
        return leafList;
    }
    public void traverseTreeNode(){
        Collections.reverse(list);
        TreeNode root = list.get(0);
        traverseTreeNodes(root);
    }
    public void traverseTreeNodes(TreeNode node){
        System.out.println(node.getName());
        if(node.getLeft() != null)
            traverseTreeNodes(node.getLeft());
        if(node.getRight() != null)
            traverseTreeNodes(node.getRight());
    }

    public List<TreeNode> getList() {
        if(list == null)
            return list;
        Collections.reverse(list);
        return list;
    }

    public void setList(List<TreeNode> list) {
        this.list = list;
    }

    public TreeNode getRoot() {
        return root;
    }

    public void setRoot(TreeNode root) {
        this.root = root;
    }
}
