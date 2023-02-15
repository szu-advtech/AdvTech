package org.yaodong.util;

/**
 * 定义二叉树
 * @author yaodong
 */

public class TreeNode {
    // 1. 二叉树左节点
    private TreeNode left;

    // 2. 二叉树右节点
    private TreeNode right;

    // 3. 节点中的数据
    private String data;

    // 4. 节点数据值对应的hash值
    private String hash;

    // 5. 节点名称
    private String name;

    public TreeNode() {
    }

    public TreeNode(String data) {
        this.data = data;
        this.hash = SHAUtil.sha256BasedHutool(data);
        this.name = "[节点: " + data + "]";
    }

    public TreeNode getLeft() {
        return left;
    }

    public void setLeft(TreeNode left) {
        this.left = left;
    }

    public TreeNode getRight() {
        return right;
    }

    public void setRight(TreeNode right) {
        this.right = right;
    }

    public String getData() {
        return data;
    }

    public void setData(String data) {
        this.data = data;
    }

    public String getHash() {
        return hash;
    }

    public void setHash(String hash) {
        this.hash = hash;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
