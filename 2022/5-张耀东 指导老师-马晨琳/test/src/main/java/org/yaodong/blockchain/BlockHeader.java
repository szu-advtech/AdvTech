package org.yaodong.blockchain;

import com.google.common.hash.BloomFilter;

import java.util.List;

/**
 * 区块头
 * @author yaodong
 * version -- 版本号
 * hashPreviousBlock -- 前一个区块的hash
 * hashMerkleRoot -- merkle根节点的hash
 * publicKey -- 生成该区块的公钥
 * number -- 区块的序号
 * timeStamp -- 时间戳
 * nonce -- 32位随机数
 * hashList -- 该区块里每条交易信息的hash集合，按顺序来的，通过该hash集合能算出根节点hash
 */
public class BlockHeader {
    private int version;
    private String hashPreviousBlock;
    private String hashMerkleRoot;
    private String publicKey;
    private int number;
    private long timeStamp;
    private long nonce;
    private List<String> hashList;
    private boolean isFullNode;

    private BloomFilter bloomFilter;

    public BloomFilter getBloomFilter() {
        return bloomFilter;
    }

    public void setBloomFilter(BloomFilter bloomFilter) {
        this.bloomFilter = bloomFilter;
    }

    public boolean isFullNode() {
        return isFullNode;
    }

    public void setFullNode(boolean fullNode) {
        isFullNode = fullNode;
    }

    public int getVersion() {
        return version;
    }

    public void setVersion(int version) {
        this.version = version;
    }

    public String getHashPreviousBlock() {
        return hashPreviousBlock;
    }

    public void setHashPreviousBlock(String hashPreviousBlock) {
        this.hashPreviousBlock = hashPreviousBlock;
    }

    public String getHashMerkleRoot() {
        return hashMerkleRoot;
    }

    public void setHashMerkleRoot(String hashMerkleRoot) {
        this.hashMerkleRoot = hashMerkleRoot;
    }

    public String getPublicKey() {
        return publicKey;
    }

    public void setPublicKey(String publicKey) {
        this.publicKey = publicKey;
    }

    public int getNumber() {
        return number;
    }

    public void setNumber(int number) {
        this.number = number;
    }

    public long getTimeStamp() {
        return timeStamp;
    }

    public void setTimeStamp(long timeStamp) {
        this.timeStamp = timeStamp;
    }

    public long getNonce() {
        return nonce;
    }

    public void setNonce(long nonce) {
        this.nonce = nonce;
    }

    public List<String> getHashList() {
        return hashList;
    }

    public void setHashList(List<String> hashList) {
        this.hashList = hashList;
    }
}
