package org.yaodong.blockchain;

import cn.hutool.crypto.digest.DigestUtil;

/**
 * 区块
 * @author yaodong
 */
public class Block {
    // 1. 区块头
    private BlockHeader blockHeader;
    // 2. 区块体
    private BlockBody blockBody;
    // 3. 区块hash
    private String blockHash;

    public BlockHeader getBlockHeader() {
        return blockHeader;
    }

    public void setBlockHeader(BlockHeader blockHeader) {
        this.blockHeader = blockHeader;
    }

    public BlockBody getBlockBody() {
        return blockBody;
    }

    public void setBlockBody(BlockBody blockBody) {
        this.blockBody = blockBody;
    }

    //根据该区块的所有属性计算sha256
    public String getBlockHash() {
        return DigestUtil.sha256Hex(blockHeader.toString()+blockBody.toString());
    }

    public void setBlockHash(String blockHash) {
        this.blockHash = blockHash;
    }
}
