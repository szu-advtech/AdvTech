package org.yaodong.blockchain;

import java.util.List;

/**
 * 区块体，存放交易数组
 * @author yaodong
 */
public class BlockBody {
    private List<ContentInfo> contentInfos;

    public List<ContentInfo> getContentInfos() {
        return contentInfos;
    }

    public void setContentInfos(List<ContentInfo> contentInfos) {
        this.contentInfos = contentInfos;
    }
}
