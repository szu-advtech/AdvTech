package org.yaodong.p2ppbft;

import java.util.List;

public class VoteInfo {
    // 1. 投票状态码
    private int code;

    // 2. 待写入块的内容
    private List<String> list;

    private String hash;

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }

    public List<String> getList() {
        return list;
    }

    public void setList(List<String> list) {
        this.list = list;
    }

    public String getHash() {
        return hash;
    }

    public void setHash(String hash) {
        this.hash = hash;
    }
}
