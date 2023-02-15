package org.yaodong.p2ppbft;

public enum VoteEnum {
    PREPREPARE("节点将自己生成Block", 100),
    PREPARE("节点收到请求生成block的消息，进入准备状态，并对外广播该状态",
            200),
    COMMIT("每个节点收到超过2f+1个不同节点的commit消息后，则认为该区块已经达成一致，即进入committed状态，并将其持久化到区块链数据库中", 400);

    VoteEnum(String msg, int code) {
        this.msg = msg;
        this.code = code;
    }
    // 1. 投票情况描述
    private String msg;
    // 2. 投票情况状态码
    private int code;
    public static VoteEnum find(int code){
        for(VoteEnum ve: VoteEnum.values())
            if(ve.code == code)
                return ve;
        return null;
    }

    public String getMsg() {
        return msg;
    }

    public void setMsg(String msg) {
        this.msg = msg;
    }

    public int getCode() {
        return code;
    }

    public void setCode(int code) {
        this.code = code;
    }
}
