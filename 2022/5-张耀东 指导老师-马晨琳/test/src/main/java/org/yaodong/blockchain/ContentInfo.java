package org.yaodong.blockchain;

/**
 * 区块体内一条内容实体
 * @author yaodong
 * jsonContent -- 新的Json内容
 * timeStamp -- 时间戳
 * publicKey -- 公钥
 * sign-- 签名
 * hash -- 该操作的hash
 */
public class ContentInfo {
    private String jsonContent;
    private Long timeStamp;
    private String publicKey;
    private String sign;
    private String hash;

    public String getJsonContent() {
        return jsonContent;
    }

    public void setJsonContent(String jsonContent) {
        this.jsonContent = jsonContent;
    }

    public Long getTimeStamp() {
        return timeStamp;
    }

    public void setTimeStamp(Long timeStamp) {
        this.timeStamp = timeStamp;
    }

    public String getPublicKey() {
        return publicKey;
    }

    public void setPublicKey(String publicKey) {
        this.publicKey = publicKey;
    }

    public String getSign() {
        return sign;
    }

    public void setSign(String sign) {
        this.sign = sign;
    }

    public String getHash() {
        return hash;
    }

    public void setHash(String hash) {
        this.hash = hash;
    }
}
