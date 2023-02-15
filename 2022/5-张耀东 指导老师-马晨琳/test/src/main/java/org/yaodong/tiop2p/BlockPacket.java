package org.yaodong.tiop2p;

import org.tio.core.intf.Packet;

/**
 * 区块链底层定制的packet
 * @author yaodong
 */
public class BlockPacket extends Packet {
    // 1. 网络传输需要序列化，这里采用java自带的序列化方式
    private static final long serialVersionUID = -172060606924066412L;

    // 2. 消息头的长度
    public static final int HEADER_LENGTH = 4;

    // 3. 字节编码类型
    public static final String CHARSET = "utf-8";

    // 4. 传输内容的字节
    private byte[] body;

    public byte[] getBody() {
        return body;
    }

    public void setBody(byte[] body) {
        this.body = body;
    }
}
