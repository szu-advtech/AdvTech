package org.yaodong.tiop2p;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tio.client.intf.ClientAioHandler;
import org.tio.core.ChannelContext;
import org.tio.core.GroupContext;
import org.tio.core.exception.AioDecodeException;
import org.tio.core.intf.Packet;

import java.nio.ByteBuffer;

/**
 * 基于t-io的区块链底层p2p网络平台的客户端Handler
 * @author yaodong
 */
public class BlockChainClientAioHandler implements ClientAioHandler {
    // 1. 日志记录
    private Logger logger = LoggerFactory.getLogger(BlockChainClientAioHandler.class);

    private static BlockPacket heartbeatPacket = new BlockPacket();

    @Override
    public BlockPacket heartbeatPacket() {
        return heartbeatPacket;
    }

    @Override
    public Packet decode(ByteBuffer byteBuffer, int limit, int position, int readableLength,
                         ChannelContext channelContext) throws AioDecodeException {
        // 收到的数据组不了包则返回null
        if(readableLength < BlockPacket.HEADER_LENGTH) return  null;

        // 读取消息体的长度
        int bodyLength = byteBuffer.getInt();

        // 数据异常，抛出解码异常
        if(bodyLength < 0)
            throw new AioDecodeException( "bodyLength [" + bodyLength + "] is not right, remote:" + channelContext.getClientNode());
        int neededLength = BlockPacket.HEADER_LENGTH + bodyLength;
        int isDataEnough = readableLength - neededLength;
        if(isDataEnough < 0) return  null;
        else{
            BlockPacket packet = new BlockPacket();
            if(bodyLength > 0){
                byte[] dst = new byte[bodyLength];
                byteBuffer.get(dst);
                packet.setBody(dst);
            }
            return packet;
        }
    }

    @Override
    public ByteBuffer encode(Packet packet, GroupContext groupContext, ChannelContext channelContext) {
        // 多态：将应用层的packet强转为我们的区块链packet
        BlockPacket blockPacket = (BlockPacket) packet;

        // 拿到消息体，并判空
        byte[] body = blockPacket.getBody();
        int bodyLen = 0;
        if(body != null)
            bodyLen = body.length;
        //bodyBuffer的总长度 = 消息头长度+消息体长度
        int totalLen = BlockPacket.HEADER_LENGTH + bodyLen;
        //创建一个新的byteBuffer
        ByteBuffer buffer = ByteBuffer.allocate(totalLen);
        //设置字节序
        buffer.order(groupContext.getByteOrder());
        //写消息头
        buffer.putInt(bodyLen);
        //写消息体
        if(body != null)
            buffer.put(body);
        return buffer;
    }

    @Override
    public void handler(Packet packet, ChannelContext channelContext) throws Exception {
        BlockPacket blockPacket = (BlockPacket) packet;
        byte[] body = blockPacket.getBody();
        if(body != null){
            String str = new String(body, BlockPacket.CHARSET);
            logger.info("512客户端收到消息："+str);
        }
        return;
    }
}
