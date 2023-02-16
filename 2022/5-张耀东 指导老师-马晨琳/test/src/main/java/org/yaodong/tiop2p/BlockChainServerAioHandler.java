package org.yaodong.tiop2p;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tio.core.ChannelContext;
import org.tio.core.GroupContext;
import org.tio.core.Tio;
import org.tio.core.exception.AioDecodeException;
import org.tio.core.intf.Packet;
import org.tio.server.intf.ServerAioHandler;

import java.nio.ByteBuffer;

/**
 * 基于t-io的区块链底层p2p网络层平台的服务器Handler
 * @author yaodong
 */
public class BlockChainServerAioHandler implements ServerAioHandler {

    //1. 日志记录
    private Logger logger = LoggerFactory.getLogger(BlockChainServerAioHandler.class);

    /**
     * 解码：把接收到的ByteBuffer，解码成应用可以识别的业务消息包
     * 总体消息结构：消息头+消息体
     * 消息头结构：4字节，存储消息体的长度
     * 消息体结构：对象的json串的byte[]
     *
     * @param byteBuffer -- 消息体
     * @param i -- limit
     * @param i1 -- start position
     * @param readableLength -- readableLength
     * @param channelContext --
     * @return
     * @throws AioDecodeException
     */
    @Override
    public Packet decode(ByteBuffer byteBuffer, int i, int i1, int readableLength, ChannelContext channelContext) throws AioDecodeException {
        //判空，如果收到的数据组不了业务包（<4字节），则返回null告诉框架 数据不够
        if(readableLength < BlockPacket.HEADER_LENGTH) return null;

        // 读取消息体的长度
        int bodyLength = byteBuffer.getInt();

        // 数据不正确，则抛出解码异常
        if(bodyLength < 0)
            throw new AioDecodeException("bodyLength [" + bodyLength + "] is not right, remote:" + channelContext.getClientNode());

        // 计算本次需要的数据长度
        int neededLength = BlockPacket.HEADER_LENGTH + bodyLength;

        // 收到的数据是否最够组包
        int isDataEnough = readableLength - neededLength;

        //不够消息体长度（剩下的buffer组不了消息体）
        if(isDataEnough < 0)
            return null;
        else{
            //组包成功
            BlockPacket packet = new BlockPacket();
            if(bodyLength > 0){
                byte[] dst = new byte[bodyLength];
                byteBuffer.get(dst);
                packet.setBody(dst);
            }
            return packet;
        }
    }

    /**
     * 编码：把业务消息打包成可以发送的byteBuffer
     * 消息结构依然是：消息头 + 消息体
     * body header:     4 字节，存储消息体的长度
     * body :           对象的json串的byte[]
     * @param packet -- 应用层的数据包
     * @param groupContext -- 配置线程池、确定监听端口、维护客户端的各种数据
     * @param channelContext -- 通道上下文相关的类
     * @return 打包好的ByteBuffer
     */
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

    /**
     * 处理消息
     * @param packet
     * @param channelContext
     * @throws Exception
     */
    @Override
    public void handler(Packet packet, ChannelContext channelContext) throws Exception {
        // 强转
        BlockPacket blockPacket = (BlockPacket) packet;
        // 拿到消息体
        byte[] body = blockPacket.getBody();
        if(body != null){
            //将其转化为string，设置消息体
            String str = new String(body, BlockPacket.CHARSET);
            logger.info("512服务端收到消息：" + str);
            BlockPacket responsePacket = new BlockPacket();
            responsePacket.setBody(("512服务端已收到，你的消息是："+str).getBytes(BlockPacket.CHARSET));
            //给客户端发送回复消息
            Tio.send(channelContext, responsePacket);
        }
        return;
    }
}
