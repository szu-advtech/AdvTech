package net.floodlightcontroller.core.internal;
import org.projectfloodlight.openflow.protocol.OFMessage;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToByteEncoder;
public class OFMessageEncoder extends MessageToByteEncoder<Iterable<OFMessage>> {
    @Override
    protected void encode(ChannelHandlerContext ctx, Iterable<OFMessage> msgList, ByteBuf out) throws Exception {
        for (OFMessage ofm :  msgList) {
            ofm.writeTo(out);
        }
    }
}