package net.floodlightcontroller.core.util;
import java.util.ArrayList;
import java.util.List;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufInputStream;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.LengthFieldBasedFrameDecoder;
import org.apache.thrift.TBase;
import org.apache.thrift.protocol.TCompactProtocol;
import org.apache.thrift.transport.TIOStreamTransport;
public abstract class ThriftFrameDecoder<T extends TBase<?,?>> extends LengthFieldBasedFrameDecoder {
    public ThriftFrameDecoder(int maxSize) {
        super(maxSize, 0, 4, 0, 4);
    }
    protected abstract T allocateMessage();
    @Override
    protected final Object decode(ChannelHandlerContext ctx,
                            ByteBuf buffer) throws Exception {
        List<T> ms = null;
        ByteBuf frame = null;
        while (null != (frame = (ByteBuf) super.decode(ctx, buffer))) {
            if (ms == null) ms = new ArrayList<T>();
            ByteBufInputStream is = new ByteBufInputStream(frame);
            TCompactProtocol thriftProtocol =
                    new TCompactProtocol(new TIOStreamTransport(is));
            T message = allocateMessage();
            message.read(thriftProtocol);
            ms.add(message);
        }
        return ms;
    }
    @Override
    protected final ByteBuf extractFrame(ChannelHandlerContext ctx, ByteBuf buffer, int index,
            int length) {
        return buffer.slice(index, length);
    }
}