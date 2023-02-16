package net.floodlightcontroller.core.internal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;
import org.projectfloodlight.openflow.protocol.OFFactories;
import org.projectfloodlight.openflow.protocol.OFFactory;
import org.projectfloodlight.openflow.protocol.OFMessage;
import org.projectfloodlight.openflow.protocol.OFMessageReader;
import org.projectfloodlight.openflow.protocol.OFVersion;
public class OFMessageDecoder extends ByteToMessageDecoder {
	private OFMessageReader<OFMessage> reader;
	public OFMessageDecoder() {
		setReader();
	}
	public OFMessageDecoder(OFVersion version) {
		setVersion(version);
		setReader();
	}
	private void setReader() {
		reader = OFFactories.getGenericReader();
	}
	public void setVersion(OFVersion version) {
		OFFactory factory = OFFactories.getFactory(version);
		this.reader = factory.getReader();
	}
	@Override
	protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) throws Exception {
		if (!ctx.channel().isActive()) {
			return;
		}
		OFMessage singleMessage = null;
		List<OFMessage> list = null;
		boolean first = true;
		for (;;) {
			OFMessage message = reader.readFrom(in);
			if (message == null) {
				break;
			}
			if (first) {
				singleMessage = message;
				first = false;
			} else {
				if (list == null) {
					list = new ArrayList<>();
					list.add(singleMessage);
					singleMessage = null;
				}
				list.add(message);
			}
		}
		if (list != null) {
			out.add(list);
		} else if (singleMessage != null) {
			out.add(Collections.singletonList(singleMessage));
		}
	}
}