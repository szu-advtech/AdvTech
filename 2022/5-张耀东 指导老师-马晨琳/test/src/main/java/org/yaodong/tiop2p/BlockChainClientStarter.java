package org.yaodong.tiop2p;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.tio.client.ClientChannelContext;
import org.tio.client.ClientGroupContext;
import org.tio.client.ReconnConf;
import org.tio.client.TioClient;
import org.tio.client.intf.ClientAioHandler;
import org.tio.client.intf.ClientAioListener;
import org.tio.core.Node;
import org.tio.core.Tio;

import javax.annotation.PostConstruct;
import java.io.IOException;

/**
 * 基于t-io的区块链底层p2p网络平台客户端
 * @author yaodong
 */
@Component
public class BlockChainClientStarter {
    // 1. 日志记录
    private Logger logger = LoggerFactory.getLogger(BlockChainClientStarter.class);

    // 2. 服务器节点
    private Node serverNode;

    // 3. handler，包括编码、解码、消息处理
    private ClientAioHandler clientAioHandler;

    // 4. 事件监听器，可以为null，但建议实现该接口，
    private ClientAioListener aioListener = null;

    // 5. 断链后自动连接，不想自动连接则设为null
    private ReconnConf reconnConf = new ReconnConf(5000L);

    //6. 一组连接共用的上下文对象
    private ClientGroupContext clientGroupContext;

    private TioClient tioClient = null;

    private ClientChannelContext clientChannelContext = null;

    /**
     * 启动程序入口
     */
//    @PostConstruct
//    @Order(2)
    public void start(){
        try {
            logger.info("512客户端即将启动");

            serverNode = new Node(TioConst.SERVER, TioConst.PORT);
            clientAioHandler = new BlockChainClientAioHandler();
            clientGroupContext = new ClientGroupContext(clientAioHandler, aioListener, reconnConf);

            clientGroupContext.setHeartbeatTimeout(TioConst.TIMEOUT);
            tioClient = new TioClient(clientGroupContext);
            clientChannelContext = tioClient.connect(serverNode);

            //连上后，发条消息测试
            sendMessage();

            logger.info("512客户端启动完毕");
        }catch (Exception e) {
            logger.error(e.getMessage());
        }
    }
    private void sendMessage() throws Exception {
        BlockPacket packet = new BlockPacket();
        packet.setBody(("tal say hello world to blockchain!").getBytes(BlockPacket.CHARSET));
        Tio.send(clientChannelContext, packet);
    }
}
