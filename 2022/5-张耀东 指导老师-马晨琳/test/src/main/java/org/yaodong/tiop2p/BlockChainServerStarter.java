package org.yaodong.tiop2p;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.tio.server.ServerGroupContext;
import org.tio.server.TioServer;
import org.tio.server.intf.ServerAioHandler;
import org.tio.server.intf.ServerAioListener;

import javax.annotation.PostConstruct;
import java.io.IOException;

/**
 * 基于t-io的区块链底层p2p网络平台的服务器
 * @author  yaodong
 */
@Component
public class BlockChainServerStarter {
    // 1. 日志记录
    private Logger logger = LoggerFactory.getLogger(BlockChainServerStarter.class);

    // 2. handler，包括编码、解码、消息处理
    public static ServerAioHandler aioHandler = new BlockChainServerAioHandler();

    // 3. 事件监听器，可为null，但最好自己实现，可以参考showcase接口
    public static ServerAioListener aioListener = null;

    // 4. 一组连接共用的上下文对象
    public static ServerGroupContext serverGroupContext =
            new ServerGroupContext("hello-tio-server", aioHandler, aioListener);

    // 5. tioServer对象
    public static TioServer tioServer = new TioServer(serverGroupContext);

    // 6. 有时候需要绑定ip，不需要则null
    public static String serverIp = null; //TioConst.SERVER;

    // 7. 监听窗口
    public static int serverPort = TioConst.PORT;

//    @PostConstruct
//    @Order(1)
    public void start(){
        try {
            logger.info("512服务端即将启动----");
            serverGroupContext.setHeartbeatTimeout(TioConst.TIMEOUT);
            tioServer.start(serverIp, serverPort);
            logger.info("512服务端启动完毕0.0");
        } catch (IOException e) {
            logger.error(e.getMessage());
        }

    }
}

