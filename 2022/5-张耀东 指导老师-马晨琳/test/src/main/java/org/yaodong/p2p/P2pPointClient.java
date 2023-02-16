package org.yaodong.p2p;

import org.java_websocket.WebSocket;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.testng.util.Strings;

import javax.annotation.PostConstruct;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;


/**
 * 基于springboot2.0的websocket客户端
 * @author yaodong
 */
@Component
public class P2pPointClient {
    // 1.日志记录
    private Logger logger = LoggerFactory.getLogger(P2pPointClient.class);

    // 2. p2p网络中的节点既是服务端，又是客户端，作为Server运行在7001端口，同时作为client
    // 通过ws://localhost:7200/连接到服务端
    private String websocketUrl = "ws://localhost:7200/";

    // 3. 所有客户端WebSocket的连接缓存池
    private List<WebSocket> localSockets = new ArrayList<>();

    public List<WebSocket> getLocalSockets() {
        return localSockets;
    }

    public void setLocalSockets(List<WebSocket> localSockets) {
        this.localSockets = localSockets;
    }

    /**
     * 连接到服务端
     */
//    @PostConstruct
//    @Order(2)
    public void connectPeer(){
        try{
            final WebSocketClient socketClient = new WebSocketClient(new URI(websocketUrl)) {
                @Override
                public void onOpen(ServerHandshake serverHandshake) {
                    sendMessage(this, "客户端打开");
                    localSockets.add(this);
                }

                @Override
                public void onMessage(String s) {
                    logger.info("收到来自服务端发送的消息："+s);
                }

                @Override
                public void onClose(int i, String s, boolean b) {
                    logger.info("客户端关闭");
                    localSockets.remove(this);
                }

                @Override
                public void onError(Exception e) {
                    logger.info("客户端报错");
                    localSockets.remove(this);
                }
            };
            //客户端开始连接
            socketClient.connect();
        }catch (URISyntaxException e) {
            logger.info("连接错误："+e.getMessage());
        }
    }

    /**
     * 向服务端发送消息，当前websocket的远程socket的地址，就是服务器端
     * @param ws
     * @param msg
     */
    public void sendMessage(WebSocket ws, String msg){
        logger.info("发送给："+ws.getRemoteSocketAddress().getPort()+"的p2p消息是："+msg);
        ws.send(msg);
    }

    /**
     * 向所有连接过的服务端广播消息
     * @param msg
     */
    public void broadcast(String msg){
        if(localSockets.size()==0 || Strings.isNullOrEmpty(msg))
            return;
        logger.info("Glad to see broadcast to servers being started!");
        for (WebSocket localSocket : localSockets) {
            this.sendMessage(localSocket, msg);
        }
        logger.info("broadcast to servers has overread!");
    }
}
