package org.yaodong.p2p;

import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.java_websocket.WebSocket;
import org.testng.util.Strings;

import javax.annotation.PostConstruct;
import java.net.InetSocketAddress;
import java.util.List;
import java.util.ArrayList;

/**
 * 基于springboot 2.0的java websocket服务端
 * @author yaodong
 */
@Component
public class P2pPointServer {
    // 1. slf4j日志记录
    private Logger logger = LoggerFactory.getLogger(P2pPointServer.class);

    // 2. websocket 端口
    private int port = 7200;

    // 3. 所有连接到服务器端的Websocket缓存器
    private List<WebSocket> localSockets = new ArrayList<>();

    public List<WebSocket> getLocalSockets() {
        return localSockets;
    }

    public void setLocalSockets(List<WebSocket> localSockets) {
        this.localSockets = localSockets;
    }

    /**
     * 向连接到本机的某客户端发送消息
     * @param ws
     * @param msg
     */
    public void sendMessage(WebSocket ws, String msg){
        logger.info("发送给："+ws.getRemoteSocketAddress().getPort()+"的p2p消息是："+msg);
        ws.send(msg);
    }

    /**
     * 向所有连接到本机的客户端广播消息
     * @param msg
     */
    public void broadcast(String msg){
        if(localSockets.size()==0 || Strings.isNullOrEmpty(msg))
            return;
        logger.info("Glad to see broadcast to clients being started!");
        for (WebSocket localSocket : localSockets) {
            this.sendMessage(localSocket, msg);
        }
        logger.info("broadcast to clients has overread!");
    }

    /**
     * 初始化WebSocket服务器，定义内部类对象socketServer，重写5个事件方法
     * InetSocketAddress(port)是WebSocketServer构造器的参数 InetSocketAddress是(IP地址+端口号)类型，亦即端口地址类型
     * 5个启动分别是服务器启动时onStart、创建连接成功时onOpen、断开连接时onClose。收到客户端来信时onMessage、连接发生错误时onClose
     */
//    @PostConstruct
//    @Order(1)
    public void initServer(){
        final WebSocketServer socketServer = new WebSocketServer(new InetSocketAddress(port)) {
            @Override
            public void onOpen(WebSocket webSocket, ClientHandshake clientHandshake) {
                sendMessage(webSocket, "hello world");
                //当websocket连接创建成功时，将该连接加入连接池
                localSockets.add(webSocket);
            }

            @Override
            public void onClose(WebSocket webSocket, int i, String s, boolean b) {
                logger.info(webSocket.getRemoteSocketAddress()+"客户端与服务器断开连接。");

                //当客户端断开连接时，websocket连接池将其删除
                localSockets.remove(webSocket);
            }

            @Override
            public void onMessage(WebSocket webSocket, String s) {
                logger.info("鸡你太美" + s);
                sendMessage(webSocket, "鸡你太美");
            }

            @Override
            public void onError(WebSocket webSocket, Exception e) {
                logger.info(webSocket.getRemoteSocketAddress() + "客户端连接错误。");
                localSockets.remove(webSocket);
            }

            @Override
            public void onStart() {
                logger.info("WebSocket Server端启动");
            }
        };
        socketServer.start();
        logger.info("监听socketServer端口："+ port);
    }
}
