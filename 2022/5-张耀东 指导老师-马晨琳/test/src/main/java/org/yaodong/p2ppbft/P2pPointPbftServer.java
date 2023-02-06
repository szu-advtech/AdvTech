package org.yaodong.p2ppbft;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.testng.util.Strings;
import org.yaodong.util.MerkleTree;
import org.yaodong.util.MerkleTreeUtil;

import javax.annotation.PostConstruct;
import java.net.InetSocketAddress;
import java.util.ArrayList;
import java.util.List;


@Component
public class P2pPointPbftServer {
    //1. 日志记录
    private Logger logger = LoggerFactory.getLogger(P2pPointPbftClient.class);
    //2. websocket端口，可变
    private int port = 7002;
    //3. 所有连接到服务端的websocket缓存器
    private List<WebSocket> localSockets = new ArrayList<>();

    @PostConstruct
    @Order(1)
    public void initServer(){
        final WebSocketServer webSocketServer = new WebSocketServer(new InetSocketAddress(port)) {
            @Override
            public void onOpen(WebSocket webSocket, ClientHandshake clientHandshake) {
                sendMessage(webSocket, "---512服务端打开---");

                // 当成功创建一个WebSocket连接时，将该链接加入连接池
                localSockets.add(webSocket);
            }

            @Override
            public void onClose(WebSocket webSocket, int i, String s, boolean b) {
                logger.info(webSocket.getRemoteSocketAddress() + "" +
                        "" +
                        "与服务器断开连接！");

                // 当客户端断开连接时，WebSocket连接池删除该链接
                localSockets.remove(webSocket);
            }

            @Override
            public void onMessage(WebSocket webSocket, String msg) {

                logger.info("512接收到客户端消息：" + msg);

                //收到入库的消息则不再发送
                if("客户端开始区块入库啦".equals(msg))
                    return;

                //如果收到的不是json数据，说明是双方在建立连接的过程中，目前连接已建立完毕，发起投票
                if(!msg.startsWith("{")){
                    VoteInfo vi = createVoteInfo(VoteEnum.PREPREPARE);
                    sendMessage(webSocket, JSON.toJSONString(vi));
                    logger.info("512发送pbft消息到客户端："+JSON.toJSONString(vi));
                    return;
                }

                // 如果是json数据，则说明进入到了投票阶段
                JSONObject jsonObject = JSON.parseObject(msg);
                if(!jsonObject.containsKey("code"))
                    logger.info("512收到非JSON数据");
                int code = jsonObject.getIntValue("code");
                if(code == VoteEnum.PREPARE.getCode()){
                    //检验hash
                    VoteInfo voteInfo = JSON.parseObject(msg, VoteInfo.class);
                    if(!voteInfo.getHash().equals(MerkleTreeUtil.getTreeNodeHash(voteInfo.getList()))){
                        logger.info("512收到错误的JSON数据");
                        return;
                    }

                    //校验成功，发送下一个状态的数据
                    VoteInfo vi = createVoteInfo(VoteEnum.COMMIT);
                    sendMessage(webSocket, JSON.toJSONString(vi));
                    logger.info("512发送客户端pbtf消息：" + JSON.toJSONString(vi));
                }
            }

            @Override
            public void onError(WebSocket webSocket, Exception e) {
                logger.info(webSocket.getRemoteSocketAddress() + "客户端链接错误！");
                localSockets.remove(webSocket);
            }

            @Override
            public void onStart() {
                logger.info("512--dididi--WebSocket Server端启动...");
            }
        };
        webSocketServer.start();
        logger.info("512监听socketServer端口："+port);
    }

    /**
     * 根据VoteEnum构建对应状态的VoteInfo
     * @param ve
     * @return
     */
    private VoteInfo createVoteInfo(VoteEnum ve){
        VoteInfo vi = new VoteInfo();
        vi.setCode(ve.getCode());
        List<String> list = new ArrayList<>();
        list.add("AI");
        list.add("BlockChain");
        vi.setList(list);
        vi.setHash(MerkleTreeUtil.getTreeNodeHash(list));
        return vi;
    }

    /**
     * 向连接到本机的某客户端发送消息
     *
     * @param ws
     * @param message
     */
    public void sendMessage(WebSocket ws, String message) {
        logger.info("发送给" + ws.getRemoteSocketAddress().getPort() + "的p2p消息是:" + message);
        ws.send(message);
    }

    /**
     * 向所有连接到本机的客户端广播消息
     *
     * @param message：待广播内容
     */
    public void broatcast(String message) {
        if (localSockets.size() == 0 || Strings.isNullOrEmpty(message)) {
            return;
        }

        logger.info("Glad to say broadcast to clients being startted!");
        for (WebSocket socket : localSockets) {
            this.sendMessage(socket, message);
        }
        logger.info("Glad to say broadcast to clients has overread!");
    }

    public List<WebSocket> getLocalSockets() {
        return localSockets;
    }

    public void setLocalSockets(List<WebSocket> localSockets) {
        this.localSockets = localSockets;
    }
}
