package org.yaodong.p2ppbft;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import org.java_websocket.WebSocket;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import org.java_websocket.server.WebSocketServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.testng.util.Strings;
import org.yaodong.util.MerkleTreeUtil;

import javax.annotation.PostConstruct;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

@Component
public class P2pPointPbftClient {
    private Logger logger = LoggerFactory.getLogger(P2pPointPbftClient.class);

    private  String wsUrl = "ws://localhost:7002/";

    private List<WebSocket> localSockets = new ArrayList<>();


    @PostConstruct
    @Order(2)
    /**
     * 连接到客户端
     */
    public void connectPeer(){
        try{
            final WebSocketClient webSocketClient = new WebSocketClient(new URI(wsUrl)) {
                @Override
                public void onOpen(ServerHandshake serverHandshake) {
                    sendMessage(this, "512客户端打开");
                    localSockets.add(this);
                }

                @Override
                public void onMessage(String msg) {
                    logger.info("收到512服务端发送的信息："+msg);
                    //如果收到的不是JSON数据则说明不是pbft阶段
                    if(!msg.startsWith("{"))return;
                    //如果收到的是JSON数据则说明是pbft阶段
                    JSONObject jsonObject = JSON.parseObject(msg);
                    if(!jsonObject.containsKey("code"))
                        logger.info("512收到JSON数据");
                    int code = jsonObject.getIntValue("code");
                    if(code == VoteEnum.PREPREPARE.getCode()){
                        //校验hash
                        VoteInfo voteInfo = JSON.parseObject(msg, VoteInfo.class);
                        if(!voteInfo.getHash().equals(MerkleTreeUtil.getTreeNodeHash(voteInfo.getList()))){
                            logger.info("收到512服务端错误的JSON数据");
                            return;
                        }
                        //校验成功,发送下一个状态的数据
                        VoteInfo vi = createVoteInfo(VoteEnum.PREPARE);
                        sendMessage(this, JSON.toJSONString(vi));
                        logger.info("512发送客户端pbft消息："+JSON.toJSONString(vi));
                        return;
                    }
                    if(code == VoteEnum.COMMIT.getCode()){
                        // 校验hash
                        VoteInfo voteInfo = JSON.parseObject(msg, VoteInfo.class);
                        if (!voteInfo.getHash().equals(MerkleTreeUtil.getTreeNodeHash(voteInfo.getList()))) {
                            logger.info("收到512服务端错误的JSON数据");
                            return;
                        }

                        // 校验成功，检验节点确认个数是否有效
                        if (getConnecttedNodeCount() >= getLeastNodeCount()) {
                            sendMessage(this, "512客户端开始区块入库啦");
                            logger.info("512客户端开始区块入库啦");
                        }
                    }
                }

                @Override
                public void onClose(int i, String s, boolean b) {
                    logger.info("512客户端关闭");
                    localSockets.remove(this);
                }

                @Override
                public void onError(Exception e) {
                    logger.info("512客户端报错");
                    localSockets.remove(this);
                }
            };
            webSocketClient.connect();
        }catch (URISyntaxException e){
            logger.info("好未来北京连接错误:" + e.getMessage());
        }
    }

    // 已经在连接的节点的个数
    private int getConnecttedNodeCount() {
        // 本机测试时，写死为1.实际开发部署多个节点时，按实际情况返回
        return 1;
    }

    // pbft消息节点最少确认个数计算
    private int getLeastNodeCount() {
        // 本机测试时，写死为1.实际开发部署多个节点时，pbft算法中拜占庭节点数量f，总节点数3f+1
        return 1;
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
