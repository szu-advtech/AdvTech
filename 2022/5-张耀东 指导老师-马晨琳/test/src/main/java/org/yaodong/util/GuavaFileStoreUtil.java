package org.yaodong.util;

import com.google.common.base.Charsets;
import com.google.common.io.Files;
import org.tio.core.intf.Packet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * 模拟区块里文件存储方式：guava版
 * @author yaodong
 */
public class GuavaFileStoreUtil {
    private static final int FILE_SIZE = 1024; // 单位KB

    /**
     * 将文件内容写入目标文件：以文件名方式（Guava）
     * @param targetFileName
     * @param content
     */
    public static void writeIntoTargetFile(String targetFileName, String content){
        File newFile = new File(targetFileName);
        try {
            Files.write(content.getBytes(), newFile);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    // 将文件内容向后追加写入目标文件:FileWriter方式
    public static void appendToTargetFile(String targetFileName, String content) {
        try {
            // 打开一个写文件器，构造函数中的第二个参数true表示以追加形式写文件
            FileWriter writer = new FileWriter(targetFileName, true);
            writer.write(content);
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    /**
     * 以追加方式写入目标文件：Guava方式
     * @param targetFileName
     * @param content
     */
    public static void appendToTargetFileByGuava(String targetFileName, String content){
        File file = new File(targetFileName);
        try {
            Files.append(content, file, Charsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void writeIntoFile(String content){
        // 1.查看当前目录下是否存在写入的loging文件，有则继续使用，无则创建
        File root = new File(".//");
        File[] files = root.listFiles();
        if(files == null){
            // 如果根目录下没有任何文件则创建文件，loging后缀表示正在写入文件
            String targetFileName = ".//blockchain-"+System.currentTimeMillis()+".loging";
            appendToTargetFileByGuava(targetFileName, content);
            return;
        }
        // 如果根目录下有文件则寻找是否存在loging后缀的文件，
        boolean has = false;
        for (File file : files) {
            String name = file.getName();
            if(name.endsWith(".loging") && name.startsWith("blockchain-")){
                System.out.println(file.getPath());
                appendToTargetFileByGuava(file.getPath(), content);
                appendToTargetFileByGuava(file.getPath(), content);
                has = true;

                if(file.length() >= FILE_SIZE){
                    String logPath = file.getPath().replace("loging","log");
                    File log = new File(logPath);
                    try {
                        Files.copy(file, log);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    file.delete();
                }
            }
        }

        if(!has) {
            String targetFileName = ".//blockchain-"+System.currentTimeMillis()+".loging";
            appendToTargetFileByGuava(targetFileName, content);
            return;
        }
    }

    public static void writeIntoBlockFile(){
        List<String> list = new ArrayList<>();
        list.add("AI");
        list.add("BlockChain");
        list.add("BrainScience");
        for(int i=0; i<20; i++){
            list.add(generateVCode(6));
            writeIntoFile(MerkleTreeUtil.getTreeNodeHash(list)+"\n");
        }
    }
    public static String generateVCode(int length){
        Long vCode = (long)((Math.random()+1)*Math.pow(10, length-1));
        return vCode.toString();
    }
}
