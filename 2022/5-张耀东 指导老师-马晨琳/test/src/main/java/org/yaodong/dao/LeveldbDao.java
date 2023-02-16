package org.yaodong.dao;

import org.iq80.leveldb.DB;
import org.iq80.leveldb.DBFactory;
import org.iq80.leveldb.DBIterator;
import org.iq80.leveldb.Options;
import org.iq80.leveldb.impl.Iq80DBFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.util.Strings;

import javax.annotation.PostConstruct;
import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.IOException;
import java.util.Map;

public class LeveldbDao {
    private Logger logger = LoggerFactory.getLogger(LeveldbDao.class);
    private DB db;

    @PostConstruct
    public void init(){
        try {
            DBFactory factory = new Iq80DBFactory();
            Options options = new Options();
            options.createIfMissing(true);
            String path = "//.leveldb";
            db = factory.open(new File(path), options);
            db.put("hello world".getBytes(), "tal tech".getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // leveldb--put--存入/修改数据
    public void put(String key, String value){
        if(Strings.isNullOrEmpty(key) || Strings.isNullOrEmpty(value))
            return;
        db.put(Iq80DBFactory.bytes(key), Iq80DBFactory.bytes(value));
    }

    // 获取数据
    public String get(String key){
        if(Strings.isNullOrEmpty(key)) return null;
        byte[] valusBytes = db.get(Iq80DBFactory.bytes(key));
        return Iq80DBFactory.asString(valusBytes);
    }

    // 删除数据
    public void delete(String key) {
        if (com.google.common.base.Strings.isNullOrEmpty(key)) {
            return;
        }

        db.delete(Iq80DBFactory.bytes(key));
    }

    // 遍历所有入库数据
    public void traverseAllDatas() {
        DBIterator iterator = db.iterator();

        while (iterator.hasNext()) {
            Map.Entry<byte[], byte[]> next = iterator.next();
            String key = Iq80DBFactory.asString(next.getKey());
            String value = Iq80DBFactory.asString(next.getValue());
            logger.info("levelDb key=" + key + ";value=" + value);
        }
    }
    public static void main(String[] args) {
        try {
            DBFactory factory = new Iq80DBFactory();
            Options options = new Options();
            options.createIfMissing(true);
            // 配置db存储目录
            String path = ".//leveldb";
            DB db = factory.open(new File(path), options);

            String key = "hello world", value = "tal tech";
            db.put(Iq80DBFactory.bytes(key), Iq80DBFactory.bytes(value));

            byte[] valueBytes = db.get(Iq80DBFactory.bytes(key));
            System.out.println(Iq80DBFactory.asString(valueBytes));

            DBIterator iterator = db.iterator();
            while (iterator.hasNext()) {
                Map.Entry<byte[], byte[]> next = iterator.next();
                String keyT = Iq80DBFactory.asString(next.getKey());
                String valueT = Iq80DBFactory.asString(next.getValue());
                System.out.println("levelDb keyT=" + keyT + ";valueT=" + valueT);
            }

//            db.delete(Iq80DBFactory.bytes(key));
//            valueBytes = db.get(Iq80DBFactory.bytes(key));
//            System.out.println(Iq80DBFactory.asString(valueBytes));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


}
