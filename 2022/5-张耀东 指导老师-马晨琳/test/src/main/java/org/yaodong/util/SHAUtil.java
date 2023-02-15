package org.yaodong.util;

import java.io.UnsupportedEncodingException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

import cn.hutool.crypto.digest.DigestUtil;
import org.apache.commons.codec.binary.Hex;
/**
 * 用apache commons-codec工具类实现SHA-256加密
 * @author yaodong
 */
public class SHAUtil {
    /**
     * 用Apache commons-codec的工具类实现SHA-256加密
     * @param originalString 明文
     * @return 密文
     */
    public static String getSHA256BasedMD(String originalString){
        MessageDigest messageDigest;
        String encodeString = "";
        try{
            messageDigest = MessageDigest.getInstance("SHA-256");
            byte[] hash = messageDigest.digest(originalString.getBytes("UTF-8"));
            encodeString = Hex.encodeHexString(hash);
        }catch (NoSuchAlgorithmException | UnsupportedEncodingException e){
            e.printStackTrace();
        }
        return encodeString;
    }

    /**
     * 用Hutool的工具类实现SHA-256加密
     * @param originalString 明文
     * @return 密文
     */
    public static String sha256BasedHutool(String originalString){
        return DigestUtil.sha256Hex(originalString);
    }
}
