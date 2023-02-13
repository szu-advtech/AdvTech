package com.example.application0;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.app.Application;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.XmlResourceParser;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.media.SoundPool;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.text.TextUtils;
import android.util.Log;
import android.util.Xml;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.xmlpull.v1.XmlPullParser;
import org.xmlpull.v1.XmlPullParserFactory;
import org.xmlpull.v1.XmlSerializer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements View.OnClickListener, View.OnTouchListener {

    private Button startBtn;
    private Button objectBtn;
    private Button levelBtn;
    private Button musicBtn;
    private Button descriptionBtn;
    private Button contourBtn;
    private ImageView sample_img;
    private TextView tnow;

    private List<String> obj_name = new ArrayList<>();
    private Map<String, List<MatOfPoint>> map_contours = new HashMap<>();
    private Map<String, List<MatOfPoint>> level_contours = new HashMap<>();
    private String beforeObject = "";
    private String currentObject = "";
    private Boolean isLock = false;
    private Boolean isContourFeel = false;

    private SoundPoolUtils spu;
    private MediaPlayer mediaPlayer;

    private final double[][] array_curvature = {
            {-1.0/16, 5.0/16, -1.0/16},
            {5.0/16, -1.0, 5.0/16},
            {-1.0/16, 5.0/16, -1.0/16}
    };

    protected double curvatureConv(Point touch) {
        Bitmap image = ((BitmapDrawable)sample_img.getDrawable()).getBitmap();
        double ans = 0.0;
        int x = (int) touch.x;
        int y = (int) touch.y;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                int argb = image.getPixel(x+i, y+j);
                int r = (argb >> 16) &0xff;
                int g = (argb >> 8) &0xff;
                int b = argb &0xff;
                double gray = 0.2989*r+0.587*g+0.114*b;
                ans += gray*array_curvature[i+1][j+1];
            }
        }
        return Math.abs(ans);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        View decorView = this.getWindow().getDecorView();
        decorView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_FULLSCREEN
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);

        sample_img = (ImageView) findViewById(R.id.sample_img);
        tnow = (TextView) findViewById(R.id.tnow);
        sample_img.setClickable(true);
        sample_img.setOnTouchListener(this);

        loadOpencv();
        mediaPlayer = new MediaPlayer();

        levelBtn = (Button)this.findViewById(R.id.level_btn);
        levelBtn.setOnClickListener(this);
        objectBtn = (Button)this.findViewById(R.id.object_btn);
        objectBtn.setOnClickListener(this);
        musicBtn = (Button)this.findViewById(R.id.music_btn);
        musicBtn.setOnClickListener(this);
        descriptionBtn = (Button)this.findViewById(R.id.description_btn);
        descriptionBtn.setOnClickListener(this);
        startBtn = (Button)this.findViewById(R.id.start_btn);
        startBtn.setOnClickListener(this);
        contourBtn = (Button)this.findViewById(R.id.contour_btn);
        contourBtn.setOnClickListener(this);

        handlePermission();
    }

    @Override
    public void onClick(View view) {
        ImageView iv = (ImageView)this.findViewById(R.id.sample_img);
        double[][] HSV_VALUE_LOW = {
            {35, 120, 100}, //green
            {100, 176, 160},    //blue
            {0, 43, 46}    //red
        };
        double[][] HSV_VALUE_HIGH = {
            {77, 255, 255},
            {130, 255, 255},
            {10, 255, 255}
        };
        String[] fil = {
             "church","forest","house","mist","moon","mountain","starcloud","stars","tree"
        };
        String[] lev = {"starry", "village", "cypress"};
        int[] draw = {
              R.drawable.church, R.drawable.forest, R.drawable.house, R.drawable.mist, R.drawable.moon,
              R.drawable.mountain, R.drawable.starcloud, R.drawable.stars, R.drawable.tree
        };
        switch (view.getId()){
            case R.id.start_btn:
                tnow.setText(" ");
                map_contours.clear();
                level_contours.clear();
                iv.setImageBitmap(BitmapFactory.decodeResource(this.getResources(), R.drawable.source));
                if (mediaPlayer != null && mediaPlayer.isPlaying()){
                    mediaPlayer.stop();
                    mediaPlayer.reset();
                }
                try {
                    mediaPlayer.setDataSource(getExternalFilesDir(null)+"/Atest/music/start.mp3");
                    mediaPlayer.prepare();
                    mediaPlayer.start();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                break;
            case R.id.object_btn:
                if (isLock) {
                    map_contours.clear();
                    map_contours = getUserInfo("object");
                    mediaPlayer.stop();
                    mediaPlayer.reset();
                    isContourFeel = false;
                    isLock = false;
                }
                tnow.setText(" ");
                map_contours.clear();
                level_contours.clear();
                iv.setImageBitmap(BitmapFactory.decodeResource(this.getResources(), R.drawable.source));
                if (mediaPlayer != null && mediaPlayer.isPlaying()){
                    mediaPlayer.stop();
                    mediaPlayer.reset();
                }
                map_contours = getUserInfo("object");

                break;
            case R.id.level_btn:
                if (isLock) {
                    isContourFeel = false;
                    isLock = false;
                }

                tnow.setText(" ");
                map_contours.clear();
                iv.setImageBitmap(BitmapFactory.decodeResource(this.getResources(), R.drawable.source));
                if (mediaPlayer != null && mediaPlayer.isPlaying()){
                    mediaPlayer.stop();
                    mediaPlayer.reset();
                }
                level_contours = getUserInfo("level");
                map_contours = level_contours;
                break;
            case R.id.music_btn:
                level_contours.clear();
                if (isLock && level_contours.size()==0){
                    try {
                        if (mediaPlayer != null)    mediaPlayer.stop();
                        mediaPlayer.reset();
                        if (currentObject.equals("stars")){
                            mediaPlayer.setDataSource(getExternalFilesDir(null)+"/Atest/music/starsound.wav");
                        }
                        else if (currentObject.equals("tree")){
                            mediaPlayer.setDataSource(getExternalFilesDir(null)+"/Atest/music/treesound.wav");
                        }
                        else if (currentObject.equals("church") || currentObject.equals("house") || currentObject.equals("forest")){
                            mediaPlayer.setDataSource(getExternalFilesDir(null)+"/Atest/music/villagesound.wav");
                        }
                        else {
                            break;
                        }
                        mediaPlayer.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                            @Override
                            public void onPrepared(MediaPlayer mediaPlayer) {
                                mediaPlayer.start();
                            }
                        });
                        mediaPlayer.prepareAsync();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
            case R.id.contour_btn:
                if (isLock && level_contours.size()==0) {
                    isContourFeel = true;
                }
                break;
            case R.id.description_btn:
                if (isLock && level_contours.size()==0){
                    try {
                        if (mediaPlayer != null)    mediaPlayer.stop();
                        mediaPlayer.reset();
                        mediaPlayer.setDataSource(getExternalFilesDir(null)+"/Atest/music/"+currentObject+"_des.mp3");
                        mediaPlayer.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                            @Override
                            public void onPrepared(MediaPlayer mediaPlayer) {
                                mediaPlayer.start();
                            }
                        });
                        mediaPlayer.prepareAsync();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
            default:
                break;
        }
    }

    public boolean saveUserInfo(String name, List<MatOfPoint> contour, String file){
        try {
            //sdk 大于6.0的判断
            if (Build.VERSION.SDK_INT >= 23) {
                int REQUEST_CODE_CONTACT = 101;
                String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
                //验证是否许可权限
                for (String str : permissions) {
                    if (this.checkSelfPermission(str) != PackageManager.PERMISSION_GRANTED) {
                        //申请权限
                        this.requestPermissions(permissions, REQUEST_CODE_CONTACT);
                    } else {
                        String path = getExternalFilesDir(null)+"/Atest/doc";
                        Log.e("------path", path);
                        File files = new File(path);
                        if (!files.exists()) {
                            files.mkdirs();
                        }
                        try {
                            FileWriter fw = new FileWriter(path + File.separator + file + ".txt", true);
                            fw.write("\r\n");
                            fw.write(name);
                            for(MatOfPoint point : contour){
                                fw.write("##");
                                List<Point> temp = point.toList();
                                for (Point p : temp){
                                    fw.write(p.toString());
                                }
                            }
                            fw.flush();
                            fw.close();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    public Map<String, List<MatOfPoint>> getUserInfo(String file){
        String path = getExternalFilesDir(null)+"/Atest/doc/"+ file +".txt";
        Map<String, List<MatOfPoint>> tempmap = new HashMap<>();
        try {
            FileInputStream fis = new FileInputStream(path);
            BufferedReader reader = new BufferedReader(new InputStreamReader(fis));
            String text;
            while ((text = reader.readLine()) != null){
                if(!TextUtils.isEmpty(text)){
                    String[] infos = text.split("##");
                    List<MatOfPoint> contour = new ArrayList<>();
                    for(int i = 1; i < infos.length; ++i){
                        boolean fx = false;
                        boolean fy = false;
                        int x = 0;
                        int y = 0;
                        List<Point> lpt = new ArrayList<>();
                        for(int j = 0; j < infos[i].length(); ++j){
                            if(infos[i].charAt(j) == '}'){
                                lpt.add(new Point(x, y));
                            }
                            if(infos[i].charAt(j) == '{'){
                                fx = true;
                                x = j+1;
                                continue;
                            }
                            if(infos[i].charAt(j) == ','){
                                fy = true;
                                y = j+2;
                                continue;
                            }
                            if(infos[i].charAt(j) == '.'){
                                if(fx)  x = Integer.parseInt(infos[i].substring(x, j));
                                if(fy)  y = Integer.parseInt(infos[i].substring(y, j));
                                fx = false;
                                fy = false;
                            }
                        }
                        MatOfPoint mop = new MatOfPoint();
                        mop.fromList(lpt);
                        contour.add(mop);
                    }
                    tempmap.put(infos[0], contour);
                }
            }
            return tempmap;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return new HashMap<>();
    }

    private boolean saveXML(){
        try {
            //sdk 大于6.0的判断
            if (Build.VERSION.SDK_INT >= 23) {
                int REQUEST_CODE_CONTACT = 101;
                String[] permissions = {Manifest.permission.WRITE_EXTERNAL_STORAGE};
                //验证是否许可权限
                for (String str : permissions) {
                    if (this.checkSelfPermission(str) != PackageManager.PERMISSION_GRANTED) {
                        //申请权限
                        this.requestPermissions(permissions, REQUEST_CODE_CONTACT);
                    } else {
                        String path = getExternalFilesDir(null)+"/Atest";
                        Log.e("------path", path);
                        File files = new File(path);
                        if (!files.exists()) {
                            files.mkdirs();
                        }
                        try {
                            File file = new File(path, "object.xml");
                            FileOutputStream fos = new FileOutputStream(file);
                            XmlSerializer serializer = Xml.newSerializer();
                            serializer.setOutput(fos, "UTF-8");
                            serializer.startDocument("UTF-8", true);
                            serializer.startTag(null, "paint");
                            for (String name : obj_name) {
                                serializer.startTag(null, "name");
                                serializer.text(name);
                                serializer.endTag(null, "name");
                                serializer.startTag(null, "contours");
                                String points = "";
                                for (MatOfPoint point : map_contours.get(name)) {
                                    List<Point> temp = point.toList();
                                    for (Point p : temp) {
                                        points = points.concat((int) p.x + "," + (int) p.y + "#");
                                    }
                                }
                                serializer.text(points);
                                serializer.endTag(null, "contours");
                            }
                            serializer.endTag(null, "paint");
                            serializer.endDocument();
                            serializer.flush();
                            fos.close();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
            return true;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return false;
    }

    public Map<String, List<MatOfPoint>> getXML(String path) throws Throwable {
        FileInputStream stream = new FileInputStream(getExternalFilesDir(null)+"/Atest/"+path);
        Map<String, List<MatOfPoint>> temp_map = new HashMap<>();
        XmlPullParserFactory parserFactory = XmlPullParserFactory.newInstance();
        XmlPullParser parser = parserFactory.newPullParser();
        parser.setInput(stream, "UTF-8");
        int type = parser.getEventType();
        String key = "";
        while (type != XmlPullParser.END_DOCUMENT) {
            switch (type) {
                case XmlPullParser.START_TAG:
                    String name = parser.getName();
                    if (name.equals("name")) {
                        key = parser.nextText();
                        obj_name.add(key);
                    }
                    if (name.equals("contours")) {
                        String text = parser.nextText();
                        Log.d("text", key+text);
                        if (!TextUtils.isEmpty(text)) {
                            String[] infos = text.split("#");
                            List<MatOfPoint> contour = new ArrayList<>();
                            for (int i = 0; i < infos.length; ++i) {
                                List<Point> lpt = new ArrayList<>();
                                int len = infos[i].length();
                                for (int j = 0; j < len; ++j) {
                                    if (infos[i].charAt(j) == ',') {
                                        int x = Integer.parseInt(infos[i].substring(0, j));
                                        int y = Integer.parseInt(infos[i].substring(j + 1, len));
                                        lpt.add(new Point(x, y));
                                        break;
                                    }
                                }
                                MatOfPoint mop = new MatOfPoint();
                                mop.fromList(lpt);
                                contour.add(mop);
                            }
                            temp_map.put(key, contour);
                        }
                    }
                    break;
                default:
                    break;
            }
            type = parser.next();
        }
        return temp_map;
    }

    private void loadOpencv(){
        boolean success = OpenCVLoader.initDebug();
        if(success){
        }
        else{
            Toast.makeText(this.getApplicationContext(), "not load opencv", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public boolean onTouch(View v, MotionEvent event) {
        switch (event.getAction()){
            case MotionEvent.ACTION_DOWN:
                View decorView = getWindow().getDecorView();
                decorView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                        | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                        | View.SYSTEM_UI_FLAG_FULLSCREEN
                        | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
                break;
            case MotionEvent.ACTION_MOVE:
                if(!isLock && map_contours.size() > 0){
                    for (Map.Entry<String, List<MatOfPoint>> entry : map_contours.entrySet()){
                        List<MatOfPoint> temp_contour = entry.getValue();
                        if(temp_contour.size() > 0){
                            Point srcPt = touchInvert(event);
                            for(MatOfPoint point : temp_contour){
                                MatOfPoint2f point2f = new MatOfPoint2f();
                                point.convertTo(point2f, CvType.CV_32F);
                                double inOrOut = Imgproc.pointPolygonTest(point2f, srcPt, false);
                                if(inOrOut >= 0 && Imgproc.contourArea(point)>100){

                                    double touch_curvature = curvatureConv(srcPt);//could not be real-time
                                    tnow.setText(entry.getKey()+" "+touch_curvature);
                                    Log.e("curvature",entry.getKey()+" "+touch_curvature);

                                    if (level_contours.size() > 0){
                                        try {
                                            if (mediaPlayer != null)    mediaPlayer.stop();
                                            mediaPlayer.reset();
                                            mediaPlayer.setDataSource(getExternalFilesDir(null)+"/Atest/music/"+entry.getKey()+".mp3");
                                            mediaPlayer.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                                                @Override
                                                public void onPrepared(MediaPlayer mediaPlayer) {
                                                    mediaPlayer.start();
                                                }
                                            });
                                            mediaPlayer.prepareAsync();
                                        } catch (IOException e) {
                                            e.printStackTrace();
                                        }
                                        break;
                                    }

                                    if (!beforeObject.equals(entry.getKey())){
                                        spu = new SoundPoolUtils(this.getApplicationContext());
                                        spu.startAudioAndVibrator(entry.getKey(), 1000);
                                    }
                                    beforeObject = entry.getKey();
                                    break;
                                }
                                else tnow.setText(" ");
                            }
                        }
                        if (map_contours.containsKey(tnow.getText()))   break;
                    }
                }
                if (isContourFeel && level_contours.size()==0) {
                    List<MatOfPoint> temp_contour = map_contours.get(currentObject);
                    if (temp_contour.size() > 0) {
                        Point srcPt = touchInvert(event);
                        for (MatOfPoint point : temp_contour) {
                            MatOfPoint2f point2f = new MatOfPoint2f();
                            point.convertTo(point2f, CvType.CV_32F);
                            double inOrOut = Imgproc.pointPolygonTest(point2f, srcPt, true);
                            if (inOrOut >= 0 && inOrOut <= 40 && Imgproc.contourArea(point) > 100) {
                                spu = new SoundPoolUtils(this.getApplicationContext());
                                spu.startVibrator(1000);
                                break;
                            }
                            else tnow.setText(" ");
                        }
                    }
                }
                break;
            case MotionEvent.ACTION_UP:
                if (!isLock) currentObject = beforeObject;
                if (level_contours.size() == 0) {
                    isLock = true;
                    List<MatOfPoint> tempcontour = map_contours.get(currentObject);
                    map_contours.clear();
                    map_contours.put(currentObject, tempcontour);
                }
                break;
            default:
                break;
        }
        return false;
    }

    //屏幕触摸点和图像坐标的转换
    private Point touchInvert(MotionEvent e){
        float dst[] = new float[2];
        Matrix imageMatrix = sample_img.getImageMatrix();
        Matrix inverseMatrix = new Matrix();
        imageMatrix.invert(inverseMatrix);
        inverseMatrix.mapPoints(dst, new float[]{e.getX(), e.getY()});
        return new Point(dst[0], dst[1]);
    }

    public void handlePermission() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {

        }
        else {
            ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }
    }

}

//音效和振动封装的工具包类
class SoundPoolUtils {

    private static final int MAX_STREAMS = 16;
    private static final int DEFAULT_QUALITY = 0;
    private static final int DEFAULT_PRIORITY = 1;
    private static final int LEFT_VOLUME = 1;
    private static final int RIGHT_VOLUME = 1;
    private static final int LOOP = 0;
    private static final float RATE = 1.0f;

    private static SoundPoolUtils sSoundPoolUtils;

    /**
     * 音频的相关类
     */
    private SoundPool mSoundPool;
    private Context mContext;
    private Vibrator mVibrator;


    public SoundPoolUtils(Context context) {
        mContext = context;
        //初始化行营的音频类
        intSoundPool();
        initVibrator();
    }

    private void intSoundPool() {
        //根据不同的版本进行相应的创建
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            mSoundPool = new SoundPool.Builder()
                    .setMaxStreams(MAX_STREAMS)
                    .build();
        } else {
            mSoundPool = new SoundPool(MAX_STREAMS, AudioManager.STREAM_MUSIC, DEFAULT_QUALITY);
        }
    }

    private void initVibrator() {
        mVibrator = (Vibrator) mContext.getSystemService(Context.VIBRATOR_SERVICE);
    }

    public static SoundPoolUtils getInstance(Context context) {
        if (sSoundPoolUtils == null) {
            synchronized (SoundPoolUtils.class) {
                if (sSoundPoolUtils == null) {
                    sSoundPoolUtils = new SoundPoolUtils(context);
                }
            }
        }
        return sSoundPoolUtils;
    }

    public void playVideo(int resId) {
        if (mSoundPool == null) {
            intSoundPool();
        }
        int load = mSoundPool.load(mContext, resId, DEFAULT_PRIORITY);
        mSoundPool.play(load, LEFT_VOLUME, RIGHT_VOLUME, DEFAULT_PRIORITY, LOOP, RATE);
    }

    public void playAudio(String file){
        if (mSoundPool == null) {
            intSoundPool();
        }
        final int load = mSoundPool.load(this.mContext.getExternalFilesDir(null)+"/Atest/music/"+file+".mp3", DEFAULT_PRIORITY);
        mSoundPool.setOnLoadCompleteListener(new SoundPool.OnLoadCompleteListener() {
            @Override
            public void onLoadComplete(SoundPool soundPool, int i, int i1) {
                mSoundPool.play(load, LEFT_VOLUME, RIGHT_VOLUME, DEFAULT_PRIORITY, LOOP, RATE);
            }
        });
    }

    public void startVibrator(long milliseconds) {
        if (mVibrator == null) {
            initVibrator();
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            VibrationEffect vibrationEffect = VibrationEffect.createOneShot(milliseconds, 100);
            mVibrator.vibrate(vibrationEffect);
        } else {
            mVibrator.vibrate(1000);
        }
    }

    public void startVideoAndVibrator(int resId, long milliseconds) {
        playVideo(resId);
        startVibrator(milliseconds);
    }

    public void startAudioAndVibrator(String file, long milliseconds){
        playAudio(file);
        startVibrator(milliseconds);
    }

    public void release(int resID) {
        //释放所有的资源
        if (mSoundPool != null) {
            mSoundPool.autoPause();
            mSoundPool.unload(resID);
            mSoundPool.release();
            mSoundPool = null;
        }

        if (mVibrator != null) {
            mVibrator.cancel();
            mVibrator = null;
        }
    }
}

abstract class DoubleClickListener implements View.OnClickListener {
    private static final long DOUBLE_TIME = 500;
    private static long lastClickTime = 0;

    @Override
    public void onClick(View view) {
        long currentTimeMillis = System.currentTimeMillis();
        if (currentTimeMillis - lastClickTime < DOUBLE_TIME) {
            onDoubleClick(view);
        }
        lastClickTime = currentTimeMillis;
    }

    public abstract void onDoubleClick(View v);
}
