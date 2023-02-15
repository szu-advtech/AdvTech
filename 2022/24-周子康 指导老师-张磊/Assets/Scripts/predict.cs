using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// 预测相机的类
struct CameraState
{
    public float yaw;
    public float pitch;
    public float roll;
    public float x;
    public float y;
    public float z;
}
/// <summary>
/// 用于预测相机的视角，采用的预测方法是简单的线性回归算法。
/// </summary>
public class predict 
{
    #region 辅助类
    class CamClass
    {
        public CameraState camdata = new CameraState();

        public void SetFromTransform(Transform t)
        {
            camdata.pitch = t.eulerAngles.x;
            camdata.yaw = t.eulerAngles.y;
            camdata.roll = t.eulerAngles.z;
            camdata.x = t.position.x;
            camdata.y = t.position.y;
            camdata.z = t.position.z;
        }
    }
    #endregion

    #region 参数设置

    public GameObject cam;           // 观看相机
    public GameObject void_cam;      // 预测相机    
    private int Max;                 // 历史窗口大小
    private int prewin;              // 预测窗口
    private CameraState pre_cam;     // 存放预测的相机信息
    Queue<CameraState> history = new Queue<CameraState>();   // 历史信息
    CamClass new_cam = new CamClass();
    #endregion


    public predict()       
    {
        cam = GameObject.Find("myCamera"); 
        void_cam = GameObject.Find("voidCamera");

        prewin = 5;
        Max = 4; 
    }

    public void Exe_pre()
    {   
        // 信息采样
        new_cam.SetFromTransform(cam.transform);
        history.Enqueue(new_cam.camdata);

        if (history.Count >= Max)
        {
            history.Dequeue();
        }

        // 执行预测
        predict_1();

        // 将信息更新到预测的相机中
        updateCam();
    }

    private void updateCam()
    {
        void_cam.transform.localEulerAngles = new Vector3(pre_cam.pitch, pre_cam.yaw, pre_cam.roll);
        void_cam.transform.localPosition = new Vector3(pre_cam.x, pre_cam.y, pre_cam.z);
    }


    private void predict_1()
    {
        CameraState[] his = history.ToArray();

        // 初始时，数据不够
        if (history.Count < Max)
        {
            pre_cam = his[history.Count - 1];  // 获取最新的dian
            return;
        }

        // 初始化直线辅助参数，一共6条，a,b 公用参数，c,d 各自有
        float a = 0;
        float b = 0;
        float[] c = new float[] { 0, 0, 0, 0, 0, 0 };
        float[] d = new float[] { 0, 0, 0, 0, 0, 0 };
        // 计算辅助参数
        for (int i = 0; i < Max; i++)
        {
            a += i * i;
            b += i;
        }
        for (int i = 0; i < Max; i++)
        {
            c[0] += i * his[i].x;
            c[1] += i * his[i].y;
            c[2] += i * his[i].z;
            c[3] += i * his[i].yaw;
            c[4] += i * his[i].pitch;
            c[5] += i * his[i].roll;
            d[0] += his[i].x;
            d[1] += his[i].y;
            d[2] += his[i].z;
            d[3] += his[i].yaw;
            d[4] += his[i].pitch;
            d[5] += his[i].roll;

        }

        // 计算直线
        float[] k = new float[6];
        float[] t = new float[6];
        for (int i = 0; i < 6; i++)
        {
            k[i] = (c[i] * 3 - b * d[i]) / (a * 3 - b * b);
            t[i] = (a * d[i] - c[i] * b) / (a * 3 - b * b);
        }

        // 计算预测值
        pre_cam.x = (Max + prewin) * k[0] + t[0];
        pre_cam.y = (Max + prewin) * k[1] + t[1];
        pre_cam.z = (Max + prewin) * k[2] + t[2];
        pre_cam.yaw = (Max + prewin) * k[3] + t[3];
        pre_cam.pitch = (Max + prewin) * k[4] + t[4];
        pre_cam.roll = (Max + prewin) * k[5] + t[5];
    }
}
