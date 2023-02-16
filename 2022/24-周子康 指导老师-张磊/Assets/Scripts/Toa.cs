using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using System.Linq;


public class Toa : MonoBehaviour
{
    private predict pre_cam;          //预测类的实例
    private GameObject cam;           // 用于接收预测的结果
    private CreateTile createTile;    // 创建tile的实例
    private GameObject actul_cam;
    public List<string> filename;

    private List<CameraState> cam_position_list;
    private int cam_pos_index = 0;

    void Start()
    {
        filename = new List<string>();
        pre_cam = new predict();
        createTile = new CreateTile();
        cam = GameObject.Find("voidCamera");
        actul_cam = GameObject.Find("myCamera");
        cam_position_list = new List<CameraState>();
        initial_cam_pos();
    }

    private void initial_cam_pos()
    {
        // 读取相机的轨迹，用于实验
        BinaryReader file_read = new BinaryReader(new FileStream("F:/PLY_test/camera_pos.txt", FileMode.Open, FileAccess.Read));
        while(file_read.BaseStream.Position < file_read.BaseStream.Length)
        {
            CameraState temp = new CameraState();
            temp.x = file_read.ReadSingle();
            temp.y = file_read.ReadSingle();
            temp.z = file_read.ReadSingle();
            temp.yaw = file_read.ReadSingle();
            temp.pitch = file_read.ReadSingle();
            temp.roll = file_read.ReadSingle();
            cam_position_list.Add(temp);
            
        }         
    }

    private void new_cam(int index)
    {
        CameraState temp = cam_position_list[index];
        cam.transform.localEulerAngles = new Vector3(temp.yaw, temp.pitch, temp.roll);
        cam.transform.localPosition = new Vector3(temp.x, temp.y, temp.z);
    }

    private void write_file()
    {
        // 在 FixedUpdate中调用，写一个相机的轨迹文件。
        FileStream file_Stream = new FileStream("F:/PLY_test/camera_pos.txt", FileMode.Append, FileAccess.Write);
        BinaryWriter binary_Writer = new BinaryWriter(file_Stream);
        binary_Writer.Write(actul_cam.transform.localPosition.x);
        binary_Writer.Write(actul_cam.transform.localPosition.y);
        binary_Writer.Write(actul_cam.transform.localPosition.z);
        binary_Writer.Write(actul_cam.transform.localEulerAngles.x);
        binary_Writer.Write(actul_cam.transform.localEulerAngles.y);
        binary_Writer.Write(actul_cam.transform.localEulerAngles.z);
        binary_Writer.Close();
        file_Stream.Close();

    }

    // 数据部分代码
    void FixedUpdate()
    {
        if (cam_pos_index >= cam_position_list.Count)
        {
            cam_pos_index %= cam_position_list.Count;
        }
        new_cam(cam_pos_index);
        cam_pos_index += 1;

        pre_cam.Exe_pre();
        createTile.PV(cam);
        createTile.OV(cam);
        createTile.DV(cam);
        filename.Clear();
        createTile.Get_tile_number(filename);
    }
}
