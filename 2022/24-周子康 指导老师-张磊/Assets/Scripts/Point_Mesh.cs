using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System;


[RequireComponent(typeof(MeshFilter), typeof(MeshRenderer))]
public class Point_Mesh : MonoBehaviour
{
    private Point_cloud_data cloud_data;
    private List<string> file_list;
    private int maxPoint_number = 60000;    // 每一个Mesh的最大点数
    private int FPS = 5;
    private int frame_number = 300;         // 视频帧的总数
    private int frame = 1;
    private float time_inter;
    private GameObject root_mesh;

    void Start()
    {
        cloud_data = new Point_cloud_data();
        file_list = new List<string>();
        time_inter = 0;
        root_mesh = new GameObject();
        root_mesh.name = "Root_Mesh";
    }

    private void update_file_list(int frameindex)
    {
        file_list.Clear();
        string path = "F:/我的文件/ply/1_1_1/" + frameindex.ToString() + ".ply";
        file_list.Add(path);
        // string path = "F:/PLY/" + frameindex.ToString() + "/" + frameindex.ToString() + "_";   // 块的文件路径：如 F:/PLY/1/1_111_20.ply。代表第一帧 编号111 的块，压缩率20% 。
        /*Toa toa = GameObject.Find("Empty").GetComponent<Toa>();
        List<string> file = new List<string>();
        for (int i = 0; i < toa.filename.Count; i++)
        {
            string temp = path + toa.filename[i];
            if (!File.Exists(temp))
            {
                continue;
            }
            file_list.Add(temp);
        }*/
    }

    private void play()
    {
        cloud_data.Clear_cloud();
        Read_Cloud read_temp = new Read_Cloud();
        for(int i =0; i<file_list.Count;i++)
        {
            cloud_data.Append_cloud(read_temp.read_file(file_list[i]));
        }
        mesh();
    }
    
    #region 创建mesh渲染
    private void ClearChilds(Transform parent)
    {
        if (parent.childCount > 0)
        {
            for (int i = 0; i < parent.childCount; i++)
            {
                Destroy(parent.GetChild(i).gameObject);
            }
        }
    }

    private void mesh()
    {
        int point_number = cloud_data.position.Count;
        int mesh_number = point_number / maxPoint_number;
        int left_point = point_number % maxPoint_number;

        ClearChilds(root_mesh.transform);

        for (int i = 0; i < mesh_number; i++)
        {
            GameObject obj = new GameObject();
            obj.name = i.ToString();
            obj.transform.parent = root_mesh.transform;
            obj.AddComponent<MeshFilter>();
            obj.AddComponent<MeshRenderer>();

            Mesh tempMesh = new Mesh();
            CreateMesh(ref tempMesh, ref cloud_data.position, ref cloud_data.color, i * maxPoint_number, maxPoint_number);
            Material material = new Material(Shader.Find("Custom/VectorShader"));
            obj.GetComponent<MeshFilter>().mesh = tempMesh;
            obj.GetComponent<MeshRenderer>().material = material;
        }
        GameObject objLeft = new GameObject();
        objLeft.name = mesh_number.ToString();
        objLeft.transform.parent = root_mesh.transform;
        objLeft.transform.position = new Vector3(0, 0, 0);
        objLeft.AddComponent<MeshFilter>();
        objLeft.AddComponent<MeshRenderer>();
        Mesh tempMeshLeft = new Mesh();
        CreateMesh(ref tempMeshLeft, ref cloud_data.position, ref cloud_data.color, mesh_number * maxPoint_number, left_point);
        Material materialLeft = new Material(Shader.Find("Custom/VectorShader"));
        objLeft.GetComponent<MeshFilter>().mesh = tempMeshLeft;
        objLeft.GetComponent<MeshRenderer>().material = materialLeft;
    }

    void CreateMesh(ref Mesh mesh, ref List<Vector3> arrayListXYZ, ref List<Color32> arrayListRGB, int beginIndex, int pointsNum)
    {
        Vector3[] points = new Vector3[pointsNum];
        Color[] colors = new Color[pointsNum];
        int[] indecies = new int[pointsNum];
        for (int i = 0; i < pointsNum; ++i)
        {
            points[i] = (Vector3)arrayListXYZ[beginIndex + i];
            indecies[i] = i;
            colors[i] = (Color)arrayListRGB[beginIndex + i];
        }

        mesh.vertices = points;
        mesh.colors = colors;
        mesh.SetIndices(indecies, MeshTopology.Points, 0);

    }
    #endregion

    void Update()
    {
        time_inter += Time.deltaTime;
        if (time_inter >= 1f / FPS)
        {
            if (frame > frame_number)
            {
                frame = frame % 300;
            }
            update_file_list(frame);
            play();
            time_inter = 0;
            frame++;
        }

    }
}
