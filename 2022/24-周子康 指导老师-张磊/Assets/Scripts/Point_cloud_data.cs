using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;


public class Point_cloud_data
{
    public List<Vector3> position;
    public List<Color32> color;

    public Point_cloud_data()
    {
        position = new List<Vector3>();
        color = new List<Color32>();
    }

    public void Append_point( float x, float y, float z,
                                byte r, byte g, byte b, byte a)
    {
        position.Add(new Vector3(x, y, z));
        color.Add(new Color32(r, g, b, a));
    }

    public void Append_cloud(Point_cloud_data point_b)
    {
        this.position = position.Concat(point_b.position).ToList<Vector3>();
        this.color = color.Concat(point_b.color).ToList<Color32>();
    }

    public void Clear_cloud()
    {
        position.Clear();
        color.Clear();
    }
}
