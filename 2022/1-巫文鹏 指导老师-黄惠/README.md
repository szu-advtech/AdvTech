# Iterative Poisson Surface Reconstruction (iPSR) for Unoriented Points

Iterative Poisson Surface Reconstruction (iPSR) for Unoriented Points  
[Fei Hou](https://lcs.ios.ac.cn/~houf/),    Chiyu Wang,    [Wencheng Wang](https://lcs.ios.ac.cn/~whn/),    [Hong Qin](https://www3.cs.stonybrook.edu/~qin/),    Chen Qian,    [Ying He](https://personal.ntu.edu.sg/yhe/)  
*ACM Transactions on Graphics, 41, 4, Article 128, 13 pages, 2022. (SIGGRAPH 2022)*

iPSR extends the popular Poisson Surface Reconstruction ([https://github.com/mkazhdan/PoissonRecon](https://github.com/mkazhdan/PoissonRecon)). iPSR has no more need of oriented normals as input, but infers the normals in an iterative manner. It is used to reconstruct surface from only points input.  
[project page](https://lcs.ios.ac.cn/~houf/pages/ipsr/index.html)

## Compilation:
Windows: Open ipsr.vcxproj and use Visual Studio to compile .  
Linux: The code is tested by GCC and Clang with makefile.  

```shell
cd ./ipsr
make
```

GUI: GUI implementation is in iPSRgui/iPSR.exe

## Two way to use

## Run in Terminal

The following command is tested in `Ubuntu` system

### **Example**

After compilation, cd to `\ipsr` first. Run command：

```shell
./ipsr --in ./data/bunny_sparse.ply --mode 1 --out ./Reconed_model/default_bunny_sparse.ply
```

### Recon all input file in different mode

After compilation, cd to `/ipsr` first. 

Move your input file(.ply file to be reconstructed) into `./data/` 

Run `runiPSR.py`

```shell
python runiPSR.py
```

And you can find result in `./Reconed_model` dir.

### The parameter is as follow:

\-\-in &lt;input ply file name&gt;  
The input file should be in .ply format and only 3D point coordinates are needed.

\-\-out &lt;output ply file name&gt;  
The output file name. It should be in .ply format.

### The following is optional

[--mode &lt;Different mode of updating&gt;]
Four different mode:	1-default paper way; 	2-without_area;	3-distance_weighted;	4-ball_query. The default value for this parameter is 1.

\[\-\-iters &lt;maximum number of iterations&gt;\]  
The maximum number of iterations. The default value of this parameter is 10.

\[\-\-pointWeight &lt;interpolation weight&gt;\]  
The pointWeight parameter of screened Poisson surface reconstruction. The default value for this parameter is 10.

\[\-\-depth &lt;reconstruction depth&gt;\]  
The depth parameter of screened Poisson surface reconstruction. It is the maximum possible depth of the octree. The default value of this parameter is 10.

\[\-\-neighbors &lt;number of neighbors&gt;\]  
The number of the closest sample points to search from every reconstructed triangle face. The suggested range is between 10 and 20. The default value of this parameter is 10.

## GUI

This gui is tested in `win64` system

The GUI parameter (Only have `--in` `、--out`、`--mode`、`--iters`、`--neighbors`) is the same as `using in Termnal`

The difference is the `input file path` and `output file path` have to be Absolute path

