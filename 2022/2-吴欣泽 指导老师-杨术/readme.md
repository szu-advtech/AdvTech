环境p4_environmen文件夹中还缺少了bmv2交换机（behavioral-model）和p4c-bm编译器，请自行下载，别忘了修改env.sh
有问题欢迎邮件2210274022@email.szu.edu.cn讨论

## Test
The method to start the p4 switch is to set up a topology on **mininet** where we run the switch.

1. Open a terminal and enter the directory `hash_version`
2. Run `run_demo.sh` in terminal to start the switch (the data plane of NetHCF) on the topology defined in `topo.py`，顺利的话你会进入mininet
3. mininet基本的使用在'/home/myp4/P4/tutorials/exercises'，相信你已经看过这些练习了，更多的使用可以参考官方文档 according to [mininet walkthrough](http://mininet.org/walkthrough/#run-a-simple-web-server-and-client). 这里我们xterm h1 h2 来打开客户机终端
4. Run `controller.py` 来统计不正常包数，控制NetHCF的过滤和学习状态。状态转换阈值可以设置
5. 你可以在cmd中直接输入`/home/myp4/Downloads/Anti-spoof/p4_environment/behavioral-model/targets/simple_switch/sswitch_CLI  hop_count.json 22223`来运行CLI，他有很多指令，比如register_read 、register_read ，在control.py文件中使用了这些指令。
6. Then, just try!

## Source Code
`hop_count.p4`  This is the p4 source code.

`includes/headers.p4`  This is the p4 code which defines the header used in NetHCF.

`includes/parser.p4`  This is the p4 code which defines the parser graph of NetHCF.

`topo.py` This script will set up the topology of this model and starts the CLI of p4 switch.

`run_demo.sh` Start the data plane on the topology defined in `topo.py` without log.  

`commands.txt` There are table entries here, which will be loaded into the swtich by `topo.py`. You can also add the entries manually through CLI.

`env.sh` Set p4 related environment variables.

`cleanup.sh` Clean up the environment such as the pcap file and accessed webpage.

`controller.py` NetHCF's conrol plane program running on CPU, masters the running state of switch and matains a global view.

`controller_update.py` 双倍寄存器表示时间更新,每过一段时间要用缓存表覆盖原表，但是这里并没有实现，2^23太大了，跑起来很慢，我就写了一个记录动过数据的，然后把动过的地方给覆盖过去，但是没动的地方用0覆盖我没有办法实现。这个终归是要结合硬件来实现的。

send和receive都做了监听和发送，在xterm中运行，receive是接收端，send是发送端。写的比较冗余，不过也挺易懂的，可根据测试需要更改

`hop_count.json` P4编译后的产物，判断自己修改的P4代码有没有错，其实可以看这个json文件有没有更新内容，编译错误他是不会更新的。当然终端在一开始也会有报错。



