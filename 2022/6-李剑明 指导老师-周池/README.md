# boreas-scheduler-improve

## Deploy Boreas scheduler

### Deploy from YAML file

+ ```shell
  $ kubectl create -f scheduler-improve.yaml
  ```

### Deploy from source code

首先安装 Buildah：

+ ```shell
  $ sudo apt-get install buildah
  ```

开启本地的镜像仓库，拉取 registry 镜像并运行：

+ ```shell
  $ registryctr=$(buildah from docker.io/library/registry:2)
  ```

+ ```shell
  $ buildah run --net=host $registryctr /entrypoint.sh /etc/docker/registry/config.yml
  ```

运行部署脚本：

```shell
$ bash build/deploy-locally
```

该脚本包含如下三条命令：

1. 根据 Dockerfile 打包源码为镜像：

   ```shell
   $ buildah bud -t boreas-scheduler-improve:local .
   ```

2. 将镜像推送到本地仓库：

   ```shell
   $ buildah push --tls-verify=false boreas-scheduler-improve docker://localhost:5000/boreas-scheduler-improve:local
   ```

3. 启动调度器，scheduler-local.yaml 会从本地仓库中拉取Boreas镜像：

   ```shell
   $ kubectl create -f scheduler-local.yaml
   ```

## Experiment

运行原始的Boreas调度器和改进后的Boreas调度器：

+ ```shell
  $ kubectl create -f scheduler.yaml
  ```

+ ```shell
  $ kubectl create -f scheduler-improve.yaml
  ```

注意：两个调度器同时只能存在一个

### Experiment ①

+ ```shell
  $ kubectl create -f myTest1-default.yaml
  ```

+ ```shell
  $ kubectl create -f myTest1-boreas.yaml
  ```

对比Kubernetes默认调度器和Boreas调度器的调度效果

kube-scheduler 会有其中一个Pod调度失败，而Boreas则是所有Pod都调度成功

### Experiment ②

+ ```shell
  $ kubectl create -f myTest2-1.yaml
  ```

+ ```shell
  $ kubectl create -f myTest2-2.yaml
  ```

先运行 myTest2-1.yaml，等进入下一个调度周期后，再运行 myTest2-2.yaml；对比改进前后Boreas调度器的调度效果

运行 myTest2-2.yaml 时，改进前的Boreas调度失败，而改进后的Boreas能令所有Pod都调度成功















