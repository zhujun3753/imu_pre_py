# 根据 Dockerfile 里的内容创建镜像
# 当使用Dockerfile构建镜像时，Docker会利用缓存来加速构建过程。具体来说，Docker将Dockerfile中的每条指令视为一个层，并会缓存这些层。如果Dockerfile中的指令没有变化，且之前的构建缓存还在，那么Docker就会重用这些缓存层，而不会重新执行指令（包括下载等操作）。
# docker build -t imu_pre:v1 .

# 启动容器，并根据镜像创建容器挂载目录，容器不会保存，用完即销毁（与运行代码隔一行）

docker run -it --rm `
  -v "D:\SLAM\IMU_pre_python:/app"   `
  -w /app -e MPLBACKEND=Agg imu_pre:v1 bash

# MPLBACKEND=Agg 用于无显示环境下生成并保存图像。

# 镜像信息： Debian 12  ==> Ubuntu基于Debian. Debian更注重稳定性和自由软件；Ubuntu更注重用户友好性和商业支持
# $ cat /etc/os-release
# PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"
# NAME="Debian GNU/Linux"
# VERSION_ID="12"
# VERSION="12 (bookworm)"
# VERSION_CODENAME=bookworm
# ID=debian
# HOME_URL="https://www.debian.org/"
# SUPPORT_URL="https://www.debian.org/support"
# BUG_REPORT_URL="https://bugs.debian.org/"



# docker ps: 列出正在运行的容器。
# docker images: 列出本地已有的镜像。
# docker stop <container>
# docker rm <container>
# docker rmi <image>
# 使用 docker commit命令将容器保存为新镜像：
# docker commit [容器ID或名称] [新镜像名]:[标签]
# 慎用 docker commit​​ 这种方式会保存容器的​​当前状态​​（包括文件修改、环境变量等），但可能导致镜像臃肿。​​推荐优先使用 Dockerfile构建镜像​​（可复现且更规范）。

# 为了 clang-format 但好像没有用到
# $llvm = "C:\Program Files\LLVM\bin"   # 换成你实际的目录
# $env:Path += ";" + $llvm
# clang-format --version
