# 配置全局用户信息​​：
git config --global user.name "zhujun3753"
git config --global user.email "zhujun3753@163.com"

git init
git add .
# 提交更改
git commit -m "Initial commit"

git remote add origin git@git@github.com:zhujun3753/imu_pre_py.git
# 修改现有远程仓库 URL
# git remote set-url origin git@git@github.com:zhujun3753/imu_pre_py.git 

检查 SSH 密钥是否存在
dir %USERPROFILE%\.ssh
# 生成 ED25519 密钥（更安全）
ssh-keygen -t ed25519 -C "zhujun3753@163.com"
# 或者生成 RSA 密钥（兼容性更好）
ssh-keygen -t rsa -b 4096 -C "zhujun3753@163.com"
# 检查本地分支状态
git branch
# master重命名分支为 main
git branch -M master main
# 首次推送需要设置上游分支
git push -u origin main

# 后续推送只需
git push