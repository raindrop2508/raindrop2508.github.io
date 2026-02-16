---
title: "Ubuntu笔记"
date: 2019-06-30T11:00:55+08:00
---

>这篇笔记会根据情况隔段时间更新一次
# 终端操作
>cd+文件夹名  进入文件夹 可以使用TAB键进行自动补全或选择 <br>
>cd   ..( 退出几层就加几个..)  退出到上n层路径  
>ls 显示文件夹中的目录  
>pwd 显示当前路径  
>gedit 使用文本编辑器编辑文件   
>cat 在终端中查看文件（多用做文本文件）  
>sudo  进入管理员模式（超级用户)  
>sudo su 进入根用户  
> nvidia-smi  查看显卡运行状态（A卡没有试过）

在ubuntu中一些文件被写保护，需要更高级权限才能修改，可以在命令前加sudo  
## 复制一个文件到指定目录  

>(1) 在桌面上打开终端，输入sudo su  
>(2) 输入密码，就切换到root用户下  
>(3) 切换到桌面 命令输入为 cd 桌面  
>(4) 然后输入复制命令行 cp -r studio.zip /home/androidstudio  
>(5) 回车 大功告成 可以看看/home/androidstudio目录下是否有studio.zip  

作者：飞奔的小付   
来源：CSDN   
[原文](https://blog.csdn.net/feibendexiaoma/article/details/73739319) 

移动文件（假设现在在该文件的目录下）
>sudo mv xxx.xx 新的位置的路径  

解压文件 
>sudo unzip opencv-3.3.0.zip  

重命名文件 
>sudo cp 原文件名 新文件名  

# 软件安装

软件安装前一般要先更新软件目录
>sudo apt-get update  

升级已有软件
>sudo apt-get upgrade
常用软件安装
>https://www.jianshu.com/p/f44e1ae080a5

下载好网易云音乐安装完成后可能会遇到问题打不开。可以在终端中使用
>sudo netease  TAb(自动补全)  

当下在好软件安装包后，也可以在本地安装

>sudo sh xxx.sh  
>dpkg -i xxx.deb

[google浏览器安装](https://jingyan.baidu.com/article/335530da98061b19cb41c31d.html)  
# 踩过的坑
  
[没有声音解决办法](https://blog.csdn.net/sannerlittle/article/details/77479656)  
具体情况是：可以调节音量，但却没有声音
 这个问题到现在，我仍旧是每次开机仍要打开文章中所说的软件修改一次
ubuntu和win双系统中无法访问win中的磁盘分区  

>sudo ntfsfix /dev/磁盘号  
>如sudo ntfsfix /dev/sda6

[ubuntu中python2和3共存](https://blog.csdn.net/u010801439/article/details/79485914) 
>source activate  [要激活的环境名称]  
>source deactivate  

[如何修复 apt-get update 无法添加新的 CD-ROM 的错误](https://linux.cn/article-5409-1.html)


**侵权即删**