#!/bin/bash

# 使用 wget 的版本
for i in {1..17}; do
    num=$(printf "%03d" $i)
    url="tar -xvzf ds116_sub${num}.tgz"
    echo $url
#    wget --progress=bar:force "$url" || echo "下载失败: $url"
done



