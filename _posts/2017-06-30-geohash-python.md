---
layout: post
title: "Geohash 原理及 Python 实现"
description: ""
category:
tags:
---

> 本文代码见 [github](https://github.com/sighsmile/geohash)。内容参考维基百科 [Geohash](https://en.wikipedia.org/wiki/Geohash)

Geohash 是一种地理编码系统，可以把经纬度坐标转换为短字符串，原理是用 Z-order curve 填充平面。在给定的精度下，相同的字符串对应于相同的区域，相同前缀的字符串对应于相邻近的区域。可以利用这一点，快速粗略搜索给定点的周围区域。但是注意，逆命题并不成立，相邻的区域未必有相同的前缀，这是因为 Z-order curve 有突变。

以经纬度 (39.9078, 116.3977) 为例，相应的 12 位编码为 `wx4g08ypycpg`，5 位编码为 `wx4g0`。下面以此为例，演示 geohash 原理及实现。


第一步，base32 解码，将字符串转为二进制串。这里的 base32 编码字符集为阿拉伯数字和英文字母，去掉 a, i, l, o 以免混淆。

`wx4g0` 解码得到 `11100 11101 00100 01111 00000`。

```
BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'  # 0-9 and b-z
CHARMAP = {c: i for i, c in enumerate(BASE32)}

def _char2bits(c):
    """
    convert char c from base32 to 5-bit-string
    this is slower than bit operations
    and is only meant to demonstrate the algorithm
    """
    return "{0:05b}".format(CHARMAP[c])

def _geohash2bits(geohash):
    """
    decode from base32 using CHARMAP
    return a bits string
    """
    bits = ''.join([_char2bits(c) for c in geohash])
    return bits    
```

第二步，将二进制串拆分为经度和纬度两个二进制串。从左向右，下标从 0 开始，偶数位是经度，奇数位是纬度。换言之，最高位是经度，次高位是纬度，两者交替。

`11100 11101 00100 01111 00000` 拆分为经度 `1101001011000` 和纬度 `101110001100`。

第三步，每个二进制串，对应于一系列二分查找结果。纬度从 (-90, 90) 开始，经度从 (-180, 180) 开始，不断二分。如果坐标位于中点的右侧，相应的二进制位就是 1，反之为 0。最终返回的坐标值是最后一次二分区间的中点，误差是该区间的一半。

以纬度 `101110001100` 为例：首位是 1，表明坐标位于 (-90, 90) 的中点 0 的右侧，即位于 (0, 90)；次高位是 0，表明坐标位于 (0, 90) 的中点 45 的左侧，即 (0, 45)；以此类推，具体数值如下：

```
b lo hi mid
1 -90 90 0.0
0 0.0 90 45.0
1 0.0 45.0 22.5
1 22.5 45.0 33.75
1 33.75 45.0 39.375
0 39.375 45.0 42.1875
0 39.375 42.1875 40.78125
0 39.375 40.78125 40.078125
1 39.375 40.078125 39.7265625
1 39.7265625 40.078125 39.90234375
0 39.90234375 40.078125 39.990234375
0 39.90234375 39.990234375 39.9462890625
- 39.90234375 39.9462890625 39.92431640625
```

最终，由纬度的二进制串 `101110001100` 得到纬度坐标为 39.92431640625，误差为 0.02197265625。
同理，经度坐标为 116.38916015625，误差为 0.02197265625。

```
import itertools

def _bits2coordinate(bits, lo, hi):
    """
    decode bits (iteratable) into coordinate value and error
    """
    for b in bits:
        mid = (lo + hi) / 2
        if b == '1':
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2, (hi - lo) / 2

def _decode_val_err(geohash):
    """
    decode geohash string to coordinates and errors
    return longitude value, latitude value, longitude error, latitude error
    this uses string conversions which are slower than bit operations
    and is only meant to demonstrate the algorithm
    """
    bits = _geohash2bits(geohash)
    lat_bits = itertools.islice(bits, 1, None, 2)
    lat_val, lat_err = _bits2coordinate(lat_bits, -90, 90)
    lng_bits = itertools.islice(bits, 0, None, 2)
    lng_val, lng_err = _bits2coordinate(lng_bits, -180, 180)
    return lat_val, lng_val, lat_err, lng_err
```

注意，舍入误差不应当超过上述计算误差，否则其编码将落在错误的区间。例如上述经纬度 (39.92431640625 ± 0.02197265625, 116.38916015625 ± 0.02197265625)，可以近似为 (39.92, 116.39)，其编码仍然为 `wx4g0`。但如果进一步舍入到 (39.9, 116.4)，编码就变成了 `wx4fb`。

由此不难意识到，尽管相同前缀的地点是相邻近的，但是邻近的点未必有相同前缀，在边界附近会出现突变。不妨考虑一下极端情形，紧邻赤道两侧的两个点，对应的二进制串首位分别是 0 和 1，从而两者的公共前缀长度为零。出现这一现象的原因是 Z-order curve 本身的性质。事实上也可以用其他的平面填充曲线来完成这一变换，但是实现起来会更复杂。

此外，由于经纬度坐标投影非线性，相同的经纬度数误差在赤道和两极附近所对应的距离误差也会有很大的差异。

上述代码实现，意在逐步演示 geohash 的原理，效率并不是最优的。在实践中，有必要把字符串操作改成效率更高的位操作，如下所示：

```
def decode_val_err(geohash):
    """
    decode geohash string to coordinates and errors
    return longitude value, latitude value, longitude error, latitude error
    this uses bit operations
    """

    lat_lo, lat_hi = -90, 90
    lng_lo, lng_hi = -180, 180
    is_lng = True
    masks = [16, 8, 4, 2, 1]  # use bit operation to make base32 convert fast

    for c in geohash:
        d = CHARMAP[c]
        for mask in masks:
            if is_lng:
                mid = (lng_lo + lng_hi) / 2
                if d & mask:
                    lng_lo = mid
                else:
                    lng_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if d & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            is_lng = not is_lng

    lat_val = (lat_lo + lat_hi) / 2
    lng_val = (lng_lo + lng_hi) / 2
    lat_err = (lat_hi - lat_lo) / 2
    lng_err = (lng_hi - lng_lo) / 2

    return lat_val, lng_val, lat_err, lng_err
```
