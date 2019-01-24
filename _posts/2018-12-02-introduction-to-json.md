---
layout: post
title: "JSON 入门笔记"
description: ""
category:
tags:
---

继续读 O'Reilly 的入门系列图书，这次是 Lindsay Bassett 的 Introduction to JavaScript Object Notation: A To-the-point Guide to JSON，非常薄，只有一百页出头。本文是我对这本书的简单笔记，信息量不多。

JSON = JavaScript Object Notation，是一种数据交换格式。

虽然 JSON 名字与 JavaScript 有关，它实际上不依赖任何语言，只是基于 JavaScript 的对象字面量的形式。

JSON 使用键值对表示属性数据，每个键是一个属性名称，值是相应的字面量，也可以是数组或者嵌套的对象。

例如生活中如何描述一双鞋，写成规范的 JSON 格式（双引号、冒号、逗号、花括号）
```JavaScript
{
    "brand": "Crocs",
    "color": "pink",
    "size": 9,
    "hasLaces": false,
    "laceColor": null
}
```

包含特殊字符的字符串，注意做转义处理。
```JavaScript
{
    "story": "Say \"Hi!\" to your cat \\n every day."
}
```

除了基本的语法检查以外，对数据类型、范围的一致性检查也很重要，即 JSON Schema 验证
```JavaScript
{
    "$schema": "http://json-schema.org/draft-04/schema#",
    "title": "Shoes",
    "properties": {
        "brand": {
            "type": "string",
            "minLength": 3,
            "maxLength": 20
            },
        "size": {
            "type": "number", 
            "description": "UK shoe size",
            "minimum": 0
            },
        "hasLaces": {"type": "boolean"},
        "description": {"type": "string"}
    },
    "required": ["brand", "size", "hasLaces"]
}
```
定义这样的 Schema 之后，就可以确认一段 JSON 是否符合它的规范。

下面是一些安全相关事项。因为不做 web 开发，这部分只是简单记录，没有深入探究。

如果网站使用 JSON 来传递敏感信息，有可能被 Cross-Site Request Forgery (CSRF) 盗取利用。比如银行网站使用 top-level JSON array，那么它既是有效的 JSON，又是 JavaScript 脚本，在登陆银行网站之后打开钓鱼网站，在 js 上做手脚盗取这段信息
```JavaScript
[
    {"user": "bobbarker"},
    {"phone": "555-555-5555"}
]
```
银行网站应该采取更安全的做法，使它只是一个 JSON 对象而不是有效的 JavaScript 脚本，并且只允许 POST 而禁止 GET 直连
```JavaScript
{
    "info": [
        {"user": "bobbarker"},
        {"phone": "555-555-5555"}
    ]
}
```

要想在 JavaScript 中将 JSON 字符串转化为 js 对象，不应该使用可能执行恶意代码的危险的 `eval("(" + jsonString + ")")`，而应该使用更安全的 `JSON.parse(jsonString)`，避免注入攻击。此外还要注意 JSON 字符串是否包含 HTML。

在 NoSQL 数据库中，document store 是以 XML 或 JSON 的格式存储数据的，例如 CouchDB 的每一条记录是一条 JSON 文档，使用 HTTP API 和 JavaScript 查询。

书中还介绍了 XMLHttpRequest、jQuery、AngularJS 及其服务器端如何处理 JSON 等内容，与我关系不大所以略过。

作为 Python 用户，我使用 JSON 主要是做配置文件。下面将 JSON 与 INI、XML 以及书中没提到的 YAML 格式进行比较

- 可读性：YAML >= INI > JSON >> XML
- 对于复杂的数据的灵活表示，例如列表和嵌套：YAML > JSON >> XML >> INI
- YAML 支持注释，JSON 不支持
- 一般来说 YAML 用于配置，JSON 更多用于数据交换
- JSON 更简单，容易解析，也更快
