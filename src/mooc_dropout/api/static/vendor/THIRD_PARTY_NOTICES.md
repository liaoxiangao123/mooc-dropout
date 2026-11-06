# 第三方组件声明（`api/static/vendor/`）

本目录存放 **Swagger UI** 的前端静态资源，用于让 `/docs` 在**完全离线**时也能正常渲染。
FastAPI 默认的 `/docs` 从 `cdn.jsdelivr.net` 拉取这两个文件，现场断网或 CDN 不可达时页面会一片空白，
因此改为随包内置。

## 组件

| 项目 | 内容 |
|---|---|
| 名称 | Swagger UI（`swagger-ui-dist` 发行版） |
| 版本 | 5.11.0（bundle 内含 `5.11.0` 标记；精确字节以下方 SHA-256 为准） |
| 版权 | Copyright 2016-2024 SmartBear Software |
| 许可 | Apache License 2.0，全文见同目录 [`LICENSE-Apache-2.0.txt`](LICENSE-Apache-2.0.txt) |
| 上游 | https://github.com/swagger-api/swagger-ui |
| 取得方式 | 从 `cdn.bootcdn.net/ajax/libs/swagger-ui/5.11.0/` 下载原样保存，**未做任何修改** |

## 文件与指纹

| 文件 | 字节 | SHA-256 |
|---|---|---|
| `swagger-ui-bundle.js` | 1 400 489 | `aba894c6d9e1c56e6568fe48a7d3f6913992925826531ac87f1ebc2b03128aab` |
| `swagger-ui.css` | 152 041 | `93f1d44a8ee6589e7bc923c1c30e95dab867a0a8f91d2ab58f8d69258cb6aa07` |

复核命令：

```bash
sha256sum src/mooc_dropout/api/static/vendor/swagger-ui-*
```

## 关于 `swagger-ui-bundle.js.LICENSE.txt`

该 bundle 由 webpack 打包，首行注释指向同名的 `.LICENSE.txt`（webpack 汇总的第三方许可清单）。
`swagger-ui-dist` 的 CDN 目录并不提供这个文件（实测返回 403），因此**单独补入 Apache-2.0 全文**
以满足 Apache-2.0 第 4(a) 条「随分发提供许可证副本」的要求。如需完整的三方许可清单，
可从 npm 包 `swagger-ui-dist@5.11.0` 中取出同名文件放入本目录，无需改动代码。

## 升级步骤

1. 从官方发行版取新的 `swagger-ui-bundle.js` / `swagger-ui.css`，覆盖本目录同名文件；
2. 同步更新上表的字节数、SHA-256 与版本号；
3. 跑 `pytest tests/test_docs_ui.py`，确认 `/docs` 仍渲染且引用的是本地路径。

> 缺文件时的行为：`app.py` 会检测本目录是否同时存在 js 与 css，缺任一个就退回 FastAPI 默认的 CDN 版本，
> 页面不会 500，只会重新依赖外网。
