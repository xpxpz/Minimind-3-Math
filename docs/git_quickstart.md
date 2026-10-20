# 本项目的 Git 与 VS Code 操作入门

Git 保存项目的版本历史；GitHub 存放上传后的仓库。VS Code 的“源代码管理”和终端操作的是同一份 Git 仓库。

## 四个操作

| 操作 | 含义 | 是否上传 GitHub |
| --- | --- | --- |
| 保存 Save | 将编辑内容写入本地文件 | 否 |
| 暂存 Stage | 选出这次准备提交的改动 | 否 |
| 提交 Commit | 在本地保存一个带说明的版本快照 | 否 |
| 推送 Push | 将本地提交上传到 GitHub | 是 |

日常顺序：保存文件 → 查看差异 → 暂存 → 写说明并提交 → 推送。

## 在 VS Code 中操作

1. 打开本项目文件夹。点击左侧“源代码管理”图标，或按 **Control + Shift + G**（Mac 这里是 Control）。
2. 在“更改 / Changes”中点击文件，查看与上一版本的差异。`U` 表示新文件，`M` 表示已修改，`D` 表示删除。
3. 文件右侧点击 **+**，将它加入“暂存的更改 / Staged Changes”。确认全部文件都需要提交时，可点“更改”标题旁的 **+** 暂存全部。
4. 在上方消息框填写本次改动，例如“补充数学数据校验”。点击 **提交 / Commit**。
5. 点击 **… → 推送 / Push**，将提交上传。也可以使用“同步更改 / Sync Changes”，它会先拉取远端提交再推送。
6. 打开 GitHub 仓库并刷新，确认文件和最新提交说明出现。页面左上方的分支选择应与自己推送的分支一致。

如果某次改动还不想提交，在“暂存的更改”旁点 **−** 取消暂存，文件编辑仍然保留。“放弃更改 / Discard Changes”会丢弃文件改动，与取消暂存不同。

左下角显示当前分支。`main` 是仓库主页默认展示的主分支；`codex/math-posttraining` 是这次开发使用的分支。分支是同一仓库里的不同开发线，提交到开发分支不会自动改变 main。

## 使用终端完成相同操作

在 VS Code 菜单“终端 → 新建终端”打开终端，确认位于本项目根目录。下面命令用于以后修改文件之后；刚刚同步完成且没有改动时，无需重复提交。

```bash
# 看看哪些文件发生变化，以及当前分支
git status

# 将某个文件加入本次提交；替换成自己实际修改的文件
git add README.md

# 保存到本地版本历史
git commit -m "docs: update README"

# 上传当前分支到它关联的 GitHub 分支
git push
```

如果检查后确认所有改动都要提交，可以用 `git add .` 替代单个文件的 add；它会遵守 .gitignore。暂存后用 `git diff --cached --stat` 查看这次提交包含哪些文件。

`git pull --ff-only` 用于拉取远端的新提交；若提示分支分叉或冲突，先查看情况，不要直接强制推送。遇到错误保留完整提示，再判断如何处理。

## 本项目提交哪些文件

提交：源码、测试、可复现配置、README/文档、依赖锁文件、数据哈希清单和少量可审查的实验记录。

默认忽略：模型权重和适配器、artifacts/、虚拟环境、生成的大批数据、完整运行日志、configs/local.json 里的本机模型路径及 .env。需要分享权重时，在模型托管平台上传，再在 README 中链接。

.gitignore 对尚未被 Git 跟踪的文件生效；已经提交的文件不会因为后来添加忽略规则就自动移出历史。因此每次提交前仍要看暂存文件列表。

## 判断操作是否完成

- “源代码管理”中更改列表为空：本地文件与当前提交一致。
- 本地提交之后仍需 Push；只有 Commit 并不表示已上传。
- `git status` 显示与 origin/main 一致，且 GitHub 上能看到对应提交：本地主分支与远端已同步。
- 被 .gitignore 排除的文件仍保存在电脑上；列表中不显示不代表文件被删除。

参考：[VS Code 暂存与提交](https://code.visualstudio.com/docs/sourcecontrol/staging-commits)、[VS Code 远程仓库与同步](https://code.visualstudio.com/docs/sourcecontrol/repos-remotes)、[GitHub 的 Git 介绍](https://docs.github.com/en/get-started/using-git/about-git)。
