---
layout: post
title: "Useful command line commands"
date: "2019-08-07 08:46:58 +0100"
---

This post will be updated regularly with command line things that I have found useful. ```man``` is very comprehensive and so I wanted to note the small number of options that I typically use.

### Find all python files under a directory (recursively) containing the word global

```grep -r global * --include \*.py --color=always```
The colouring option is very handy here for large output.

### Count lines in anything
``` wc -l```
This is the wordcount command. the -l option counts lines. Pipe it the output of a grep to see how many hits you got.

### Tail a log file to see any changes as they happen
``` tail -f myfile.log```

### Human readable file sizes
```ls -h```
