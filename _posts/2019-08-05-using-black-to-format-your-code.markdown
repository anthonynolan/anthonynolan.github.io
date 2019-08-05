---
layout: post
title:  "Using black to format your python"
categories: python
---

Keeping your code tidy matters in any language. Python enforces some of this with indents replacing braces in code, but still leaves a lot to individual developer taste and editor setup. If you are working with other developers differing formatting in your source code can lead to annoying diffs on checkin and additional cognitive load as you try to understand someone elses code. My experience of 'ignore white space' in these tools has not been good.

[Black](https://black.readthedocs.io/en/stable/) is a code formatter that deals with this problem. It is opinionated about how to format python. You can provide a very small list of command line options, but mostly you are expected to just use it as it is. The only exception I would make here is to specify a different line length to the default. My reasoning is to get it to match the default used in [Atom](https://atom.io/) - my editor of choice.

```black --line-length 79 my_file.py```

Getting all developers in a project to agree to this switch is pretty straightforward - put the black command with your line length choice in source control for all to use.

Once run, black just makes the changes to your source. So run it before check-in and you are good. If you would like to see what black is going to do first just run it with the ```--diff option```. It will echo the proposed changes, but leave your source alone.
