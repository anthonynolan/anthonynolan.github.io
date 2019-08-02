---
layout: post
title:  "Writing a precommit hook for git in python"
categories: python
---

Once you initialise a git repo you get a ```.git/hooks``` folder. It contains a full set of sample hooks. To use - and customise - one of these remove the ```.sample``` from the end of the filename. ```prepare-commit-msg``` is my only live one in this repo:

```
applypatch-msg.sample		
pre-push.sample
commit-msg.sample		
pre-rebase.sample
fsmonitor-watchman.sample
pre-receive.sample
post-update.sample		
prepare-commit-msg
pre-applypatch.sample		
update.sample
pre-commit.sample
```

This file is executed as you are preparing your precommit message - in other words writing in the file that git opens with some boilerplate comments. So if you use an argument like -m to provide the comment on the command line this will not execute.

Mostly hooks are written in bash, or perl, but using a #! you can use anything that is installed on your local system. Here is the python content of my ```prepare-commit-msg``` file:

{% highlight python %}
#!/usr/bin/env python3

import sys
# this part opens the commit comment file - it's passed as the first arg by git when you issue commit with no args
f = open(sys.argv[1], 'w')
# put a string in there to get the committer started
f.write("# Comment your commit")
f.close()

# this part is just to show the full arg list
# First is this python file in hooks and second the commit message file
for a in sys.argv:
  print(a)

{% endhighlight %}

If you find yourself doing a lot in hooks like these you might be better using something like jenkins to build a more sophisticated process. A good use here might be to check if your code contains a reference to a ticketting system like jira.
