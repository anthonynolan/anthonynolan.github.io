---
layout: post
title:  "Type checking your python code with mypy"
categories: python
---

Python is a dynamically typed language. This is one of its strengths. Usually types are obvious and letting the language work them out for your works fine. However sometimes this can be a source of bugs.

Recently on a datascience project I accidentally assigned an array to a variable that had pointed to a dataframe. My subsequent attempt to call a dataframe method on that array gave an error. Basic mistake, but type checking would have made it easier to find.

Python has an optional type checking system. Optional - because running a script which has been annotated for type checking does not do anything differently. You need to use some sort of tool to check your type usage matches your annotations. [Mypy](http://mypy-lang.org/) is one such tool.

This code snippet illustrates simple usage.
{% highlight python %}

# Here the param is annotated as is the return type.
# This function takes a str only and returns an str.
def thing(a: str) -> str:
    print("thing string {}".format(a))
    return "answer"


# This works fine
thing("james")

# This will not:
# thing(1)

{% endhighlight %}

I would not suggest that you use type checking throughout your python work, but if you are building something substantial, more than one developer will be involved, or if you have just come up with a tricky bug - it might be for you.
