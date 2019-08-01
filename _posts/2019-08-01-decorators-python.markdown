---
layout: post
title:  "Using decorators in python"
categories: python
---

Decorators allow you to do something before and after the execution of a function. You can even modify parameters before use.

This is that basic usage including arg modification

{% highlight python %}
#!/usr/bin/env python3

def my_deco(func):
    "This is the decorator function"
    def wrapper(*args, **kwargs):
        print('you are before {}'.format(func.__name__))

        #changing one of the keyword args
        kwargs['rating']=10

        # exec the wrapped function and get its return value
        ret = func(*args, **kwargs)
        print('you are after {}'.format(func.__name__))
        return ret
    return wrapper

@my_deco
def another_function(name, rating):
    "The decorated function. Has positional and named args."
    print('another function, param: {}, {}'.format(name, rating))
    return 'I am returning this'

# Call our decorated function with rating 4
val = another_function('anthony', rating=4)
print(val)

{% endhighlight %}

And this is the output

```
you are before another_function
another function, param: anthony, 10
you are after another_function
I am returning this
```

A simple application of this is to time execution of a method and print the time taken.

{% highlight python %}
#!/usr/bin/env python3

from datetime import datetime

def how_long(func):
    def wrapper():
        before = datetime.now()
        func()
        after = datetime.now()
        print('{} seconds'.format(after - before))
    return wrapper

@how_long
def another_function():
    for _ in range(int(1e7)):
        pass

another_function()
{% endhighlight %}
