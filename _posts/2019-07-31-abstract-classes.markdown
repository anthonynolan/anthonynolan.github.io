---
layout: post
title:  "Creating Abstract classes in python"
date:   2019-07-31 19:00:41 +0100
categories: python
---

If you want to make a class in python abstract there are a couple of steps.

1. inherit from ABC and
2. mark at least one of the methods as @abstract



{% highlight python %}
#! /usr/bin/env python3

from abc import abstractmethod, ABC

class Parent(ABC):
    @abstractmethod
    def my_method(self):
        pass

    def test(self):
        print('test')

class Child(Parent):
    def __init__(self):
      pass

    def my_method(self):
        print('it is here now')


m = Child()

# method on the parent
m.test()

m.my_method()

# You can't instantiate one of these
# p = Parent()
{% endhighlight %}

You can find all of the details [here](https://docs.python.org/3/library/abc.html)
